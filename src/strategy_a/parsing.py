"""Response parsing (LLM_SPEC.md §7).

1. Extract the JSON object: the last fenced ``json`` block if present, otherwise
   the last ``{...}`` object in the text.
2. Parse leniently (trailing commas, single quotes, Python literals).
3. Map labels back to prop_keys. Missing H keys -> null, extra keys dropped; both logged.
4. Values are kept as returned; normalization is done by the evaluation.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n?(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


@dataclass
class ParseResult:
    predictions: dict[str, Any]  # prop_key -> value or None, exactly the H keys
    ok: bool  # a JSON object with at least one H key was found
    events: list[str] = field(default_factory=list)
    parsed: Any = None


def _objects(text: str) -> list[str]:
    """Top-level balanced ``{...}`` spans, ignoring braces inside strings."""
    spans, depth, start, quote, escape = [], 0, None, None, False
    for i, ch in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'" and depth > 0:
            # Only treat ' as a quote when it opens a string, not in words like "don't".
            if ch == '"' or text[i - 1] in "{[,: \n\t":
                quote = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def loads_lenient(s: str) -> Any:
    s = s.strip()
    for candidate in (s, _TRAILING_COMMA.sub(r"\1", s)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    py = re.sub(r"\bnull\b", "None", re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", s)))
    try:
        return ast.literal_eval(_TRAILING_COMMA.sub(r"\1", py))
    except (ValueError, SyntaxError):
        return None


def extract_json(text: str) -> tuple[Any, list[str]]:
    events = []
    fences = _FENCE.findall(text)
    json_fences = [body for lang, body in fences if lang.lower() == "json"]
    candidates = []
    if json_fences:
        candidates.append(("json_fence", json_fences[-1]))
    elif fences:
        candidates.append(("fence", fences[-1][1]))
    objects = _objects(text)
    if objects:
        candidates.append(("last_object", objects[-1]))
    for source, body in candidates:
        obj = loads_lenient(body)
        if isinstance(obj, dict):
            events.append(f"json_from:{source}")
            return obj, events
        inner = _objects(body)  # fence with prose around the object
        if inner:
            obj = loads_lenient(inner[-1])
            if isinstance(obj, dict):
                events.append(f"json_from:{source}_inner")
                return obj, events
    events.append("no_json")
    return None, events


def _norm(key: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(key)).strip().casefold()


def map_predictions(obj: dict, labels: dict[str, str], H: list[dict]) -> ParseResult:
    keys = [h["prop_key"] for h in H]
    lookup: dict[str, str] = {}
    for h in H:  # weakest aliases first, so labels win
        lookup[_norm(h["property_name"])] = h["prop_key"]
        lookup[_norm(h["prop_key"])] = h["prop_key"]
    for label, key in labels.items():
        lookup[_norm(label)] = key
    predictions: dict[str, Any] = {k: None for k in keys}
    seen, events = set(), []
    for raw_key, value in obj.items():
        key = labels.get(raw_key) or lookup.get(_norm(raw_key))
        if key is None:
            events.append(f"extra_key:{raw_key}")
            continue
        if key in seen:
            events.append(f"duplicate_key:{raw_key}")
            continue
        seen.add(key)
        predictions[key] = value
    for k in keys:
        if k not in seen:
            events.append(f"missing_key:{k}")
    return ParseResult(predictions, ok=bool(seen), events=events, parsed=obj)


_CSV_HEADER = re.compile(r"^\W*property\W*,\W*value\W*$", re.I)


def extract_csv(text: str) -> tuple[dict | None, list[str]]:
    """``property,value`` lines (the published prompts' output format).

    The header row marks a valid answer even with no rows ("If no facts can be
    verified, output only the header row"), so header-only is an abstention, not
    a parse error. Without a header, ``key,value`` lines are still read.
    """
    events = []
    fences = _FENCE.findall(text)
    body = fences[-1][1] if fences else text
    if fences:
        events.append("csv_from:fence")
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    header_at = next((i for i, ln in enumerate(lines) if _CSV_HEADER.match(ln)), None)
    if header_at is None:
        events.append("csv_no_header")
        rows = lines
    else:
        rows = lines[header_at + 1 :]
    out = {}
    for ln in rows:
        if "," not in ln or ln.startswith("```"):
            continue
        key, value = ln.split(",", 1)
        key = key.strip().strip("`*\"' ")
        value = value.strip().strip('"').strip()
        if key:
            out[key] = value
    if header_at is None and not out:
        events.append("no_csv")
        return None, events
    return out, events


def parse_response(text: str, labels: dict[str, str], H: list[dict], output: str = "json") -> ParseResult:
    if output == "csv":
        obj, events = extract_csv(text)
        if obj is None:
            return ParseResult({h["prop_key"]: None for h in H}, ok=False, events=events)
        result = map_predictions(obj, labels, H)
        result.events = events + result.events
        result.ok = True  # a CSV answer with a header is valid even if it gives no value (abstention)
        return result
    obj, events = extract_json(text)
    if obj is None:
        return ParseResult({h["prop_key"]: None for h in H}, ok=False, events=events)
    result = map_predictions(obj, labels, H)
    result.events = events + result.events
    if not result.ok:
        result.events.append("no_target_keys")
    return result


REPAIR_INSTRUCTION = (
    "Your answer could not be parsed. Reply with only a valid JSON object with exactly these keys, "
    "using null for values you do not know:\n{output_format}"
)
REPAIR_INSTRUCTION_CSV = (
    "Your answer could not be parsed. Reply with only CSV with header `property,value` and one line per "
    "property you are confident about, among: {properties}. If no facts can be verified, output only the header row."
)


def repair_instruction(output: str, labels: dict[str, str]) -> str:
    if output == "csv":
        return REPAIR_INSTRUCTION_CSV.replace("{properties}", ",".join(labels))
    return REPAIR_INSTRUCTION.replace(
        "{output_format}", json.dumps({label: "<value or null>" for label in labels}, ensure_ascii=False)
    )

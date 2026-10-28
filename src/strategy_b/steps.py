"""The three steps of one agent run, as plain functions (AGENT_SPEC.md §5, §8).

They hold the logic and the validation layer; ``graph.py`` only wires them with LangGraph.
"""

from __future__ import annotations

import json
import re
from typing import Any

from strategy_a.parsing import _norm, extract_json
from strategy_a.tasks import Task
from strategy_a.templates import Template, fill, load_template, path_label, render_known, render_targets, target_labels

from .search import SearchResult, is_blocked

PLAN_ALLOWED = {"node_type", "known_properties", "target_properties", "graph_context", "max_queries"}
SYNTH_ALLOWED = PLAN_ALLOWED | {"evidence", "output_format"}
PLAN_REQUIRED = ({"known_properties"}, {"target_properties"})
SYNTH_REQUIRED = ({"known_properties"}, {"target_properties"}, {"evidence"}, {"output_format"})
IDENTIFYING = ("name", "iata", "icao", "code", "city", "country")


def load_plan_template(path) -> Template:
    return load_template(path, PLAN_ALLOWED, PLAN_REQUIRED)


def load_synth_template(path) -> Template:
    return load_template(path, SYNTH_ALLOWED, SYNTH_REQUIRED)


# Rendering ----------------------------------------------------------------------


def render_graph_context(task: Task, with_context: bool) -> str:
    """The task's context entries (published format ``[kind, path, type, property, value]``), or nothing."""
    if not with_context:
        return ""
    entries = task.meta.get("context")
    entries = json.loads(entries) if isinstance(entries, str) and entries else (entries or [])
    if not entries:
        return ""
    lines = []
    for e in entries:
        kind, path, etype, prop, value = (list(e) + [None] * 5)[:5]
        via = " → ".join(path) if isinstance(path, list) else str(path)
        lines.append(f"- {etype} ({kind} via {via}) {prop}: {value}")
    return "Related elements in the graph:\n" + "\n".join(lines)


def _values(task: Task, labels: dict[str, str], with_context: bool, max_queries: int) -> dict[str, str]:
    return {
        "node_type": task.node_type,
        "known_properties": render_known(task.G, "list"),
        "target_properties": render_targets(labels),
        "graph_context": render_graph_context(task, with_context),
        "max_queries": str(max_queries),
    }


def render_plan(tpl: Template, task: Task, with_context: bool, max_queries: int) -> tuple[str | None, str, dict]:
    labels = target_labels(task.H)
    values = _values(task, labels, with_context, max_queries)
    return (fill(tpl.system, values).strip() or None), _tidy(fill(tpl.user, values)), labels


def synth_output_format(labels: dict[str, str]) -> str:
    entry = {"value": "<value or null>", "unit": "<unit or null>", "sources": ["<evidence numbers>"]}
    return json.dumps({"predictions": {label: entry for label in labels}}, ensure_ascii=False)


def render_synthesis(tpl: Template, task: Task, with_context: bool, evidence: list[dict]) -> tuple[str | None, str]:
    labels = target_labels(task.H)
    values = _values(task, labels, with_context, 0)
    values["evidence"] = render_evidence(evidence)
    values["output_format"] = synth_output_format(labels)
    return (fill(tpl.system, values).strip() or None), _tidy(fill(tpl.user, values))


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()  # an empty {graph_context} leaves blank lines


# Plan -------------------------------------------------------------------------


def identifying_text(task: Task) -> str:
    """Values of G that name the entity (name, codes, city, country), for default queries."""
    picked = [str(g["value"]) for g in task.G if g["property_name"] in IDENTIFYING and g.get("value") is not None]
    if not picked:
        picked = [str(g["value"]) for g in task.G[:2] if g.get("value") is not None]
    return " ".join(dict.fromkeys(picked))


def _target_lookup(task: Task, labels: dict[str, str]) -> dict[str, str]:
    lookup = {_norm(h["property_name"]): h["prop_key"] for h in task.H}
    lookup.update({_norm(h["prop_key"]): h["prop_key"] for h in task.H})
    lookup.update({_norm(label): key for label, key in labels.items()})
    return lookup


def parse_plan(text: str, task: Task, labels: dict[str, str], max_queries: int) -> tuple[list[dict], list[str], bool]:
    """Queries ``[{id, query, targets: [prop_key]}]``, validation events, and whether the plan was usable.

    §8: every H property must be targeted (a default query is added otherwise); at most
    ``max_queries`` queries from the model (extra ones are dropped and logged).
    """
    events: list[str] = []
    obj, ev = extract_json(text)
    events += ev
    raw_queries = None
    if isinstance(obj, dict):
        raw_queries = obj.get("queries")
    if raw_queries is None:
        stripped = text.strip()
        if stripped.startswith("["):
            try:
                raw_queries = json.loads(stripped)
            except json.JSONDecodeError:
                raw_queries = None
    if not isinstance(raw_queries, list):
        events.append("plan_no_queries")
        return [], events, False
    lookup = _target_lookup(task, labels)
    queries, seen = [], set()
    for i, q in enumerate(raw_queries):
        if isinstance(q, str):
            q = {"query": q, "targets": []}
        if not isinstance(q, dict) or not str(q.get("query") or "").strip():
            events.append(f"plan_bad_query:{i}")
            continue
        query = " ".join(str(q["query"]).split())
        if query.casefold() in seen:
            events.append(f"plan_duplicate_query:{query}")
            continue
        targets = q.get("targets") or []
        targets = [targets] if isinstance(targets, str) else targets
        mapped = []
        for t in targets:
            key = lookup.get(_norm(t))
            if key is None:
                events.append(f"plan_unknown_target:{t}")
            elif key not in mapped:
                mapped.append(key)
        seen.add(query.casefold())
        queries.append({"id": f"q{len(queries) + 1}", "query": query, "targets": mapped, "source": "model"})
    if len(queries) > max_queries:
        events.append(f"plan_truncated:{len(queries)}>{max_queries}")
        queries = queries[:max_queries]
    if not queries:
        events.append("plan_empty")
        return [], events, False
    covered = {k for q in queries for k in q["targets"]}
    ident = identifying_text(task)
    for h in task.H:
        if h["prop_key"] not in covered:
            label = next(lb for lb, k in labels.items() if k == h["prop_key"])
            query = f"{ident} {task.node_type} {label}".strip()
            queries.append({"id": f"q{len(queries) + 1}", "query": query, "targets": [h["prop_key"]], "source": "default"})
            events.append(f"plan_default_query:{h['prop_key']}")
    return queries, events, True


PLAN_REPAIR = (
    'Your answer could not be used. Reply with only a JSON object {{"queries": [{{"id": "q1", "query": "...", '
    '"targets": ["<property>"]}}]}} with at most {max_queries} queries covering: {targets}.'
)


# Evidence -----------------------------------------------------------------------


def build_evidence(
    results: dict[str, list[SearchResult]],
    blocked_domains: list[str],
    max_tokens: int,
    count_tokens,
    max_chars: int,
) -> tuple[list[dict], list[dict], list[str]]:
    """Numbered evidence for synthesis, all results (with ``blocked``) for the log, and events.

    Blocked domains are filtered here as well as in the search request (§7). Results are
    deduplicated by URL; the evidence keeps the best result of every query first, then the
    others by score, within ``max_tokens`` (§5.2).
    """
    events: list[str] = []
    logged: list[dict] = []
    by_url: dict[str, SearchResult] = {}
    per_query: dict[str, list[SearchResult]] = {}
    for qid, items in results.items():
        for r in items:
            r.query_id = qid
            r.blocked = is_blocked(r.url, blocked_domains)
            logged.append(r.as_dict())
            if r.blocked:
                events.append(f"leak_filtered:{r.url}")
                continue
            if not r.url or r.url in by_url:
                continue
            by_url[r.url] = r
            per_query.setdefault(qid, []).append(r)

    def score(r):
        return r.score if r.score is not None else 0.0

    first = [max(items, key=score) for items in per_query.values() if items]
    rest = sorted((r for r in by_url.values() if r not in first), key=score, reverse=True)
    evidence, used = [], 0
    for r in first + rest:
        content = " ".join((r.content or "").split())[:max_chars]
        entry = {"id": len(evidence) + 1, "url": r.url, "title": r.title, "content": content,
                 "score": r.score, "query_id": r.query_id}
        cost = count_tokens(render_evidence([entry]))
        if evidence and used + cost > max_tokens:
            events.append(f"evidence_truncated_at:{len(evidence)}")
            break
        evidence.append(entry)
        used += cost
    return evidence, logged, events


def render_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(no evidence was found)"
    return "\n\n".join(f"[{e['id']}] {e['title']}\nURL: {e['url']}\n{e['content']}" for e in evidence)


# Synthesis ----------------------------------------------------------------------

_NULLS = {"", "null", "none", "unknown", "n/a", "na", "not found", "not available"}


def _is_null(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().casefold() in _NULLS)


def _sources(raw) -> list:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for s in raw:
        nums = re.findall(r"\d+", str(s)) if not isinstance(s, (int, float)) else [int(s)]
        out += [int(n) for n in nums]
    return out


def flatten(value, unit) -> Any:
    """§5.3: the value with its unit appended when a unit is given and not already in the value."""
    if _is_null(value):
        return None
    if _is_null(unit) or str(unit).strip().casefold() in str(value).casefold():
        return value
    return f"{value} {str(unit).strip()}"


def parse_synthesis(text: str, task: Task, labels: dict[str, str], evidence: list[dict], mode: str):
    """``predictions {prop_key: value|None}``, ``evidence {prop_key: [url]}``, events, ok.

    §8: exactly the H keys; in ``evidence_only`` mode a value without a valid citation
    becomes null (``uncited_value``); a citation of a blocked source is impossible because
    blocked results never reach the evidence, but is checked anyway (``leak_blocked``).
    """
    events: list[str] = []
    obj, ev = extract_json(text)
    events += ev
    predictions = {h["prop_key"]: None for h in task.H}
    cites: dict[str, list[str]] = {}
    if not isinstance(obj, dict):
        return predictions, cites, events + ["synthesis_no_json"], False
    body = obj.get("predictions", obj)
    if not isinstance(body, dict):
        return predictions, cites, events + ["synthesis_bad_schema"], False
    lookup = _target_lookup(task, labels)
    by_id = {e["id"]: e for e in evidence}
    matched = 0
    for raw_key, entry in body.items():
        key = lookup.get(_norm(raw_key))
        if key is None:
            events.append(f"extra_key:{raw_key}")
            continue
        matched += 1
        if isinstance(entry, dict):
            value, unit, sources = entry.get("value"), entry.get("unit"), _sources(entry.get("sources"))
        else:
            value, unit, sources = entry, None, []
        value = flatten(value, unit)
        valid = [s for s in sources if s in by_id]
        if len(valid) < len(sources):
            events.append(f"invalid_citation:{key}:{sorted(set(sources) - set(valid))}")
        if value is not None and any("blocked" in by_id[s] and by_id[s]["blocked"] for s in valid):
            events.append(f"leak_blocked:{key}")
            value = None
        if value is not None and not valid:
            if mode == "evidence_only":
                events.append(f"uncited_value:{key}")
                value = None
            else:
                events.append(f"from_memory:{key}")
        predictions[key] = value
        if value is not None and valid:
            cites[key] = list(dict.fromkeys(by_id[s]["url"] for s in valid))
    for h in task.H:
        if not any(lookup.get(_norm(k)) == h["prop_key"] for k in body):
            events.append(f"missing_key:{h['prop_key']}")
    if matched == 0:
        return predictions, cites, events + ["no_target_keys"], False
    return predictions, cites, events, True


SYNTH_REPAIR = (
    "Your answer could not be parsed. Reply with only a valid JSON object in this format, "
    "using null when the evidence does not give a value:\n{output_format}"
)

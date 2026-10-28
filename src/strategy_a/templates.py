"""Prompt templates: parsing, rendering and PFE (LLM_SPEC.md §5).

Template file::

    ### SYSTEM
    <system instructions>
    ### USER
    <user message with {placeholders}>
    ### EXAMPLE
    <optional: how one few-shot example is written>

A file without markers is a user message only. ``{{`` and ``}}`` are literal braces.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmcore.config import ConfigError

from .tasks import Example, Task

PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")
SECTIONS = ("SYSTEM", "USER", "EXAMPLE")
MAIN_ALLOWED = {
    "node_type",
    "known_properties",
    "target_properties",
    "output_format",
    "examples",
    # Published DOLAP 2026 prompts: H names comma-separated ([PROPERTIES]) and the exact published user input.
    "properties_csv",
    "published_input",
}
# Each group needs one of its placeholders: the known properties, the properties to find, the output format.
REQUIRED_GROUPS = (
    {"known_properties", "published_input"},
    {"target_properties", "properties_csv"},
    {"output_format", "properties_csv"},  # the published prompts spell out the CSV format themselves
)
EXAMPLE_ALLOWED = {"node_type", "known_properties", "target_properties", "answer", "index"}

DEFAULT_EXAMPLE = """Example {index}
Known properties of the {node_type}:
{known_properties}
Properties to find:
{target_properties}
Answer:
{answer}"""

_HOP = re.compile(r"(-\[[^\]]+\]->|<-\[[^\]]+\]-)")


@dataclass
class Template:
    path: Path
    system: str
    user: str
    example: str
    text: str  # the whole file, for hashing

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def uses_examples(self) -> bool:
        return "examples" in placeholders(self.system + self.user)


def placeholders(text: str) -> set[str]:
    return {m.group(1) for m in PLACEHOLDER.finditer(text) if m.group(1)}


def fill(text: str, values: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        if m.group(0) == "{{":
            return "{"
        if m.group(0) == "}}":
            return "}"
        return values[m.group(1)]

    return PLACEHOLDER.sub(sub, text)


def strip_placeholders(text: str) -> str:
    """Template text as written by a human, without placeholders (for PFE)."""
    return fill(text, {name: "" for name in placeholders(text)})


def load_template(path: str | Path, allowed: set[str] | None = None, required_groups=None) -> Template:
    """``allowed``/``required_groups`` default to Strategy A's; Strategy B passes its own for plan/synthesis prompts."""
    allowed = MAIN_ALLOWED if allowed is None else allowed
    required_groups = REQUIRED_GROUPS if required_groups is None else required_groups
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    parts: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        marker = line.strip()
        if marker.startswith("### ") and marker[4:].strip() in SECTIONS:
            current = marker[4:].strip()
            if current in parts:
                raise ConfigError(f"{path}: section {current} appears twice")
            parts[current] = []
        elif current is None:
            parts.setdefault("USER", []).append(line)
        else:
            parts[current].append(line)
    section = {k: "\n".join(v).strip("\n") for k, v in parts.items()}
    tpl = Template(path, section.get("SYSTEM", ""), section.get("USER", ""), section.get("EXAMPLE", ""), text)
    main = placeholders(tpl.system) | placeholders(tpl.user)
    unknown = main - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown placeholders {sorted(unknown)}")
    missing = [sorted(g) for g in required_groups if not g & main]
    if missing:
        raise ConfigError(f"{path}: missing required placeholders, one of each: {missing}")
    unknown = placeholders(tpl.example) - EXAMPLE_ALLOWED
    if unknown:
        raise ConfigError(f"{path}: unknown placeholders in EXAMPLE {sorted(unknown)}")
    if not tpl.user.strip():
        raise ConfigError(f"{path}: empty USER section")
    return tpl


# Labels -------------------------------------------------------------------


def path_label(path: str, property_name: str) -> str:
    """Human-readable label for a property reached by ``path`` (§5.3).

    ``Airport`` -> ``Airport name``;
    ``Airport-[IN]->City`` -> ``City (the Airport is IN this City) name``;
    ``Airport<-[HAS]-Country`` -> ``Country (this Country HAS the Airport) name``;
    ``Airport-[HAS_ROUTE]->`` -> ``HAS_ROUTE relationship of the Airport: distance``;
    longer paths -> ``Country (path: Airport-[IN]->City-[IN]->Country) name``.
    """
    tokens = [t for t in _HOP.split(path) if t != ""]
    if len(tokens) == 1:
        return f"{tokens[0]} {property_name}"
    if len(tokens) == 2:  # edge path: "<type><hop>"
        start, hop = tokens
        rel = hop.strip("-<>[]")
        direction = "incoming " if hop.startswith("<") else ""
        return f"{rel} {direction}relationship of the {start}: {property_name}"
    if len(tokens) == 3:
        start, hop, end = tokens
        rel = hop.strip("-<>[]")
        if hop.startswith("<"):
            return f"{end} (this {end} {rel} the {start}) {property_name}"
        return f"{end} (the {start} is {rel} this {end}) {property_name}"
    return f"{tokens[-1]} (path: {path}) {property_name}"


def target_labels(H: list[dict]) -> dict[str, str]:
    """``{label: prop_key}`` for H. Labels are property names, made unique with the path if needed."""
    names = [h["property_name"] for h in H]
    labels: dict[str, str] = {}
    for h in H:
        name = h["property_name"]
        label = name if names.count(name) == 1 else path_label(h["path"], name)
        if label in labels:  # same name and path, distinct prop_keys: fall back to the key itself
            label = h["prop_key"]
        labels[label] = h["prop_key"]
    return labels


# Rendering ----------------------------------------------------------------


def _value(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def render_known(G: list[dict], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(
            [{"property": path_label(g["path"], g["property_name"]), "value": g["value"]} for g in G],
            ensure_ascii=False,
            indent=1,
        )
    return "\n".join(f"- {path_label(g['path'], g['property_name'])}: {_value(g['value'])}" for g in G)


def render_targets(labels: dict[str, str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def render_output_format(labels: dict[str, str]) -> str:
    return json.dumps({label: "<value or null>" for label in labels}, ensure_ascii=False)


def render_examples(template: Template, examples: list[Example], fmt: str) -> str:
    block = template.example or DEFAULT_EXAMPLE
    rendered = []
    for i, ex in enumerate(examples, 1):
        labels = target_labels(ex.H)
        answer = json.dumps({label: ex.answers[key] for label, key in labels.items()}, ensure_ascii=False)
        rendered.append(
            fill(
                block,
                {
                    "index": str(i),
                    "node_type": ex.node_type,
                    "known_properties": render_known(ex.G, fmt),
                    "target_properties": render_targets(labels),
                    "answer": answer,
                },
            )
        )
    return "\n\n".join(rendered)


@dataclass
class Rendered:
    system: str | None
    user: str
    labels: dict[str, str]  # label -> prop_key, for parsing


def published_input(task: Task, with_context: bool) -> str:
    """The user message of the published run, stored in tasks.csv by ``tasks import-published``/``build-published``."""
    column = "published_input_context" if with_context else "published_input"
    text = task.meta.get(column)
    if not text:
        raise ConfigError(f"task {task.task_id}: no {column} column; build the tasks with `tasks import-published`")
    return text


def render(template: Template, task: Task, examples_text: str, fmt: str, with_context: bool = False) -> Rendered:
    labels = target_labels(task.H)
    needed = placeholders(template.system) | placeholders(template.user)
    values = {
        "node_type": task.node_type,
        "known_properties": render_known(task.G, fmt),
        "target_properties": render_targets(labels),
        "output_format": render_output_format(labels),
        "examples": examples_text,
        "properties_csv": ",".join(h["property_name"] for h in task.H),
        "published_input": published_input(task, with_context) if "published_input" in needed else "",
    }
    system = fill(template.system, values).strip() or None
    return Rendered(system, fill(template.user, values).strip(), labels)


def pfe_text(template: Template, examples_text: str) -> str:
    """Text a human wrote for this variant: template without placeholders, plus the examples (§5.6)."""
    parts = [strip_placeholders(template.system), strip_placeholders(template.user)]
    if examples_text:
        parts.append(examples_text)
    return "\n".join(p for p in parts if p.strip())


# Effort (METRICS.md §6) -------------------------------------------------------------


def task_effort_texts(task: Task, with_context: bool, published: bool) -> dict[str, str]:
    """What a person would type for one task besides the template: the entity's known values,
    the graph context, and the names of the properties to find."""
    targets = ",".join(h["property_name"] for h in task.H)
    if published:
        text = published_input(task, with_context)
        found = re.search(r"input:\s*(.*?)(?:\n\s*\n|$)", text, re.S)
        context = re.search(r"graph:\s*(.*\S)", text, re.S) if with_context else None
        return {
            "input": found.group(1).strip() if found else text.strip(),
            "context": context.group(1).rstrip(".").strip() if context else "",
            "targets": targets,
        }
    own = [g for g in task.G if g["path"] == task.node_type]
    rest = [g for g in task.G if g["path"] != task.node_type]
    return {"input": render_known(own, "list"), "context": render_known(rest, "list"), "targets": targets}


def text_sizes(texts: dict[str, str], tokenizer) -> dict[str, dict[str, int]]:
    """Characters, words and tokens of each text: the inputs of PFE, KLM and token amplification."""
    return {
        name: {"chars": len(t), "words": len(t.split()), "tokens": tokenizer.count(t) if t else 0}
        for name, t in texts.items()
    }

"""Read tasks.csv (TASKDEFINITION_SPEC.md §7.1) and few-shot examples (LLM_SPEC.md §5.4).

This module never opens ground_truth.csv.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmcore.config import ConfigError


@dataclass
class Task:
    task_id: str
    task_set_version: str
    node_type: str
    node_id: str
    H: list[dict]  # {prop_key, property_name, element_type, path}
    G: list[dict]  # {prop_key, property_name, element_type, element_id, path, value, ...}
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    """A solved task used in few-shot prompts. ``answers`` maps H prop_key to its value."""

    node_id: str
    node_type: str
    H: list[dict]
    G: list[dict]
    answers: dict[str, Any]


def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    if path.name == "ground_truth.csv":
        raise ConfigError("Strategy A must never read ground_truth.csv")
    tasks = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            known = {"task_id", "task_set_version", "node_type", "node_id", "H", "G"}
            tasks.append(
                Task(
                    task_id=row["task_id"],
                    task_set_version=row["task_set_version"],
                    node_type=row["node_type"],
                    node_id=row["node_id"],
                    H=json.loads(row["H"]),
                    G=json.loads(row["G"]),
                    meta={k: v for k, v in row.items() if k not in known},
                )
            )
    if not tasks:
        raise ConfigError(f"{path}: no tasks")
    versions = {t.task_set_version for t in tasks}
    if len(versions) != 1:
        raise ConfigError(f"{path}: several task_set_version values {sorted(versions)}")
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path}: duplicate task_id")
    for t in tasks:
        if not t.H:
            raise ConfigError(f"task {t.task_id}: empty H")
    return tasks


def load_examples(path: str | Path) -> tuple[str, list[Example]]:
    """``{"version": "v1", "examples": [{node_id, node_type, G, H, answers}]}``."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    examples = []
    for i, e in enumerate(data.get("examples") or []):
        try:
            ex = Example(e["node_id"], e["node_type"], e["H"], e["G"], e["answers"])
        except KeyError as exc:
            raise ConfigError(f"{path}: example {i} is missing {exc}") from exc
        missing = {h["prop_key"] for h in ex.H} - set(ex.answers)
        if missing:
            raise ConfigError(f"{path}: example {i} has no answer for {sorted(missing)}")
        examples.append(ex)
    return str(data.get("version", "v1")), examples


def check_example_leakage(examples: list[Example], tasks: list[Task]) -> None:
    """Example nodes must not appear in the task set, as task node or G element (§5.4)."""
    used = {t.node_id for t in tasks} | {g.get("element_id") for t in tasks for g in t.G}
    leaked = sorted({e.node_id for e in examples} & used)
    if leaked:
        raise ConfigError(f"few-shot example nodes appear in tasks.csv: {leaked}")

"""Writers for tasks.csv, ground_truth.csv and manifest.json (TASKDEFINITION_SPEC.md §7).

CSV: UTF-8, comma separator, RFC 4180 quoting (CRLF line ends), header row.
H, G and ground-truth values are JSON, so numbers stay numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .difficulty import surprise
from .sampling import Task

TASK_COLUMNS = [
    "task_id",
    "task_set_version",
    "dataset",
    "node_type",
    "node_id",
    "node_centrality",
    "node_centrality_raw",
    "node_stratum",
    "H_stratum",
    "G_stratum",
    "size_H",
    "size_G",
    "H",
    "G",
    "difficulty",
    "diff_ratio",
    "diff_node_surprise",
    "diff_prop_surprise",
    # Derived per-task metrics (METRICS.md §3.1).
    "diff_prop_surprise_H",
    "diff_prop_surprise_G",
    "n_G_own",
    "n_G_context",
    "G_has_own_property",
    "G_max_hops",
    "G_candidates",
]

GROUND_TRUTH_COLUMNS = ["task_id", "prop_key", "value", "value_type"]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def hops(path: str) -> int:
    """Relationships between n and the property's element: 0 for n's own properties."""
    return path.count("]-")


def task_row(task: Task, version: str) -> dict:
    own = [c for c in task.G if hops(c.path) == 0 and c.element_id == task.node_id]
    return {
        "task_id": task.task_id,
        "task_set_version": version,
        "dataset": task.dataset,
        "node_type": task.node_type,
        "node_id": task.node_id,
        "node_centrality": repr(task.node_centrality),
        "node_centrality_raw": "" if task.node_centrality_raw is None else repr(task.node_centrality_raw),
        "node_stratum": task.node_stratum,
        "H_stratum": task.H_stratum,
        "G_stratum": task.G_stratum,
        "size_H": task.size_H,
        "size_G": task.size_G,
        "H": _json(
            [
                {"prop_key": c.prop_key, "property_name": c.property_name, "element_type": c.element_type, "path": c.path,
                 "popularity": c.popularity, "centrality": c.centrality, "score": c.score}
                for c in task.H
            ]
        ),
        "G": _json(
            [
                {
                    "prop_key": c.prop_key,
                    "property_name": c.property_name,
                    "element_type": c.element_type,
                    "element_id": c.element_id,
                    "path": c.path,
                    "value": c.value,
                    "popularity": c.popularity,
                    "centrality": c.centrality,
                    "score": c.score,
                    "hops": hops(c.path),
                }
                for c in task.G
            ]
        ),
        "difficulty": repr(task.difficulty.value),
        "diff_ratio": repr(task.difficulty.ratio),
        "diff_node_surprise": repr(task.difficulty.node_surprise),
        "diff_prop_surprise": repr(task.difficulty.prop_surprise),
        "diff_prop_surprise_H": repr(sum(surprise(c.score) for c in task.H)),
        "diff_prop_surprise_G": repr(sum(surprise(c.score) for c in task.G)),
        "n_G_own": len(own),
        "n_G_context": len(task.G) - len(own),
        "G_has_own_property": bool(own),
        "G_max_hops": max((hops(c.path) for c in task.G), default=0),
        "G_candidates": task.G_candidates,
    }


def ground_truth_rows(tasks: list[Task]) -> list[dict]:
    return [
        {"task_id": t.task_id, "prop_key": c.prop_key, "value": c.value, "value_type": c.value_type}
        for t in tasks
        for c in t.H
    ]


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tasks(path: Path, tasks: list[Task], version: str) -> None:
    write_csv(path, TASK_COLUMNS, [task_row(t, version) for t in tasks])


def write_ground_truth(path: Path, rows: list[dict]) -> None:
    write_csv(path, GROUND_TRUTH_COLUMNS, [{**r, "value": _json(r["value"])} for r in rows])


def write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

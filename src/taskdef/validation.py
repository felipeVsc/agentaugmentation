"""Checks run on every task set before it is written (TASKDEFINITION_SPEC.md §9).

The build fails if any check fails, so an invalid task set is never written.
"""

from __future__ import annotations

from collections import Counter

from .leakage import LeakagePairs, value_leaks
from .sampling import Quartiles, Task


class TaskSetValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        shown = "\n  ".join(errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        super().__init__(f"Task set validation failed ({len(errors)} errors):\n  {shown}{more}")


def validate_tasks(
    tasks: list[Task],
    ground_truth: list[dict],
    node_q: Quartiles,
    h_q: Quartiles,
    g_q: Quartiles,
    leakage_pairs: LeakagePairs,
    min_substring_len: int,
    g_value=lambda c: c.score,
) -> None:
    errors: list[str] = []

    counts = Counter(t.task_id for t in tasks)
    errors += [f"duplicate task_id {tid}" for tid, c in counts.items() if c > 1]

    gt_index: dict[tuple[str, str], list[dict]] = {}
    for row in ground_truth:
        gt_index.setdefault((row["task_id"], row["prop_key"]), []).append(row)

    for t in tasks:
        h_keys = {c.prop_key for c in t.H}
        g_keys = {c.prop_key for c in t.G}
        if h_keys & g_keys:
            errors.append(f"{t.task_id}: prop_key in both H and G: {sorted(h_keys & g_keys)}")
        split = leakage_pairs.blocked_by(h_keys) & g_keys
        if split:
            errors.append(f"{t.task_id}: leakage pair split across H and G: {sorted(split)}")
        if t.size_H != len(t.H) or t.size_G != len(t.G):
            errors.append(f"{t.task_id}: size_H/size_G do not match H/G")
        if node_q.stratum(t.node_centrality) != t.node_stratum:
            errors.append(f"{t.task_id}: node_stratum does not match the node quartiles")
        if any(h_q.stratum(c.popularity) != t.H_stratum for c in t.H):
            errors.append(f"{t.task_id}: an H property is outside H_stratum")
        if any(g_q.stratum(g_value(c)) != t.G_stratum for c in t.G):
            errors.append(f"{t.task_id}: a G property is outside G_stratum")
        for c in t.H:
            rows = gt_index.get((t.task_id, c.prop_key), [])
            if len(rows) != 1 or rows[0]["value"] is None:
                errors.append(f"{t.task_id}: {c.prop_key} needs exactly one non-null ground-truth row")
                continue
            for g in t.G:
                if value_leaks(rows[0]["value"], g.value, min_substring_len):
                    errors.append(f"{t.task_id}: ground-truth value of {c.prop_key} appears in G ({g.prop_key})")

    if errors:
        raise TaskSetValidationError(errors)

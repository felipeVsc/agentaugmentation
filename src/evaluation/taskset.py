"""Scoring of a task set built by ``tasks build`` (Algorithm 2), for Strategy A and B results.

Each hidden property of a task is scored against ``ground_truth.csv`` with the tolerance of the paper
(``dolap.matches_standard``): token overlap >= 0.75 in both directions for text (shared words over the
longer answer), relative similarity >= 0.75 for numbers, a list answer to a single value is wrong,
coordinates within 10 km (a lone latitude or longitude within 0.1°). An answer is

- ``correct`` / ``wrong`` when the model gave a value, and
- ``abstained`` when it gave null (or an abstention marker such as "unknown").

Two views, as in the draft (§5.3) and EVALUATION_SPEC §5.5:

- per run: every run scored on its own, for the spread across runs;
- voted: one answer per (model, variant, task, property) over the runs (text: most frequent value,
  numbers: mean), which is what the paper reports.

Metrics are micro-averaged over the scored properties:
precision = c / (c + w), recall = c / (c + w + a), F1, abstention rate, error rate. The paper's
"published" F1 counts only answered properties, so precision is 1 and accuracy = recall = c / (c + w);
``published_f1`` gives it for comparison with Table 3.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .dolap import coerce, matches_standard, vote

OUTCOMES = ["correct", "wrong", "abstained"]


def load_tasks(path: Path) -> pd.DataFrame:
    """Read a task set's tasks.csv and parse its H and G columns (JSON) into `H_list` and `G_list`."""
    t = pd.read_csv(path)
    t["H_list"] = t["H"].map(json.loads)
    t["G_list"] = t["G"].map(json.loads)
    return t


def load_ground_truth(path: Path) -> dict[tuple[str, str], object]:
    """Read ground_truth.csv as {(task_id, prop_key): value}."""
    gt = pd.read_csv(path)
    return {(r.task_id, r.prop_key): json.loads(r.value) for r in gt.itertuples(index=False)}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Read a strategy's results.csv and parse its predictions (JSON) into an `answer` dict per run."""
    r = pd.read_csv(results_dir / "results.csv")
    r["answer"] = r["predictions"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    return r


def load_runs_log(results_dir: Path) -> pd.DataFrame:
    """``runs.jsonl`` flattened: usage and effort fields of every run."""
    rows = []
    with open(results_dir / "runs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            row = {k: r.get(k) for k in ("task_id", "model", "variant", "run_index", "status")}
            row.update({f"usage_{k}": v for k, v in (r.get("usage") or {}).items() if not isinstance(v, (list, dict))})
            for part, sizes in (r.get("effort") or {}).items():
                if isinstance(sizes, dict):
                    row.update({f"effort_{part}_{k}": v for k, v in sizes.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def outcome(truth, value, name: str) -> str:
    """Classify one hidden property: 'abstained' (no value, null or an abstention marker), 'correct'
    (within the tolerance of `dolap.matches_standard`) or 'wrong'.
    """
    value, truth = coerce(value), coerce(truth)
    if value is None:
        return "abstained"
    return "correct" if matches_standard({name: truth}, {name: value}, name) else "wrong"


def score_runs(results: pd.DataFrame, tasks: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """One row per (run, hidden property)."""
    h = tasks.set_index("task_id")["H_list"].to_dict()
    rows = []
    for r in results.itertuples(index=False):
        for entry in h[r.task_id]:
            key, name = entry["prop_key"], entry["property_name"]
            if (r.task_id, key) not in truth:
                continue
            rows.append({"model": r.model, "variant": r.variant, "task_id": r.task_id, "run_index": r.run_index,
                         "status": r.status, "property": name, "prop_key": key,
                         "truth": truth[(r.task_id, key)], "answer": r.answer.get(key),
                         "outcome": outcome(truth[(r.task_id, key)], r.answer.get(key), name)})
    return pd.DataFrame(rows)


def score_voted(results: pd.DataFrame, tasks: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """One row per (model, variant, task, hidden property), on the answer voted over the runs."""
    h = tasks.set_index("task_id")["H_list"].to_dict()
    rows = []
    for (model, variant, task_id), g in results.groupby(["model", "variant", "task_id"]):
        answers = defaultdict(list)
        for ans in g["answer"]:
            for k, v in ans.items():
                answers[k].append(v)
        for entry in h[task_id]:
            key, name = entry["prop_key"], entry["property_name"]
            if (task_id, key) not in truth:
                continue
            voted = vote(answers.get(key, []))
            rows.append({"model": model, "variant": variant, "task_id": task_id, "runs": len(g),
                         "property": name, "prop_key": key, "truth": truth[(task_id, key)], "answer": voted,
                         "outcome": outcome(truth[(task_id, key)], voted, name)})
    return pd.DataFrame(rows)


def metrics(scored: pd.DataFrame) -> dict:
    """Metrics of a set of scored properties (c correct, w wrong, a abstained, n = c + w + a).

    precision = c / (c + w), abstentions excluded; recall = c / n, abstentions count as misses;
    F1 = 2PR / (P + R) = 2c / (2c + 2w + a); abstention_rate = a / n; error_rate = w / n.
    `published_accuracy` = c / (c + w) (equal to precision) and `published_f1` = 2c / (2c + w) (F1 ign,
    abstentions ignored) are kept for comparison with the DOLAP tables.
    """
    c = int((scored["outcome"] == "correct").sum())
    w = int((scored["outcome"] == "wrong").sum())
    a = int((scored["outcome"] == "abstained").sum())
    n = c + w + a
    p = c / (c + w) if c + w else float("nan")
    rec = c / n if n else float("nan")
    f1 = 2 * p * rec / (p + rec) if c else 0.0
    acc = c / (c + w) if c + w else float("nan")  # the paper's accuracy = recall over answered properties
    return {"properties": n, "correct": c, "wrong": w, "abstained": a, "precision": p, "recall": rec, "f1": f1,
            "abstention_rate": a / n if n else float("nan"), "error_rate": w / n if n else float("nan"),
            "published_accuracy": acc, "published_f1": 2 * acc / (1 + acc) if c else 0.0}


def metrics_by(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """`metrics` for each group of `by`, one row per group."""
    return pd.DataFrame([{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **metrics(g)}
                         for k, g in scored.groupby(by, observed=True)])


def bootstrap(scored: pd.DataFrame, stat: str = "f1", n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of a metric, resampling tasks (the unit that was sampled)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in scored.groupby("task_id")]
    vals = [metrics(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]))[stat] for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

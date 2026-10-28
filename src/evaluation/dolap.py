"""Scoring of the DOLAP 2026 replication (36 published tasks, 9 published prompts).

Two scorings, both on the representative answer of each (model, variant, task):
the vote over its runs (text: most frequent value, numbers: mean), as in the paper.

``published``
    The paper's evaluation notebook (``evaluation.ipynb`` in github.com/felipeVsc/llmaugmentation):
    every property the model answered that has a ground-truth value is compared, including
    properties that were not asked for; unanswered properties are not counted. F1 is
    sklearn's with ``y_true`` all ones, so precision is 1 whenever one answer is correct and
    recall = accuracy = correct / compared, F1 = 2·acc / (1 + acc). This is what Table 3 reports.

``standard``
    EVALUATION_SPEC §5.5–5.6: only the H properties that have a ground-truth value. Each is
    correct, wrong, or abstained (no value). precision = c / (c + w), recall = c / |H|,
    F1, abstention rate = a / |H|, error (hallucination) rate = w / |H|, exact match.

Matching uses the paper's tolerance in both: token overlap coefficient >= 0.75 for text,
relative similarity >= 0.75 for numbers, latitude/longitude together within 10 km. The paper
can only compare a coordinate when both latitude and longitude were answered, and skips the
others; the standard scoring also compares a lone coordinate (within 0.1°, about the 10 km
radius) and ``location`` as a coordinate pair, so that correct coordinates are not counted wrong.
"""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROMPT_INDEX = ["compact", "few5", "few3", "few1", "zero", "5cot", "zerocot", "3cot", "1cot"]  # p0..p8 of the paper
PROMPT_ORDER = ["compact", "zero", "few1", "few3", "few5", "zerocot", "1cot", "3cot", "5cot"]
QUADRANTS = ["easyeasy", "easyhard", "hardeasy", "hardhard"]
GEO = ("latitude", "longitude")


# Tolerance (published) --------------------------------------------------------


def normalize_tokens(s) -> list[str]:
    if s is None:
        return []
    s = str(s).lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.split()


def overlap_coefficient(s1, s2) -> float:
    t1, t2 = set(normalize_tokens(s1)), set(normalize_tokens(s2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))


def haversine(lat1, lon1, lat2, lon2) -> float:
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def relative_similarity(x, y) -> float:
    try:
        x, y = float(x), float(y)
        return 1 - abs(x - y) / max(abs(x), abs(y), 1e-9)
    except (TypeError, ValueError):
        return 0.0


def matches(correct: dict, predicted: dict, prop: str) -> bool | None:
    """``considerTolerance`` of the paper. None = not comparable."""
    if prop not in correct or prop not in predicted:
        return None
    ref, val = correct[prop], predicted[prop]
    if ref is None or val is None:
        return None
    if prop in GEO:
        if not all(k in correct and k in predicted and predicted[k] is not None for k in GEO):
            return None
        try:
            dist = haversine(float(predicted["latitude"]), float(predicted["longitude"]),
                             float(correct["latitude"]), float(correct["longitude"]))
        except (TypeError, ValueError):
            return False
        return dist < 10
    if isinstance(val, str):
        return overlap_coefficient(ref, val) >= 0.75
    return relative_similarity(val, ref) >= 0.75


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _pair(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value][:2]
    return [float(x) for x in _NUM.findall(str(value))][:2]


FEET = {"altitude", "runwaysLength", "runwaysWidth"}  # stored in feet in the airports graph
_VALUE_UNIT = re.compile(r"([+-]?\d+(?:[.,]\d+)?)(?::(\d{2}))?\s*(ft|feet|foot|m|meters?|metres?|km)?\b", re.I)


def as_number(prop: str, value) -> float | None:
    """The first number of a text answer ("2459 ft", "49 m", "2 runways", "UTC+5:30"), metres -> feet for FEET."""
    m = _VALUE_UNIT.search(str(value))
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    if m.group(2):  # hh:mm offsets
        x += math.copysign(int(m.group(2)) / 60, x)
    unit = (m.group(3) or "").lower()
    if prop in FEET and unit:
        x *= {"m": 3.28084, "meter": 3.28084, "meters": 3.28084, "metre": 3.28084, "metres": 3.28084,
              "km": 3280.84}.get(unit, 1.0)
    return x


def _coordinate(value) -> float | None:
    """A lone latitude or longitude: a number, or text like "23.96 N" / "161.15 W" (S and W are negative)."""
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUM.search(str(value))
    if not m:
        return None
    x = float(m.group())
    return -abs(x) if re.search(r"\b[SW]\b|south|west", str(value), re.I) else x


def matches_standard(correct: dict, predicted: dict, prop: str) -> bool | None:
    """``matches`` plus lone coordinates, ``location`` (POINT(lon,lat) in the ground truth) and numbers given
    as text with a unit (Strategy B answers "2459 ft"; the paper's rule compares such text as a string)."""
    if (isinstance(correct.get(prop), float) and isinstance(predicted.get(prop), str)
            and prop not in GEO and prop != "location"):
        x = as_number(prop, predicted[prop])
        if x is not None:
            predicted = {**predicted, prop: x}
    if prop in GEO and predicted.get(prop) is not None and prop in correct:
        both = all(predicted.get(k) is not None and k in correct for k in GEO)
        if not both:
            x = _coordinate(predicted[prop])
            try:
                return x is not None and abs(x - float(correct[prop])) <= 0.1
            except (TypeError, ValueError):
                return False
    if prop == "location" and predicted.get(prop) is not None and prop in correct:
        truth, pred = _pair(correct[prop]), _pair(predicted[prop])
        if len(truth) < 2 or len(pred) < 2:
            return False
        lon, lat = truth  # POINT(lon,lat)
        return min(haversine(pred[0], pred[1], lat, lon), haversine(pred[1], pred[0], lat, lon)) < 10
    return matches(correct, predicted, prop)


# Values and voting ------------------------------------------------------------

ABSTAIN = {"", "null", "none", "unknown", "n/a", "na", "not found", "not available", "-"}


def coerce(value):
    """Like pandas reading the voted CSV: numbers become floats, the rest stays text; abstention markers -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = str(value).strip()
    if text.casefold() in ABSTAIN:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def vote(values: list) -> Any:
    """Most frequent text value, or the mean when every answer is a number (paper §4.2)."""
    vals = [v for v in (coerce(x) for x in values) if v is not None]
    if not vals:
        return None
    if all(isinstance(v, float) for v in vals):
        return sum(vals) / len(vals)
    return Counter(str(v) for v in vals).most_common(1)[0][0]


# Loading ----------------------------------------------------------------------


def load_tasks(tasks_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(tasks_csv)
    df["H_names"] = df["H"].map(lambda s: [h["property_name"] for h in json.loads(s)])
    return df


def load_node_values(path: Path, key: str = "icao") -> dict[str, dict]:
    """The published ground truth, read like the paper's notebook (pandas, ``dropna``)."""
    gt = pd.read_csv(path).set_index(key)
    gt = gt[~gt.index.duplicated()]
    return {k: {p: (float(v) if isinstance(v, (int, float)) else v) for p, v in row.dropna().to_dict().items()}
            for k, row in gt.iterrows()}


def load_runs(results_dir: Path) -> pd.DataFrame:
    """One row per run, with every key the model returned (``answer``), not only the H keys."""
    from strategy_a.parsing import extract_csv

    results = pd.read_csv(results_dir / "results.csv")
    answers = {}
    with open(results_dir / "runs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            key = (rec["task_id"], rec["model"], rec["variant"], rec["run_index"])
            text = rec["calls"][-1]["response"]["text"] if rec["calls"] else ""
            obj, _ = extract_csv(text)
            answers[key] = obj if rec["status"] in ("ok", "truncated") else None
    results["answer"] = [
        answers.get((r.task_id, r.model, r.variant, r.run_index)) for r in results.itertuples(index=False)
    ]
    return results


def load_runs_b(results_dir: Path) -> pd.DataFrame:
    """Strategy B runs. ``predictions`` already holds one value (or null) per H property: no CSV to parse."""
    results = pd.read_csv(results_dir / "results.csv")
    results["answer"] = [
        {k.split(".", 1)[1]: v for k, v in json.loads(p).items() if v is not None} if s == "ok" else None
        for p, s in zip(results["predictions"], results["status"])
    ]
    return results


def voted_from_runs(runs: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    """Representative answer per (model, variant, task), plus usage summed over its runs."""
    info = tasks.set_index("task_id")
    rows = []
    for (model, variant, task_id), g in runs.groupby(["model", "variant", "task_id"]):
        per_prop = defaultdict(list)
        for ans in g["answer"]:
            for k, v in (ans or {}).items():
                per_prop[k].append(v)
        voted = {k: vote(v) for k, v in per_prop.items()}
        t = info.loc[task_id]
        rows.append({
            "source": "ollama",
            "model": model,
            "variant": variant,
            "prompt": variant.removesuffix("_ctx"),
            "context": variant.endswith("_ctx"),
            "task_id": task_id,
            "node": t["node_id"],
            "difficulty_prop": t["difficulty_prop"],
            "node_stratum": t["node_stratum"],
            "difficulty": t["difficulty"],
            "answer": {k: v for k, v in voted.items() if v is not None},
            "runs": len(g),
            "runs_format_ok": int((g["status"] == "ok").sum()),
            "tokens_in": g["tokens_in"].sum(),
            "tokens_out": g["tokens_out"].sum(),
            "latency_s": g["latency_s"].sum(),
            "energy_kwh": g["energy_kwh"].sum(),
            "co2e_kg": g["co2e_kg"].sum(),
            "pfe_tokens": g["pfe_tokens"].iloc[0],
        })
    return pd.DataFrame(rows)


PUBLISHED_FILES = {  # file -> (model, with context)
    "claude_sonnet_airports_majority_vote.csv": ("Claude Sonnet 4.5", False),
    "contextsonnet_airports_majority_vote.csv": ("Claude Sonnet 4.5", True),
    "haiku_airports_majority_vote.csv": ("Claude Haiku 4.5", False),
    "contexthaiku_airports_majority_vote.csv": ("Claude Haiku 4.5", True),
    "gpt5_airports_majority_vote.csv": ("GPT-5", False),
    "contextgptbig_airports_majority_vote.csv": ("GPT-5", True),
    "gptmini_airports_majority_vote.csv": ("GPT-5 Mini", False),
    "contextgptmini_airports_majority_vote.csv": ("GPT-5 Mini", True),
    "gemini_pro_airports_majority_vote.csv": ("Gemini 2.5 Pro", False),
    "contextgempro_airports_majority_vote.csv": ("Gemini 2.5 Pro", True),
    # gemini_flash_airports_majority_vote.csv uses "responseN" ids that cannot be linked to tasks.
    "geminiflash_airports_majority_vote.csv": ("Gemini 2.5 Flash", False),
    "contextgeminiflash_airports_majority_vote.csv": ("Gemini 2.5 Flash", True),
}


def load_published_votes(folder: Path, tasks: pd.DataFrame) -> pd.DataFrame:
    """The paper's voted answers. A missing (node, prompt, quadrant) row = the model gave no value.

    An airport with no row at all in a file was not run in that condition (the with-context Gemini files
    cover 4 of the 9 airports); its configurations are left out instead of being counted as abstentions.
    """
    info = tasks.set_index(["node_id", "difficulty_prop"])
    rows = []
    for name, (model, ctx) in PUBLISHED_FILES.items():
        path = folder / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        seen = {}
        for _, r in df.iterrows():
            parts = str(r["test"]).split("_")
            if len(parts) < 3:
                continue
            node, p, quad = parts[:3]
            prompt = PROMPT_INDEX[int(p[1:])]
            answer = {k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in r.drop(labels=["test"]).dropna().to_dict().items() if k != "class"}
            seen[(node, quad, prompt)] = answer
        covered = {n for n, _q, _p in seen}
        for (node, quad), t in info.iterrows():
            if node not in covered:
                continue
            for prompt in PROMPT_ORDER:
                rows.append({
                    "source": "published",
                    "model": model,
                    "variant": prompt + ("_ctx" if ctx else ""),
                    "prompt": prompt,
                    "context": ctx,
                    "task_id": t["task_id"],
                    "node": node,
                    "difficulty_prop": quad,
                    "node_stratum": t["node_stratum"],
                    "difficulty": t["difficulty"],
                    "answer": seen.get((node, quad, prompt), {}),
                })
    return pd.DataFrame(rows)


# Scoring ----------------------------------------------------------------------


def score_published(voted: pd.DataFrame, node_values: dict[str, dict]) -> pd.DataFrame:
    """One row per compared property, the paper's way (every answered property with a ground truth)."""
    out = []
    for r in voted.itertuples(index=False):
        correct = node_values.get(r.node, {})
        for prop in r.answer:
            if prop not in correct:
                continue
            ok = matches(correct, r.answer, prop)
            if ok is None:
                continue
            out.append({**_keys(r), "property": prop, "correct": bool(ok)})
    return pd.DataFrame(out)


def score_standard(voted: pd.DataFrame, tasks: pd.DataFrame, node_values: dict[str, dict]) -> pd.DataFrame:
    """One row per (configuration, H property with a ground truth): correct / wrong / abstained."""
    h_names = tasks.set_index("task_id")["H_names"].to_dict()
    out = []
    for r in voted.itertuples(index=False):
        correct = node_values.get(r.node, {})
        answer = {k: coerce(v) for k, v in r.answer.items()}  # "UNKNOWN", "N/A", ... are abstentions here
        for prop in h_names[r.task_id]:
            if prop not in correct:
                continue  # no ground truth (e.g. commercial_flights): not scored, as in the paper
            if answer.get(prop) is None:
                outcome = "abstained"
            else:
                ok = matches_standard(correct, answer, prop)
                outcome = "correct" if ok else "wrong"
            out.append({**_keys(r), "property": prop, "outcome": outcome})
    return pd.DataFrame(out)


def _keys(r) -> dict:
    return {k: getattr(r, k) for k in ("source", "model", "variant", "prompt", "context", "task_id", "node",
                                         "difficulty_prop", "node_stratum", "difficulty")}


def coverage(votes: pd.DataFrame) -> pd.DataFrame:
    """Airports and configurations available per model and condition."""
    return (votes.groupby(["source", "model", "context"])
                 .agg(airports=("node", "nunique"), tasks=("task_id", "nunique"), configurations=("task_id", "size"))
                 .reset_index())


def published_metrics(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g):
        acc = g["correct"].mean() if len(g) else float("nan")
        precision = 1.0 if g["correct"].any() else 0.0
        return pd.Series({"compared": len(g), "accuracy": acc, "precision": precision, "recall": acc,
                          "f1": 2 * precision * acc / (precision + acc) if precision + acc else 0.0})

    return scored.groupby(by).apply(agg, include_groups=False).reset_index()


def standard_metrics(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Micro-averaged over the H properties of the group."""
    def agg(g):
        c = (g["outcome"] == "correct").sum()
        w = (g["outcome"] == "wrong").sum()
        a = (g["outcome"] == "abstained").sum()
        n = c + w + a
        p = c / (c + w) if c + w else float("nan")
        rec = c / n if n else float("nan")
        f1 = 2 * p * rec / (p + rec) if c else 0.0
        return pd.Series({"properties": n, "correct": c, "wrong": w, "abstained": a, "precision": p,
                          "recall": rec, "f1": f1, "abstention_rate": a / n if n else float("nan"),
                          "error_rate": w / n if n else float("nan")})

    return scored.groupby(by).apply(agg, include_groups=False).reset_index()


def task_f1(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (model, variant, task): standard F1 and exact match, for difficulty correlations."""
    m = standard_metrics(scored, ["source", "model", "variant", "context", "task_id", "difficulty", "difficulty_prop"])
    m["exact_match"] = (m["correct"] == m["properties"]).astype(float)
    return m

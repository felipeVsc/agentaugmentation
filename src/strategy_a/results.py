"""results.csv (EVALUATION_SPEC.md §3.1) and runs.jsonl (LLM_SPEC.md §8.2) writers."""

from __future__ import annotations

import csv
import json
import os
import threading
from pathlib import Path

RESULT_COLUMNS = [
    "task_set_version",
    "task_id",
    "strategy",
    "variant",
    "model",
    "run_index",
    "status",
    "predictions",
    "evidence",
    "pfe_tokens",
    "tokens_in",
    "tokens_out",
    "tokens_reasoning",
    "llm_calls",
    "search_calls",
    "latency_s",
    "energy_kwh",
    "co2e_kg",
    "cost_usd",
    "prompt_version",
    "timestamp",
]

RunKey = tuple[str, str, str, int]  # (task_id, model, variant, run_index)


def run_key(row: dict) -> RunKey:
    return (row["task_id"], row["model"], row["variant"], int(row["run_index"]))


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_done(path: Path, task_set_version: str) -> set[RunKey]:
    rows = read_results(path)
    versions = {r["task_set_version"] for r in rows}
    if versions and versions != {task_set_version}:
        raise ValueError(
            f"{path} holds results for task set {sorted(versions)}, not {task_set_version}; use another output_dir"
        )
    return {run_key(r) for r in rows}


def drop_statuses(results_path: Path, runs_path: Path, statuses: set[str]) -> int:
    """Remove rows with these statuses (e.g. api_error) so they are run again."""
    rows = read_results(results_path)
    dropped = {run_key(r) for r in rows if r["status"] in statuses}
    if not dropped:
        return 0
    _rewrite_csv(results_path, [r for r in rows if run_key(r) not in dropped])
    if runs_path.exists():
        kept = []
        with open(runs_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if (rec["task_id"], rec["model"], rec["variant"], rec["run_index"]) not in dropped:
                    kept.append(line)
        tmp = runs_path.with_suffix(".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        os.replace(tmp, runs_path)
    return len(dropped)


def _rewrite_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


class ResultsWriter:
    """Appends one results.csv row and one runs.jsonl line per run, flushed immediately."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = out_dir / "results.csv"
        self.runs_path = out_dir / "runs.jsonl"
        new = not self.results_path.exists() or self.results_path.stat().st_size == 0
        if not new:
            with open(self.results_path, encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh), [])
            if header != RESULT_COLUMNS:  # written before columns were added: rewrite, new columns empty
                _rewrite_csv(self.results_path, read_results(self.results_path))
        self._csv_fh = open(self.results_path, "a", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._csv_fh, fieldnames=RESULT_COLUMNS)
        if new:
            self._csv.writeheader()
        self._log_fh = open(self.runs_path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, row: dict, log: dict) -> None:
        with self._lock:
            self._csv.writerow(row)
            self._log_fh.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
            self._csv_fh.flush()
            self._log_fh.flush()

    def close(self) -> None:
        self._csv_fh.close()
        self._log_fh.close()

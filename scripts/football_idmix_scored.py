"""Scored rows of the football idmix runs, as for airports v5 (notebooks/build_metrics_both_notebook.py §0).

    venv/bin/python scripts/football_idmix_scored.py
    venv/bin/python scripts/football_idmix_scored.py airports/v5_noctx     # any <dataset>/<set>, e.g. airports v5

For each of football v1_ctx_idmix and v1_noctx_idmix (or the sets given), writes results/<dataset>/<set>/scored/:

- scored_runs.csv: one row per (strategy, variant, model, task, run, hidden property), with the answer, the truth,
  the outcome (correct / wrong / abstained), the task's strata and identifiers, and the run's usage;
- scored_voted.csv: one row per (strategy, variant, model, task, hidden property), on the answer voted over the runs
  (what the reports score), with the answers and outcomes of the single runs.

Every strategy whose results exist is included (strategy_a, strategy_b, strategy_b_memory). When Strategy B exists, the
same rows are also written split by strategy: scored_{runs,voted}_strategy_a.csv (A only) and
scored_{runs,voted}_strategy_b.csv (B: search_evidence and search_memory)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation import taskset as E  # noqa: E402

MODELS = ["cloud-gpt-oss-20b", "cloud-gpt-oss-120b", "cloud-gemma4-31b", "cloud-mistral-large-3", "cloud-deepseek-v4.1-flash"]
STRATEGIES = {"A": "strategy_a", "B": "strategy_b", "B_memory": "strategy_b_memory"}
META = ["node_id", "node_stratum", "H_stratum", "G_stratum", "size_G", "n_G_own", "n_G_context", "condition",
        "has_name", "has_team", "difficulty"]
USAGE = ["tokens_in", "tokens_out", "tokens_reasoning", "llm_calls", "search_calls", "latency_s", "energy_kwh", "co2e_kg", "cost_usd"]
KEY = ["strategy", "variant", "model", "task_id", "run_index"]
dumps = lambda v: None if v is None else json.dumps(v, ensure_ascii=False)

SETS = sys.argv[1:] or ["football/v1_ctx_idmix", "football/v1_noctx_idmix"]
for name in SETS:
    ts, rd = ROOT / "tasks" / name, ROOT / "results" / name
    tasks = E.load_tasks(ts / "tasks.csv")
    meta = [c for c in META if c in tasks.columns]   # the identifier columns are football only
    truth = E.load_ground_truth(ts / "ground_truth.csv")
    found = {s: rd / d for s, d in STRATEGIES.items() if (rd / d / "results.csv").exists()}
    # sorted by run so that a tie between text answers goes to the first run, as in the reports
    res = pd.concat([E.load_results(d).assign(strategy=s) for s, d in found.items()], ignore_index=True)
    res = res[res.model.isin(MODELS)].sort_values(KEY, kind="stable").reset_index(drop=True)

    runs = pd.concat([E.score_runs(g, tasks, truth).assign(strategy=s) for s, g in res.groupby("strategy")], ignore_index=True)
    runs = runs.merge(res[KEY + USAGE], on=KEY, how="left").join(tasks.set_index("task_id")[meta], on="task_id")
    runs["answer"], runs["truth"] = runs.answer.map(dumps), runs.truth.map(dumps)
    runs = runs[KEY + ["status", "property", "prop_key", "truth", "answer", "outcome"] + meta + USAGE]

    voted = pd.concat([E.score_voted(g, tasks, truth).assign(strategy=s) for s, g in res.groupby("strategy")], ignore_index=True)
    per_run = (runs.sort_values("run_index").groupby(["strategy", "variant", "model", "task_id", "prop_key"])
               .agg(run_answers=("answer", list), run_outcomes=("outcome", list)).reset_index())
    voted = voted.merge(per_run, on=["strategy", "variant", "model", "task_id", "prop_key"], how="left").join(
        tasks.set_index("task_id")[meta], on="task_id")
    voted["answer"], voted["truth"] = voted.answer.map(dumps), voted.truth.map(dumps)
    voted["run_answers"] = voted.run_answers.map(lambda v: dumps([json.loads(x) if isinstance(x, str) else None for x in v]))
    voted["run_outcomes"] = voted.run_outcomes.map(json.dumps)
    voted = voted[["strategy", "variant", "model", "task_id", "runs", "property", "prop_key", "truth", "answer", "outcome",
                   "run_answers", "run_outcomes"] + meta]

    out = rd / "scored"
    out.mkdir(exist_ok=True)
    runs.to_csv(out / "scored_runs.csv", index=False)
    voted.to_csv(out / "scored_voted.csv", index=False)
    if (runs.strategy != "A").any():
        for suffix, keep in (("strategy_a", ["A"]), ("strategy_b", ["B", "B_memory"])):
            runs[runs.strategy.isin(keep)].to_csv(out / f"scored_runs_{suffix}.csv", index=False)
            voted[voted.strategy.isin(keep)].to_csv(out / f"scored_voted_{suffix}.csv", index=False)
    print(f"{name}: strategies {sorted(found)}; scored_runs.csv {len(runs):,} rows, scored_voted.csv {len(voted):,} rows;"
          f" voted outcomes {voted.outcome.value_counts().to_dict()}")

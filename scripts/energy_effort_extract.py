"""Per-run energy and effort, and voted accuracy, of the airports v5 and football idmix runs, for
notebooks/build_energy_effort_notebook.py (the runs.jsonl files are ~700 MB together; the notebook reads these instead).

    venv/bin/python scripts/energy_effort_extract.py

Writes results/energy_effort/:
- runs.csv: one row per ok run (dataset, set, strategy, model, variant, task, run): LLM calls, searches (all and real,
  i.e. not served by the search cache), tokens, EcoLogits EC (Wh), GWP (g CO2e) and WCF (mL) summed over the run's LLM
  calls, and the effort block (chars and tokens of template, input, context, targets);
- accuracy.csv: one row per (dataset, set, strategy, model, variant): correct, wrong, abstained of the voted answers
  (the scoring of the other reports), F1 and F1 with abstentions ignored."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation import taskset as E  # noqa: E402

MODELS = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-mistral-large-3", "cloud-deepseek-v4.1-flash"]
SOURCES = [  # dataset, set, strategy, results dir, tasks dir
    ("airports", "ctx", "A", "results/airports/v5_ctx/strategy_a", "tasks/airports/v5_ctx"),
    ("airports", "noctx", "A", "results/airports/v5_noctx/strategy_a", "tasks/airports/v5_noctx"),
    ("airports", "ctx", "B", "results/airports/v5_ctx/strategy_b", "tasks/airports/v5_ctx"),
    ("airports", "ctx", "B_memory", "results/airports/v5_ctx/strategy_b_memory", "tasks/airports/v5_ctx"),
    ("football", "ctx", "A", "results/football/v1_ctx_idmix/strategy_a", "tasks/football/v1_ctx_idmix"),
    ("football", "noctx", "A", "results/football/v1_noctx_idmix/strategy_a", "tasks/football/v1_noctx_idmix"),
    ("football", "ctx", "B", "results/football/v1_ctx_idmix/strategy_b", "tasks/football/v1_ctx_idmix"),
    ("football", "ctx", "B_memory", "results/football/v1_ctx_idmix/strategy_b_memory", "tasks/football/v1_ctx_idmix"),
]
PARTS = ("template", "input", "context", "targets")
OUT = ROOT / "results/energy_effort"
OUT.mkdir(parents=True, exist_ok=True)

runs, acc = [], []
for dataset, set_, strategy, rdir, tdir in SOURCES:
    n0 = len(runs)
    with open(ROOT / rdir / "runs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") != "ok" or r.get("model") not in MODELS:
                continue
            u, eff = r.get("usage") or {}, r.get("effort") or {}
            ec = gwp = wcf = 0.0
            for c in r.get("calls") or []:
                e = (c.get("response") or {}).get("energy") or {}
                ec += 1000 * (e.get("energy_kwh") or 0)
                gwp += 1000 * (e.get("co2e_kg") or 0)
                wcf += 1000 * (e.get("water_l") or 0)
            searches = u.get("search_calls", 0) or 0
            runs.append({"dataset": dataset, "set": set_, "strategy": strategy, "model": r["model"], "variant": r["variant"],
                         "task_id": r["task_id"], "run_index": int(r["run_index"]),
                         "llm_calls": len(r.get("calls") or []), "searches": searches,
                         "searches_real": searches - (u.get("search_cache_hits", 0) or 0),
                         "tokens_in": u.get("tokens_in"), "tokens_out": u.get("tokens_out"),
                         "tokens_reasoning": u.get("tokens_reasoning") or 0, "evidence_tokens": u.get("evidence_tokens") or 0,
                         "latency_s": u.get("latency_s"), "cost_usd": u.get("cost_usd"),
                         "EC_wh": ec, "GWP_g": gwp, "WCF_ml": wcf,
                         **{f"{p}_{k}": (eff.get(p) or {}).get(k, 0) for p in PARTS for k in ("chars", "tokens")}})
    print(f"{dataset} {set_} {strategy}: {len(runs) - n0:,} ok runs")

    tasks = E.load_tasks(ROOT / tdir / "tasks.csv")
    truth = E.load_ground_truth(ROOT / tdir / "ground_truth.csv")
    res = E.load_results(ROOT / rdir)
    res = res[res.model.isin(MODELS)].sort_values(["model", "variant", "task_id", "run_index"], kind="stable")
    for (m, v), g in E.score_voted(res, tasks, truth).groupby(["model", "variant"]):
        x = E.metrics(g)
        acc.append({"dataset": dataset, "set": set_, "strategy": strategy, "model": m, "variant": v, "tasks": g.task_id.nunique(),
                    "correct": x["correct"], "wrong": x["wrong"], "abstained": x["abstained"],
                    "f1": x["f1"], "f1_ign": x["published_f1"], "abstention": x["abstention_rate"]})

pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
pd.DataFrame(acc).to_csv(OUT / "accuracy.csv", index=False)
print(f"runs.csv {len(runs):,} rows; accuracy.csv {len(acc)} rows")

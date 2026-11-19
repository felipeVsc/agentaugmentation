"""Per-task comparison of Strategy A and Strategy B on airports-v5-ctx or football-v1-ctx-idmix (|H| = 1, so one
hidden property per task).

    venv/bin/python scripts/a_vs_b_per_task.py            # airports
    venv/bin/python scripts/a_vs_b_per_task.py football

Writes results/<dataset>/<set>/a_vs_b_per_task.csv: one row per model and task, with the voted answer and outcome of
A best (the model's best prompt by F1), A zero-shot and B. Runs used per model: 3 when B has all 3 runs, else run 1
(for A and B alike). For football it also gives B memory (search + memory) and the identifier columns.
Prints, per model, the task counts behind each precision and the paired outcome counts."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation import taskset as E  # noqa: E402

FOOTBALL = sys.argv[1:] == ["football"]
SET = "football/v1_ctx_idmix" if FOOTBALL else "airports/v5_ctx"
CTX = ROOT / "tasks" / SET
RES = ROOT / "results" / SET
OUT = RES / "a_vs_b_per_task.csv"
MODELS = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-mistral-large-3", "cloud-deepseek-v4.1-flash"]

tasks = E.load_tasks(CTX / "tasks.csv")
truth = E.load_ground_truth(CTX / "ground_truth.csv")
resB = E.load_results(RES / "strategy_b")
resB = resB[resB.model.isin(MODELS) & (resB.status == "ok")]
ok = resB.groupby(["model", "run_index"]).size()
RUNS = {m: 3 if all(ok.get((m, i), 0) == len(tasks) for i in (1, 2, 3)) else 1 for m in MODELS}
resB = resB[resB.run_index <= resB.model.map(RUNS)]
resA = E.load_results(RES / "strategy_a")
resA = resA[resA.model.isin(MODELS) & (resA.run_index <= resA.model.map(RUNS))]

vB = E.score_voted(resB, tasks, truth)
vA = E.score_voted(resA, tasks, truth)
if FOOTBALL:
    resM = E.load_results(RES / "strategy_b_memory")
    vM = E.score_voted(resM[resM.model.isin(MODELS) & (resM.run_index <= resM.model.map(RUNS))], tasks, truth)
fA = E.metrics_by(vA, ["model", "variant"])
BEST = fA.loc[fA.groupby("model").f1.idxmax()].set_index("model").variant.to_dict()

meta = tasks.set_index("task_id")[["node_id", "node_stratum", "H_stratum", "G_stratum", "size_G", "n_G_own", "n_G_context"]
                                  + (["condition", "has_name", "has_team"] if FOOTBALL else [])]
cols = ["task_id", "property", "truth", "answer", "outcome"]
rows = []
for m in MODELS:
    b = vB[vB.model == m][cols].rename(columns={"answer": "B answer", "outcome": "B"})
    ab = vA[(vA.model == m) & (vA.variant == BEST[m])][cols].rename(columns={"answer": "A best answer", "outcome": "A best"})
    az = vA[(vA.model == m) & (vA.variant == "zero")][cols].rename(columns={"answer": "A zero-shot answer", "outcome": "A zero-shot"})
    t = b.merge(ab.drop(columns=["truth"]), on=["task_id", "property"]).merge(az.drop(columns=["truth"]), on=["task_id", "property"])
    if FOOTBALL:
        bm = vM[vM.model == m][cols].rename(columns={"answer": "B memory answer", "outcome": "B memory"})
        t = t.merge(bm.drop(columns=["truth"]), on=["task_id", "property"])
    t.insert(0, "model", m.replace("cloud-", ""))
    t.insert(1, "runs", RUNS[m])
    t.insert(2, "A best prompt", BEST[m])
    rows.append(t.join(meta, on="task_id"))
per_task = pd.concat(rows, ignore_index=True)
per_task.to_csv(OUT, index=False)
print(f"wrote {OUT.relative_to(ROOT)} ({len(per_task)} rows)")

summary = []
for (m, runs, best), g in per_task.groupby(["model", "runs", "A best prompt"], sort=False):
    row = {"model": m, "runs": runs, "A best prompt": best}
    for c in ("B", "A best", "A zero-shot"):
        right, answered = (g[c] == "correct").sum(), (g[c] != "abstained").sum()
        row[f"{c}: correct / answered"] = f"{right} / {answered}"
        row[f"{c}: precision"] = round(right / answered, 2) if answered else None
    both = g[(g.B != "abstained") & (g["A best"] != "abstained")]
    row["both answered (B, A best)"] = len(both)
    row["B right, A wrong"] = int(((both.B == "correct") & (both["A best"] == "wrong")).sum())
    row["A right, B wrong"] = int(((both.B == "wrong") & (both["A best"] == "correct")).sum())
    row["both right"] = int(((both.B == "correct") & (both["A best"] == "correct")).sum())
    row["both wrong"] = int(((both.B == "wrong") & (both["A best"] == "wrong")).sum())
    summary.append(row)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(pd.DataFrame(summary).to_string(index=False))

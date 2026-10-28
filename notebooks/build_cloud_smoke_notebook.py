"""Builds notebooks/cloud_smoke_gemma4.ipynb: a one-run test of both strategies with gemma4:31b on Ollama Cloud.

    venv/bin/python notebooks/build_cloud_smoke_notebook.py notebooks/cloud_smoke_discussion.json
    venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/cloud_smoke_gemma4.ipynb

The discussion cells come from the JSON file (written after looking at the results)."""

import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "cloud_smoke_gemma4.ipynb"
disc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
DISC = json.loads(disc_path.read_text()) if disc_path and disc_path.exists() else {}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def discussion(key, placeholder="*Discussion written after the run.*"):
    md(DISC.get(key, placeholder))


md("""
# Test run of both strategies with gemma4:31b on Ollama Cloud

A cheap end-to-end check of the pipeline on the 36 published DOLAP 2026 tasks, **one run per configuration**
(the experiment uses 5 and votes):

| | Strategy A (LLM only) | Strategy B (single-turn search agent) |
|---|---|---|
| Model | `gemma4:31b` (31B dense, BF16) on Ollama Cloud, free tier | same |
| Configurations | 9 published prompts × with / without graph context | `search_evidence` (answer only from search results), `search_memory` (search results + own knowledge), no graph context |
| Runs | 36 tasks × 18 variants × 1 run = 648 | 36 tasks × 2 variants × 1 run = 72 |
| Search | — | Tavily **basic** depth (1 credit per query), dataset sources blocked, credit cap 150 |
| Config | `configs/strategy_a_dolap2026.yaml` (`--runs 1`) | `configs/strategy_b_dolap2026_smoke.yaml` |

With one run there is no vote, so every number here carries the full sampling noise of a single answer at
temperature 0.7. The goal is to check that the code is right and the numbers are plausible, not to draw conclusions.

**Scoring** (as in `dolap2026_ollama.ipynb`):
- *Standard*: each hidden property H with a ground truth is **correct**, **wrong** or **abstained**.
  Precision = correct / answered, recall = correct / |H|, F1 from both.
- *Published*: the paper's tolerance rules over every answered property with a ground truth; precision is 1 by
  construction, so F1 = 2·acc / (1 + acc). Shown for Strategy A only, to place gemma4 next to Table 3.
""")

code("""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
from evaluation import dolap as D

TASKS = ROOT / "tasks/airports/dolap2026/tasks.csv"
GROUND_TRUTH = ROOT / "tasks/airports/dolap2026/ground_truth.csv"
NODE_VALUES = ROOT / "data/dolap2026/ground_truth.csv"
PUBLISHED = ROOT / "data/dolap2026/published_results"
RESULTS_A = ROOT / "results/airports/dolap2026/strategy_a"
RESULTS_B = ROOT / "results/airports/dolap2026/strategy_b_smoke"
MODEL = "cloud-gemma4"

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)

# Chart style: light surface, recessive axes, text in ink tokens (dataviz reference palette).
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#c9c8c2"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "legend.frameon": False, "figure.dpi": 110,
})

def outcome_chart(df, label_col, title):
    \"\"\"Stacked shares of correct / wrong / abstained, one bar per row of df (already in display order).\"\"\"
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(df) + 1.4))
    n = df["properties"].to_numpy(float)
    left = np.zeros(len(df))
    for col, color, label in [("correct", GOOD, "Correct"), ("wrong", CRITICAL, "Wrong"), ("abstained", NEUTRAL, "Abstained")]:
        share = np.nan_to_num(df[col].to_numpy(float) / n)
        ax.barh(range(len(df)), share, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2, label=label)
        for i, (l, s) in enumerate(zip(left, share)):
            if s >= 0.07:
                ax.text(l + s / 2, i, f"{s:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if col != "abstained" else INK)
        left += share
    ax.set_yticks(range(len(df)), df[label_col]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(title, pad=28)
    fig.tight_layout(); plt.show()
""")

md("## 1. Tasks and ground truth")

code("""
tasks = D.load_tasks(TASKS)
node_values = D.load_node_values(NODE_VALUES)
gt = pd.read_csv(GROUND_TRUTH, keep_default_na=False)  # keep the literal "null" of a missing truth
print(f"{len(tasks)} tasks on {tasks.node_id.nunique()} airports; quadrants: {tasks.difficulty_prop.value_counts().to_dict()}")
print(f"{len(gt)} hidden properties; without ground truth: {(gt.value == 'null').sum()} "
      f"({', '.join(sorted(gt[gt.value == 'null'].prop_key.unique()))})")
""")

md("""
**Check against the graph.** The task ground truth comes from the published files. If the Neo4j credentials are in
`.env`, this cell compares every value with the airport node in the graph (credentials are read from the
environment and never printed).
""")

code("""
try:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / ".env")
    drv = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    with drv.session(database=os.environ.get("NEO4J_DATABASE") or None) as s:
        graph = {r["icao"]: r["p"] for r in s.run("MATCH (a:Airport) WHERE a.icao IN $ids RETURN a.icao AS icao, properties(a) AS p",
                                                    ids=sorted(tasks.node_id.unique()))}
    drv.close()
    node_of = tasks.set_index("task_id").node_id
    def graph_value(r):
        v = graph[node_of[r.task_id]].get(r.prop_key.split(".", 1)[1])
        return f"POINT({v.x},{v.y})" if hasattr(v, "x") else v
    def same(file_v, graph_v):
        file_v = json.loads(file_v)
        if isinstance(file_v, (int, float)) and isinstance(graph_v, (int, float)):
            return abs(file_v - graph_v) < 1e-6
        return file_v == graph_v or str(file_v) == str(graph_v)
    agree = [same(r.value, graph_value(r)) for r in gt.itertuples()]
    print(f"airports found in the graph: {len(graph)}/{tasks.node_id.nunique()}; "
          f"ground-truth values equal to the graph: {sum(agree)}/{len(gt)} (the nulls are absent in the graph too)")
except Exception as exc:
    print("graph check skipped:", type(exc).__name__, exc)
""")

discussion("data")

md("## 2. Strategy A: the published prompts, one run")

code("""
runs_a = D.load_runs(RESULTS_A)
runs_a = runs_a[runs_a.model.isin([MODEL, "qwen3-4b-instruct"])]
# One run per configuration for both models, so qwen3-4b (5 runs available) is cut to run 1 for a fair comparison.
runs_a = runs_a[runs_a.run_index == 1]
print(runs_a.groupby(["model", "status"]).size().unstack(fill_value=0))
votes_a = D.voted_from_runs(runs_a, tasks)      # one run: the "vote" is that run's answer
published_votes = D.load_published_votes(PUBLISHED, tasks)
ss_a = D.score_standard(votes_a, tasks, node_values)
sp_all = D.score_published(pd.concat([votes_a, published_votes], ignore_index=True), node_values)
""")

md("""
### 2.1 Format compliance

The published prompts ask for CSV `property,value`. A run without CSV counts as a format failure.
""")

code("""
order_cols = D.PROMPT_ORDER + [p + "_ctx" for p in D.PROMPT_ORDER]
comp = runs_a[runs_a.model == MODEL].assign(ok=lambda d: d.status.eq("ok")).groupby("variant").ok.mean()
print(f"gemma4: {comp.mean():.1%} of runs in CSV format; below 100%: {comp[comp < 1].round(2).to_dict() or 'none'}")
""")

md("""
### 2.2 Standard metrics per prompt and condition (gemma4)
""")

code("""
by_prompt = D.standard_metrics(ss_a[ss_a.model == MODEL], ["context", "prompt"])
tbl = by_prompt.pivot(index="prompt", columns="context", values=["f1", "precision", "recall", "abstention_rate"])
tbl = tbl.reindex(D.PROMPT_ORDER)
tbl.columns = [f"{m} {'ctx' if c else 'no ctx'}" for m, c in tbl.columns]
tbl.round(2)
""")

code("""
overall_a = D.standard_metrics(ss_a, ["model", "context"])
overall_a["label"] = overall_a.model + np.where(overall_a.context, " · with context", " · no context")
outcome_chart(overall_a.sort_values(["model", "context"]), "label",
              "Strategy A, all 9 prompts, one run: outcome of each hidden property")
overall_a[["model", "context", "properties", "correct", "wrong", "abstained", "precision", "recall", "f1"]].round(3)
""")

discussion("a_metrics")

md("""
### 2.3 Next to the published models (published scoring, best prompt per model and condition)

Same rule as Table 3 of the paper. The proprietary models' values are the published *voted* answers (5 runs);
gemma4 and qwen3-4b have one run, which usually lowers the score.
""")

code("""
pm = D.published_metrics(sp_all, ["source", "model", "context", "prompt"])
best = (pm.sort_values(["f1", "compared"], ascending=False).groupby(["model", "context"]).head(1)
          .sort_values(["context", "f1"], ascending=[True, False]))
best[["source", "model", "context", "prompt", "compared", "accuracy", "f1"]].round(3).reset_index(drop=True)
""")

discussion("a_published")

md("## 3. Strategy B: plan → search → synthesize, one run")

code("""
runs_b = D.load_runs_b(RESULTS_B)
log_b = [json.loads(l) for l in (RESULTS_B / "runs.jsonl").read_text().splitlines()]
print(runs_b.groupby(["variant", "status"]).size().unstack(fill_value=0))
# Search usage from runs.jsonl (every query of every run, whichever session ran it).
queries = [q for r in log_b for q in r["queries"]]
search_stats = pd.Series({"queries": len(queries), "answered from cache": sum(q["cached"] for q in queries),
                          "sent to Tavily": sum(not q["cached"] for q in queries),
                          "credits spent": sum(q["credits"] for q in queries),
                          "failed": sum(bool(q.get("error")) for q in queries)})
search_stats.to_frame("Strategy B search")
""")

md("""
### 3.1 What the agent searched

One row per (task, variant): how many queries the plan wrote, how many results came back, how many were dropped by
the blocklist (dataset sources), and whether the search came from the cache (the second variant reuses the first
one's plan and searches).
""")

code("""
qrows = []
for r in log_b:
    qrows.append({"task_id": r["task_id"], "variant": r["variant"], "status": r["status"],
                  "queries": len(r["queries"]), "from_cache": sum(q["cached"] for q in r["queries"]),
                  "credits": sum(q["credits"] for q in r["queries"]),
                  "results": len(r["results"]), "blocked": sum(1 for x in r["results"] if x.get("blocked")),
                  "evidence_passed": len(r["evidence_passed"])})
q = pd.DataFrame(qrows)
print(q.groupby("variant")[["queries", "from_cache", "credits", "results", "blocked", "evidence_passed"]].sum())
print("queries per task (first variant):", q[q.variant == "search_evidence"].queries.value_counts().sort_index().to_dict())
blocked_domains = pd.Series([x["url"].split("/")[2] for r in log_b for x in r["results"] if x.get("blocked")]).value_counts()
print("blocked hosts:", blocked_domains.to_dict())
""")

md("### 3.2 Standard metrics, Strategy B vs Strategy A (no graph context)")

md("""
Strategy B has no graph context, so the fair comparison is Strategy A **without context**: the `zero` prompt
(closest to B's prompts), the mean over the 9 prompts, and the best prompt chosen after the fact (optimistic).
""")

code("""
votes_b = D.voted_from_runs(runs_b, tasks).assign(context=False)
ss_b = D.score_standard(votes_b, tasks, node_values)
a_noctx = ss_a[(ss_a.model == MODEL) & ~ss_a.context]
best_prompt = D.standard_metrics(a_noctx, ["prompt"]).sort_values("f1", ascending=False).prompt.iloc[0]
rows = [("A · zero prompt", a_noctx[a_noctx.prompt == "zero"]),
        ("A · all 9 prompts", a_noctx),
        (f"A · best prompt ({best_prompt})", a_noctx[a_noctx.prompt == best_prompt])]
rows += [(f"B · {v}", ss_b[ss_b.variant == v]) for v in ["search_evidence", "search_memory"]]
cmp = pd.concat([D.standard_metrics(df.assign(_k=label), ["_k"]) for label, df in rows], ignore_index=True)
cmp = cmp.rename(columns={"_k": "configuration"})
outcome_chart(cmp, "configuration", "gemma4:31b, no graph context: Strategy A vs Strategy B (one run)")
cmp[["configuration", "properties", "correct", "wrong", "abstained", "precision", "recall", "f1"]].round(3)
""")

discussion("b_metrics")

md("### 3.3 By difficulty quadrant and by property")

code("""
quad = pd.concat([
    D.standard_metrics(a_noctx[a_noctx.prompt == "zero"].assign(cfg="A zero"), ["cfg", "difficulty_prop"]),
    D.standard_metrics(ss_b.assign(cfg="B " + ss_b.variant), ["cfg", "difficulty_prop"]),
])
print("Recall by quadrant (H-G difficulty):")
display(quad.pivot(index="difficulty_prop", columns="cfg", values="recall").reindex(D.QUADRANTS).round(2))
prop = pd.concat([
    D.standard_metrics(a_noctx[a_noctx.prompt == "zero"].assign(cfg="A zero"), ["cfg", "property"]),
    D.standard_metrics(ss_b.assign(cfg="B " + ss_b.variant), ["cfg", "property"]),
])
pt = prop.pivot(index="property", columns="cfg", values="recall")
pt.insert(0, "n", prop.groupby("property").properties.first())
print("Recall by hidden property:")
pt.sort_values("n", ascending=False).round(2)
""")

discussion("b_breakdown")

md("""
### 3.4 Examples

For a few tasks: the input the agent saw, its queries, the values it returned with their sources, and the truth.
""")

code("""
truth = gt.assign(prop=gt.prop_key.str.split(".").str[1]).set_index(["task_id", "prop"]).value
info = tasks.set_index("task_id")
def show(task_id, variant="search_evidence"):
    r = next(x for x in log_b if x["task_id"] == task_id and x["variant"] == variant)
    print(f"== {task_id} {info.loc[task_id, 'node_id']} ({info.loc[task_id, 'difficulty_prop']}) · {variant} · {r['status']}")
    print("input:", ", ".join(f"{g['property_name']}={g['value']}" for g in json.loads(info.loc[task_id, "G"])))
    for qq in r["queries"]:
        print(f"  query: {qq['query']!r} → {qq['n_results']} results{' (cache)' if qq['cached'] else ''}")
    for k, v in r["predictions"].items():
        p = k.split(".", 1)[1]
        src = r["evidence_urls"].get(k) or []
        print(f"  {p:<16} predicted={v!r:<40} truth={truth.get((task_id, p))}  {src[:1]}")
    print()

for t in ss_b[ss_b.variant == "search_evidence"].groupby("task_id").outcome.apply(lambda s: (s == "correct").mean()) \\
            .sort_values(ascending=False).index[[0, 1, -1]]:
    show(t)
""")

discussion("examples")

md("## 4. Cost and time")

code("""
cost = pd.DataFrame([
    {"strategy": "A", "runs": int((runs_a.model == MODEL).sum()),
     "tokens_in": runs_a.loc[runs_a.model == MODEL, "tokens_in"].sum(), "tokens_out": runs_a.loc[runs_a.model == MODEL, "tokens_out"].sum(),
     "LLM s per run": runs_a.loc[runs_a.model == MODEL, "latency_s"].mean(), "search credits": 0.0,
     "energy Wh (EcoLogits)": 1000 * runs_a.loc[runs_a.model == MODEL, "energy_kwh"].sum()},
    {"strategy": "B", "runs": len(runs_b), "tokens_in": runs_b.tokens_in.sum(), "tokens_out": runs_b.tokens_out.sum(),
     "LLM s per run": runs_b.latency_s.mean(), "search credits": search_stats["credits spent"],
     "energy Wh (EcoLogits)": 1000 * runs_b.energy_kwh.sum()},
])
cost["tokens per run"] = (cost.tokens_in + cost.tokens_out) / cost.runs
cost.round(2)
""")

discussion("cost")

md("## 5. Discussion")
discussion("main")
md("## 6. Changes made for this run, and limitations")
discussion("limits")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "agentaugmentation (venv)", "language": "python", "name": "agentaugmentation"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")

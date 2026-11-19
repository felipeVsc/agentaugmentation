"""Builds notebooks/dolap2026_lineup_report.ipynb: the current lineup on the 36 DOLAP 2026 tasks.

    venv/bin/python notebooks/build_dolap2026_lineup_report_notebook.py notebooks/dolap2026_lineup_report_discussion.json
    cd notebooks && ../venv/bin/jupyter nbconvert --to notebook --execute --inplace dolap2026_lineup_report.ipynb

Runs: scripts/run_dolap2026_lineup.sh (configs/strategy_a_dolap2026.yaml and configs/strategy_a_dolap2026_newprompts.yaml).
The discussion cells come from the JSON file (written after looking at the results)."""

import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "dolap2026_lineup_report.ipynb"
disc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
DISC = json.loads(disc_path.read_text()) if disc_path and disc_path.exists() else {}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def discussion(key):
    md(DISC.get(key, "*Discussion written after the runs.*"))


md("""
# The current models on the 36 DOLAP 2026 tasks

The DOLAP 2026 paper evaluated six proprietary models on 36 airport tasks. Our current experiments use open-weight
models, other tasks (airports v5) and new prompts, and score much lower. This report puts the current models on the
**same 36 tasks**, so that the remaining differences can be separated.

| Run | Models | Prompts | Runs per setup | Repair call | Source |
|---|---|---|---|---|---|
| **DOLAP (published)** | Claude Sonnet 4.5, Claude Haiku 4.5, GPT-5, GPT-5 Mini, Gemini 2.5 Pro, Gemini 2.5 Flash | the 9 published prompts | 5, voted | no | the paper's voted answers (`data/dolap2026/published_results`) |
| **Lineup, DOLAP protocol** | gpt-oss:20b, gemma4:31b, gpt-oss:120b, deepseek-v4.1-flash | the 9 published prompts | 5, voted | no | `configs/strategy_a_dolap2026.yaml` |
| **Lineup, new protocol** | the same 4 | the 9 prompts of the new version (Figure 2), with property notes and the v5 examples | 3, voted | yes | `configs/strategy_a_dolap2026_newprompts.yaml` |

Each prompt runs **without and with context**, on the exact published inputs: the airport's own properties on the
`input:` line and, with context, the published context block (neighbouring airports and the city). So each model has
18 setups. mistral-large-3, the fifth model of the lineup, was retired by Ollama Cloud and could not be run.

**Two scorings**, on the voted answer of each (model, setup, task):

- **DOLAP scoring**, as in the paper: every answered property with a ground truth is compared (asked or not),
  unanswered properties are ignored, lenient tolerance; precision is 1 by construction and
  DOLAP F1 = 2·acc / (1 + acc), acc = correct / answered. Table 3 of the paper reports the best of the 18 setups.
- **Our scoring**, as in the current experiments: only the asked properties, each correct (c), wrong (w) or
  abstained (a), with the current tolerance. **Main metrics: F1 = 2c / (2c + 2w + a) and precision = c / (c + w)**;
  beside them F1 ign = 2c / (2c + w), the DOLAP formula on the asked properties.

Unless stated otherwise, numbers pool all 18 setups (micro-average over the scored properties).
""")

code("""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
import sys; sys.path.insert(0, str(ROOT / "notebooks"))
from evaluation import dolap as D
import paper_report_lib as L   # chart style and colours of the other reports

TASKS = D.load_tasks(ROOT / "tasks/airports/dolap2026/tasks.csv")
NV = D.load_node_values(ROOT / "data/dolap2026/ground_truth.csv")
RES = {"lineup, DOLAP protocol": ROOT / "results/airports/dolap2026/strategy_a",
       "lineup, new protocol": ROOT / "results/airports/dolap2026/strategy_a_newprompts"}
LINEUP = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-deepseek-v4.1-flash"]
PUB = ["GPT-5", "GPT-5 Mini", "Gemini 2.5 Pro", "Gemini 2.5 Flash", "Claude Sonnet 4.5", "Claude Haiku 4.5"]
PROMPT = {"compact": "compact", "zero": "zero-shot", "few1": "1-shot", "few3": "3-shot", "few5": "5-shot",
          "zerocot": "zero-shot + CoT", "1cot": "1-shot + CoT", "3cot": "3-shot + CoT", "5cot": "5-shot + CoT",
          "few_1": "1-shot", "few_3": "3-shot", "few_5": "5-shot",
          "fewcot_1": "1-shot + CoT", "fewcot_3": "3-shot + CoT", "fewcot_5": "5-shot + CoT"}
PORDER = ["compact", "zero-shot", "zero-shot + CoT", "1-shot", "1-shot + CoT", "3-shot", "3-shot + CoT", "5-shot", "5-shot + CoT"]
PCOLOR = dict(zip(PUB, ["#2b2b29", "#55544f", "#6b6a66", "#8a8984", "#a3a29d", "#bdbcb6"]))
COLOR = {**{L.MNAME[m]: L.MCOLOR[m] for m in LINEUP}, **PCOLOR}
""")

md("## 1. Runs and voting")

code("""
frames, status = [], []
for proto, path in RES.items():
    if not (path / "results.csv").exists():
        continue
    runs = D.load_runs(path)
    runs = runs[runs.model.isin(LINEUP)]
    if runs.empty:
        continue
    status.append(runs.assign(protocol=proto).groupby(["protocol", "model"]).agg(
        runs=("run_index", "size"), ok=("status", lambda s: (s == "ok").mean()),
        tasks=("task_id", "nunique"), setups=("variant", "nunique")))
    # complete models only: every task × setup has its runs
    need = runs.groupby("model").variant.nunique().max()
    cover = runs.groupby(["model", "variant"]).task_id.nunique().groupby("model").min()
    full = [m for m in cover.index if cover[m] == len(TASKS) and runs[runs.model == m].variant.nunique() == need]
    if not full:
        continue
    v = D.voted_from_runs(runs[runs.model.isin(full)], TASKS)
    v["context"] = v.variant.str.contains("_ctx")
    v["prompt"] = v.variant.str.replace("_ctx", "", regex=False).map(PROMPT)
    v["source"] = proto
    v["model"] = v.model.map(L.MNAME)
    frames.append(v)
pub = D.load_published_votes(ROOT / "data/dolap2026/published_results", TASKS)
pub["prompt"] = pub.prompt.map(PROMPT); pub["source"] = "DOLAP (published)"
votes = pd.concat(frames + [pub], ignore_index=True)
display(pd.concat(status) if status else "no lineup runs yet")
print("voted configurations per source:", votes.groupby("source").size().to_dict())
""")

code("""
H = TASKS.set_index("task_id").H_names.to_dict()
G = TASKS.set_index("task_id").G_drawn.map(json.loads).to_dict()
sp = D.score_published(votes, NV)                     # DOLAP scoring: one row per compared property
sp["kind"] = ["asked" if p in H[t] else ("copied from input" if p in G[t] else "not asked")
              for t, p in zip(sp.task_id, sp.property)]
ss = D.score_standard(votes, TASKS, NV)               # our scoring: one row per asked property with a ground truth
MODELS = [m for m in [L.MNAME[x] for x in LINEUP] + PUB if m in set(votes.model)]
SOURCES = [s for s in ["DOLAP (published)", "lineup, DOLAP protocol", "lineup, new protocol"] if s in set(votes.source)]

def ours(g):
    c, w, a = [(g.outcome == o).sum() for o in ("correct", "wrong", "abstained")]
    n = c + w + a
    return pd.Series({"n": n, "F1": 2 * c / (2 * c + 2 * w + a) if n else np.nan, "precision": c / (c + w) if c + w else np.nan,
                      "F1 ign": 2 * c / (2 * c + w) if c + w else np.nan, "recall": c / n if n else np.nan,
                      "abstention": a / n if n else np.nan, "error": w / n if n else np.nan})

def dolap_f1(g):
    acc = g.correct.mean() if len(g) else np.nan
    return pd.Series({"compared": len(g), "DOLAP F1": 2 * acc / (1 + acc) if acc > 0 else 0.0})

def score(by, s=None, p=None):
    s = ss if s is None else s; p = sp if p is None else p
    a = s.groupby(by).apply(ours, include_groups=False)
    b = p.groupby(by).apply(dolap_f1, include_groups=False)
    return a.join(b, how="outer")
""")

md("""
## 2. The headline: all models on the same 36 tasks

All 18 setups pooled. For the published models the Gemini rows with context cover 4 of the 9 airports (the published
files have no other airports), so they rest on fewer tasks.
""")

code("""
head = score(["source", "model"]).reset_index()
head["order"] = head.model.map({m: i for i, m in enumerate(MODELS)})
head = head.sort_values(["source", "F1"], ascending=[True, False])
head[["source", "model", "F1", "precision", "F1 ign", "recall", "abstention", "error", "n", "DOLAP F1", "compared"]].round(3)
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 0.36 * len(head) + 1.8), sharey=True)
lab = [f"{m}  ({'published' if s.startswith('DOLAP') else s.split(', ')[1]})" for s, m in zip(head.source, head.model)]
y = np.arange(len(head))
for ax, col in zip(axes, ["F1", "precision", "F1 ign"]):
    ax.barh(y, head[col], color=[COLOR[m] for m in head.model], height=0.7)
    for yi, v in zip(y, head[col]):
        ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=8, color=L.INK2)
    ax.set_title(col); ax.set_xlim(0, 0.85); ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, lab); axes[0].invert_yaxis()
fig.suptitle("Our scoring on the 36 DOLAP tasks, all 18 setups pooled", x=0.01, ha="left")
fig.tight_layout(); plt.show()
""")

discussion("headline")

md("""
## 3. Table 3 of the paper, extended

The paper's way: for each model, the setup with the best DOLAP F1, with the number of compared properties it rests on.
Next to it, our metrics of that same setup, and our F1 of the model's best setup chosen by our F1.
""")

code("""
cfg = score(["source", "model", "context", "prompt"]).reset_index()
rows = []
for (s, m), g in cfg.groupby(["source", "model"]):
    b = g.sort_values(["DOLAP F1", "compared"], ascending=False).iloc[0]
    o = g.sort_values("F1", ascending=False).iloc[0]
    rows.append({"source": s, "model": m, "best setup (DOLAP F1)": f"{b.prompt}{' + ctx' if b.context else ''}",
                 "DOLAP F1": b["DOLAP F1"], "compared": int(b.compared), "its F1": b.F1, "its precision": b.precision,
                 "best F1 (any setup)": o.F1, "pooled F1": head.set_index(["source", "model"]).loc[(s, m), "F1"]})
t3 = pd.DataFrame(rows).sort_values(["source", "DOLAP F1"], ascending=[True, False])
t3.round(3)
""")

md("""
## 4. The protocol: published prompts or the new prompts, same models and tasks

The lineup's two runs differ only in the prompt protocol (and 5 against 3 runs). Positive = the new protocol scores higher.
""")

code("""
lin = score(["source", "model", "context"]).reset_index()
lin = lin[lin.source.str.startswith("lineup")]
if lin.source.nunique() == 2:
    pv = lin.pivot_table(index=["model", "context"], columns="source", values=["F1", "precision", "F1 ign", "abstention"])
    diff = pd.DataFrame({k: pv[(k, "lineup, new protocol")] - pv[(k, "lineup, DOLAP protocol")] for k in ["F1", "precision", "F1 ign", "abstention"]})
    display(pv.round(3))
    print("new protocol minus DOLAP protocol:")
    display(diff.round(3))
else:
    print("needs both lineup runs")
""")

discussion("protocol")

md("""
## 5. Context

Same tasks and prompts, with and without the published context block. For the Gemini models only the 4 airports
present in both conditions are used. Positive = context helps.
""")

code("""
def same_airports(df):
    keep = []
    for (s, m), g in df.groupby(["source", "model"]):
        both = set(g[g.context].node) & set(g[~g.context].node)
        keep.append(g[g.node.isin(both)])
    return pd.concat(keep)

ss_c, sp_c = same_airports(ss), same_airports(sp)
cx = score(["source", "model", "context"], ss_c, sp_c).reset_index()
pv = cx.pivot_table(index=["source", "model"], columns="context", values=["F1", "precision", "abstention", "DOLAP F1"])
ctx_delta = pd.DataFrame({k: pv[(k, True)] - pv[(k, False)] for k in ["F1", "precision", "abstention", "DOLAP F1"]})
ctx_delta.round(3)
""")

code("""
q = score(["source", "model", "context", "difficulty_prop"], ss_c, sp_c).reset_index()
hh = q[q.difficulty_prop == "hardhard"].pivot_table(index=["source", "model"], columns="context", values=["F1", "precision", "DOLAP F1"])
print("Hard H / hard G quadrant only (the paper's claim: context helps there):")
pd.DataFrame({k: hh[(k, True)] - hh[(k, False)] for k in ["DOLAP F1", "F1", "precision"]}).round(3)
""")

discussion("context")

md("""
## 6. Difficulty quadrants

Our F1 per H/G quadrant (easy/hard = popularity of the hidden / given properties), all setups.
""")

code("""
QN = {"easyeasy": "H easy, G easy", "easyhard": "H easy, G hard", "hardeasy": "H hard, G easy", "hardhard": "H hard, G hard"}
qq = score(["source", "model", "difficulty_prop"]).reset_index()
qt = qq.pivot_table(index=["source", "model"], columns="difficulty_prop", values="F1").rename(columns=QN)
qt.round(3)
""")

code("""
fig, axes = plt.subplots(1, len(SOURCES), figsize=(5 * len(SOURCES), 3.8), sharey=True)
axes = np.atleast_1d(axes)
for ax, s in zip(axes, SOURCES):
    sub = qq[qq.source == s]
    for m, g in sub.groupby("model"):
        g = g.set_index("difficulty_prop").reindex(list(QN))
        ax.plot(range(4), g.F1, marker="o", color=COLOR[m], label=m, lw=1.8)
    ax.set_xticks(range(4), [v.replace(", ", "\\n") for v in QN.values()], fontsize=8)
    ax.set_title(s); ax.legend(fontsize=7.5, loc="upper left")
axes[0].set_ylabel("F1 (our scoring)")
fig.tight_layout(); plt.show()
""")

discussion("difficulty")

md("## 7. Prompts")

code("""
pp = score(["source", "model", "prompt"]).reset_index()
for s in SOURCES:
    piv = pp[pp.source == s].pivot_table(index="model", columns="prompt", values="F1").reindex(columns=PORDER)
    piv = piv.reindex([m for m in MODELS if m in piv.index])
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(piv) + 1.3))
    L.heat(ax, piv, vmin=0, vmax=0.5, xlabels=[c.replace(" + ", "\\n+ ") for c in piv.columns])
    ax.set_title(f"F1 per prompt (both context conditions): {s}")
    fig.tight_layout(); plt.show()
""")

discussion("prompts")

md("""
## 8. Properties

Recall (share found) per asked property, all setups. `n` = how many of each property the 36 tasks ask (per setup).
""")

code("""
pr = ss.assign(found=ss.outcome.eq("correct")).pivot_table(index="property", columns="model", values="found", aggfunc="mean")
pr = pr.reindex(columns=[m for m in MODELS if m in pr.columns])
pr.insert(0, "n", ss[(ss.model == ss.model.iloc[0]) & (ss.source == ss.source.iloc[0])].groupby("property").size() / 18)
print("Lineup columns: both protocols together.")
pr.round(2)
""")

md("""
## 9. The same models on the paper's airport task set

With models and prompts held fixed (lineup, new protocol), what remains between the DOLAP tasks and the airport tasks of
the paper is the task set itself. Airport results: the 108 resampled tasks (`data/new-diff-airports_ressampled.csv`) of
`results/airports/v5_ctx/scored/scored_voted.csv` (Strategy A, all 9 prompts, 3 runs; the DOLAP runs here use 3 runs too).
Note that the context condition is designed differently: in DOLAP the context block is added to the airport's own
properties; in v5 context properties replace some of them (|G| fixed).
""")

code("""
rows = []
RESAMPLED = set(pd.read_csv(ROOT / "data/new-diff-airports_ressampled.csv").taskID)   # the 108 airport tasks used in the paper
s5 = pd.read_csv(ROOT / "results/airports/v5_ctx/scored/scored_voted.csv", low_memory=False)
s5 = s5[s5.strategy.eq("A") & s5.model.isin(LINEUP) & s5.task_id.isin(RESAMPLED)].assign(model=lambda d: d.model.map(L.MNAME))
for m, g in s5.groupby("model"):
    rows.append({"model": m, "task set": "airports, 108 resampled tasks", **ours(g).to_dict()})
new = ss[ss.source == "lineup, new protocol"]
for (m, c), g in new.groupby(["model", "context"]):
    rows.append({"model": m, "task set": "DOLAP 36, " + ("with context" if c else "no context"), **ours(g).to_dict()})
cmp5 = pd.DataFrame(rows)
cmp5.pivot_table(index="model", columns="task set", values="F1").round(3)
""")

code("""
cmp5.pivot_table(index="model", columns="task set", values="precision").round(3)
""")

discussion("taskset")

md("## 10. Tokens and cost")

code("""
use = []
for proto, path in RES.items():
    if (path / "results.csv").exists():
        r = pd.read_csv(path / "results.csv")
        r = r[r.model.isin(LINEUP)]
        use.append(r.groupby("model").agg(runs=("run_index", "size"), tokens_in=("tokens_in", "sum"),
                                          tokens_out=("tokens_out", "sum")).assign(protocol=proto))
if use:
    use = pd.concat(use).reset_index()
    import yaml
    price = {}
    for f in ["configs/strategy_a_dolap2026.yaml", "configs/strategy_a_dolap2026_newprompts.yaml"]:
        for m in yaml.safe_load(open(ROOT / f))["models"]:
            if "pricing" in m:
                price[m["id"]] = m["pricing"]
    use["cost USD (list price)"] = [r.tokens_in / 1e6 * price[r.model]["input_per_mtok"] + r.tokens_out / 1e6 * price[r.model]["output_per_mtok"]
                                    for r in use.itertuples()]
    use["model"] = use.model.map(L.MNAME)
    display(use.round(3))
""")

md("## 11. Conclusions")
discussion("main")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "agentaugmentation (venv)", "language": "python", "name": "agentaugmentation"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")

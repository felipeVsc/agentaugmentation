"""Builds notebooks/metrics_both.ipynb: the metrics of Strategy A, of Strategy B, then A against B, on airports-v5-ctx.

    venv/bin/python notebooks/build_metrics_both_notebook.py
    venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/metrics_both.ipynb

The notebook also writes the scored rows every table is aggregated from (results/airports/v5_ctx/scored/):
- scored_runs.csv: one row per (strategy, variant, model, task, run, hidden property), with the answer, the truth,
  the outcome and the usage of that run;
- scored_voted.csv: one row per (strategy, variant, model, task, hidden property), on the answer voted over the runs.
Same scoring as the v5 reports (evaluation.taskset)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "metrics_both.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Metrics of Strategy A and Strategy B (airports v5, context on)

Three parts: **Strategy A alone** (LLM only, 9 prompts), **Strategy B alone** (single-turn agent: web search with
evidence only, and the same evidence with the model's memory allowed), then **A against B** on the same tasks.

- 243 tasks, 1 hidden property each, 5 models, 3 runs per task and configuration.
- **Voted** answer per (model, configuration, task): text = most frequent value, numbers = mean; abstained only
  when all 3 runs abstain. Tables use the voted answer unless they say *per run*.
- 95% intervals: bootstrap over tasks (2,000 resamples); differences: paired bootstrap over the same tasks.

The rows every number is aggregated from are written to `results/airports/v5_ctx/scored/` (section 0).
""")

md("""
## Metrics

Each hidden property is **correct** (c), **wrong** (w) or **abstained** (a); n = c + w + a.

| Metric | Formula | Reads as |
|---|---|---|
| precision | c / (c + w) | share of the given answers that are right |
| recall | c / n | share of all properties answered right |
| **F1** | 2·P·R / (P + R) = 2c / (2c + 2w + a) | abstentions count against the model |
| accuracy (ign.) | c / (c + w) | = precision; the paper's accuracy |
| **F1 (ign.)** | 2·acc / (1 + acc) = 2c / (2c + w) | abstentions ignored (the paper's Table 3) |
| abstention | a / n | how often the model says it doesn't know |
| error rate | w / n | how often it gives a wrong value |
| run agreement | share of tasks whose 3 runs have the same outcome | stability |
| F1 sd over runs | sd of the F1 of run 1, 2, 3 scored alone | stability |
| cost, tokens, energy, latency | per run, from the run logs | effort |
| cost per correct answer | total cost of the 3 runs / c (voted) | efficiency |

*A* abstention lowers F1 but not F1 (ign.); a wrong answer lowers both.
""")

code("""
from pathlib import Path
from math import comb
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
from evaluation import taskset as E

TASKS = ROOT / "tasks/airports/v5_ctx"
RES = {"A": [ROOT / "results/airports/v5_ctx/strategy_a"],
       "B": [ROOT / "results/airports/v5_ctx/strategy_b", ROOT / "results/airports/v5_ctx/strategy_b_memory"]}
SCORED = ROOT / "results/airports/v5_ctx/scored"
FIG = ROOT / "notebooks/figures/metrics_both"
SCORED.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 60)

INK, INK2, SURFACE, GRID, MUTED = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0", "#a3a29c"
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd8", "#d9a514"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10,
    "legend.frameon": False, "figure.dpi": 110,
})
MODELS = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-mistral-large-3", "cloud-deepseek-v4.1-flash"]
MNAME = {"cloud-gpt-oss-20b": "gpt-oss:20b", "cloud-gemma4-31b": "gemma4:31b", "cloud-gpt-oss-120b": "gpt-oss:120b",
         "cloud-mistral-large-3": "mistral-large-3", "cloud-deepseek-v4.1-flash": "deepseek-v4.1-flash"}
MCOLOR = dict(zip(MODELS, [BLUE, AQUA, VIOLET, ORANGE, YELLOW]))
PROMPTS = ["compact", "zero", "few_1", "few_3", "few_5", "zerocot", "fewcot_1", "fewcot_3", "fewcot_5"]
VNAME = {"compact": "compact", "zero": "zero-shot", "few_1": "1-shot", "few_3": "3-shot", "few_5": "5-shot",
         "zerocot": "zero-shot + CoT", "fewcot_1": "1-shot + CoT", "fewcot_3": "3-shot + CoT", "fewcot_5": "5-shot + CoT",
         "search_evidence": "search (evidence only)", "search_memory": "search + memory"}
B_VARIANTS = ["search_evidence", "search_memory"]
PCT = "{:.0%}".format
# (column, title, short name) of the two F1 scores; every figure exists once per score
METRICS = [("F1", "F1", "f1"), ("F1 (ign.)", "F1, abstentions ignored", "f1_ign")]

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE)
""")

md("""
## 0. Scoring and the raw scored rows
""")

code("""
tasks = E.load_tasks(TASKS / "tasks.csv")
truth = E.load_ground_truth(TASKS / "ground_truth.csv")
META = ["node_id", "node_stratum", "H_stratum", "G_stratum", "size_G", "n_G_own", "n_G_context", "difficulty"]
USAGE = ["tokens_in", "tokens_out", "tokens_reasoning", "llm_calls", "search_calls", "latency_s", "energy_kwh", "co2e_kg", "cost_usd"]

res = pd.concat([E.load_results(d).assign(strategy=s) for s, dirs in RES.items() for d in dirs], ignore_index=True)
res = res[res.model.isin(MODELS)].reset_index(drop=True)
KEY = ["strategy", "variant", "model", "task_id", "run_index"]

runs = pd.concat([E.score_runs(g, tasks, truth).assign(strategy=s) for s, g in res.groupby("strategy")], ignore_index=True)
runs = runs.merge(res[KEY + USAGE], on=KEY, how="left").join(tasks.set_index("task_id")[META], on="task_id")
runs["answer"] = runs.answer.map(lambda v: None if v is None else json.dumps(v, ensure_ascii=False))
runs["truth"] = runs.truth.map(lambda v: json.dumps(v, ensure_ascii=False))
runs = runs[KEY + ["status", "property", "prop_key", "truth", "answer", "outcome"] + META + USAGE]

voted = pd.concat([E.score_voted(g, tasks, truth).assign(strategy=s) for s, g in res.groupby("strategy")], ignore_index=True)
run_answers = (runs.sort_values("run_index").groupby(["strategy", "variant", "model", "task_id", "prop_key"])
               .agg(run_answers=("answer", list), run_outcomes=("outcome", list)).reset_index())
voted = voted.merge(run_answers, on=["strategy", "variant", "model", "task_id", "prop_key"], how="left").join(
    tasks.set_index("task_id")[META], on="task_id")
voted["answer"] = voted.answer.map(lambda v: None if v is None else json.dumps(v, ensure_ascii=False))
voted["truth"] = voted.truth.map(lambda v: json.dumps(v, ensure_ascii=False))
voted["run_answers"] = voted.run_answers.map(lambda v: json.dumps([json.loads(x) if isinstance(x, str) else None for x in v], ensure_ascii=False))
voted["run_outcomes"] = voted.run_outcomes.map(json.dumps)
voted = voted[["strategy", "variant", "model", "task_id", "runs", "property", "prop_key", "truth", "answer", "outcome",
               "run_answers", "run_outcomes"] + META]

runs.to_csv(SCORED / "scored_runs.csv", index=False)
voted.to_csv(SCORED / "scored_voted.csv", index=False)
print(f"scored_runs.csv: {len(runs):,} rows; scored_voted.csv: {len(voted):,} rows")
display(runs.groupby(["strategy", "variant"]).agg(models=("model", "nunique"), tasks=("task_id", "nunique"),
                                                  runs=("run_index", "nunique"), rows=("outcome", "size")))
""")

md("""
The raw values: `results.csv` of each strategy holds the predictions of every run, unscored; `ground_truth.csv`
of the task set holds the true values. The two files above join them, one scored row per run or per voted answer.
Example rows of `scored_runs.csv`:
""")

code("""
runs.sample(5, random_state=1)[KEY + ["property", "truth", "answer", "outcome", "tokens_in", "cost_usd"]]
""")

code("""
def f1(c, w, a):
    d = 2 * c + 2 * w + a
    return np.where(d > 0, 2 * c / np.maximum(d, 1), np.nan)

def f1_ign(c, w, a):
    d = 2 * c + w
    return np.where(d > 0, 2 * c / np.maximum(d, 1), np.nan)

SCORES = {"F1": f1, "F1 (ign.)": f1_ign}

def counts_by_task(df):
    t = df.assign(c=df.outcome == "correct", w=df.outcome == "wrong", a=df.outcome == "abstained")
    return t.groupby("task_id")[["c", "w", "a"]].sum()

def boot(df, n=2000, seed=0):
    k = counts_by_task(df).to_numpy(float)
    s = k[np.random.default_rng(seed).integers(0, len(k), (n, len(k)))].sum(axis=1)
    return {name: np.nanpercentile(f(s[:, 0], s[:, 1], s[:, 2]), [2.5, 97.5]) for name, f in SCORES.items()}

def summary(df, ci=True):
    c, w, a = [int((df.outcome == o).sum()) for o in ("correct", "wrong", "abstained")]
    n = c + w + a
    row = {"n": n, "correct": c, "wrong": w, "abstained": a,
           "precision": c / (c + w) if c + w else np.nan, "recall": c / max(n, 1), "F1": float(f1(c, w, a)),
           "F1 (ign.)": float(f1_ign(c, w, a)), "abstention": a / max(n, 1), "error rate": w / max(n, 1)}
    if ci:
        b = boot(df)
        row["F1 CI"] = f"{b['F1'][0]:.3f}–{b['F1'][1]:.3f}"
        row["F1 (ign.) CI"] = f"{b['F1 (ign.)'][0]:.3f}–{b['F1 (ign.)'][1]:.3f}"
    return row

def summary_by(df, by, ci=True):
    return pd.DataFrame([{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **summary(g, ci)}
                         for k, g in df.groupby(by, observed=True)])

def paired(a, b, n=2000, seed=0):
    # a − b over the tasks both have
    ka, kb = counts_by_task(a), counts_by_task(b)
    ids = ka.index.intersection(kb.index)
    ka, kb = ka.loc[ids].to_numpy(float), kb.loc[ids].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))
    sa, sb = ka[idx].sum(axis=1), kb[idx].sum(axis=1)
    out = {}
    for name, f in SCORES.items():
        d = f(sa[:, 0], sa[:, 1], sa[:, 2]) - f(sb[:, 0], sb[:, 1], sb[:, 2])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        out[name] = (float(f(*ka.sum(axis=0)) - f(*kb.sum(axis=0))), lo, hi)
    return out

def stability(pr):
    # per-run F1s and the share of tasks whose runs agree on the outcome
    per = [summary(g, False) for _, g in pr.groupby("run_index")]
    agree = pr.groupby("task_id").outcome.nunique().eq(1).mean()
    return {"F1 sd over runs": np.std([p["F1"] for p in per]), "F1 (ign.) sd over runs": np.std([p["F1 (ign.)"] for p in per]),
            "run agreement": agree}

def effort(pr, correct):
    per_run = pr.drop_duplicates(KEY)
    return {"tokens in / run": per_run.tokens_in.mean(), "tokens out / run": per_run.tokens_out.mean(),
            "LLM calls / run": per_run.llm_calls.mean(), "searches / run": per_run.search_calls.mean(),
            "latency s / run": per_run.latency_s.mean(), "energy Wh / run": 1000 * per_run.energy_kwh.mean(),
            "cost $ / 1k runs": 1000 * per_run.cost_usd.mean(),
            "cost $ / correct": per_run.cost_usd.sum() / correct if correct else np.nan}

def full_table(v, r, by):
    rows = []
    for k, g in v.groupby(by, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        sel = np.logical_and.reduce([r[c] == x for c, x in zip(by, k)])
        s = summary(g)
        rows.append({**dict(zip(by, k)), **s, **stability(r[sel]), **effort(r[sel], s["correct"])})
    return pd.DataFrame(rows)

FMT = {"precision": "{:.3f}", "recall": "{:.3f}", "F1": "{:.3f}", "F1 (ign.)": "{:.3f}", "abstention": PCT, "error rate": PCT,
       "F1 sd over runs": "{:.3f}", "F1 (ign.) sd over runs": "{:.3f}", "run agreement": PCT,
       "tokens in / run": "{:,.0f}", "tokens out / run": "{:,.0f}", "LLM calls / run": "{:.2f}", "searches / run": "{:.2f}",
       "latency s / run": "{:.1f}", "energy Wh / run": "{:.3f}", "cost $ / 1k runs": "{:.3f}", "cost $ / correct": "{:.4f}"}
QUALITY = ["n", "correct", "wrong", "abstained", "precision", "recall", "F1", "F1 CI", "F1 (ign.)", "F1 (ign.) CI", "abstention", "error rate"]
STABLE = ["F1 sd over runs", "F1 (ign.) sd over runs", "run agreement"]
EFFORT = ["tokens in / run", "tokens out / run", "LLM calls / run", "searches / run", "latency s / run", "energy Wh / run",
          "cost $ / 1k runs", "cost $ / correct"]

def show(t, index, cols, grad=True):
    t = t.copy()
    if "model" in t: t["model"] = t.model.map(MNAME)
    if "variant" in t: t["variant"] = t.variant.map(lambda v: VNAME.get(v, v))
    s = t.set_index(index)[[c for c in cols if c in t]].style.format({k: f for k, f in FMT.items() if k in cols}, na_rep="–")
    if grad:
        for col in ("F1", "F1 (ign.)"):
            if col in cols:
                s = s.background_gradient(cmap=SEQ, subset=[col], vmin=0, vmax=0.8)
    return s

VA, RA = voted[voted.strategy == "A"], runs[runs.strategy == "A"]
VB, RB = voted[voted.strategy == "B"], runs[runs.strategy == "B"]
""")

# --------------------------------------------------------------------------------------------- A
md("""
## 1. Strategy A (LLM only)

### 1.1 Each model, all 9 prompts pooled
""")

code("""
TA_model = full_table(VA, RA, ["model"]).set_index("model").reindex(MODELS).reset_index()
show(TA_model, ["model"], QUALITY)
""")

code("""
show(TA_model, ["model"], STABLE + EFFORT, grad=False)
""")

md("""
### 1.2 Every model × prompt
""")

code("""
TA = full_table(VA, RA, ["model", "variant"])
TA["model"] = pd.Categorical(TA.model, MODELS, ordered=True); TA["variant"] = pd.Categorical(TA.variant, PROMPTS, ordered=True)
TA = TA.sort_values(["model", "variant"])
show(TA, ["model", "variant"], QUALITY)
""")

code("""
show(TA, ["model", "variant"], STABLE + EFFORT, grad=False)
""")

md("""
### 1.3 Model × prompt heatmaps (one figure per score)
""")

code("""
def heat(t, col, title, vmax, name):
    p = t.pivot(index="model", columns="variant", values=col).reindex(index=MODELS, columns=PROMPTS)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.imshow(p.to_numpy(float), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 * vmax else INK)
    ax.set_xticks(range(len(PROMPTS)), [VNAME[x] for x in PROMPTS], rotation=30, ha="right")
    ax.set_yticks(range(len(MODELS)), [MNAME[m] for m in MODELS]); ax.grid(False); ax.set_title(title)
    fig.tight_layout(); save(fig, name); plt.show()

TAp = TA.astype({"model": str, "variant": str})
for col, title, suf in METRICS:
    heat(TAp, col, f"Strategy A: {title}", 0.8, f"a_heatmap_{suf}")
heat(TAp, "abstention", "Strategy A: abstention rate", 1.0, "a_heatmap_abstention")
""")

md("""
### 1.4 Precision against recall

Each point is a model × prompt. Up and left: few but reliable answers (abstains often); right: answers more,
including more wrong ones.
""")

code("""
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in MODELS:
    d = TAp[TAp.model == m]
    ax.scatter(d.recall, d.precision, s=60, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1.5, label=MNAME[m])
ax.set_xlabel("recall = c / n"); ax.set_ylabel("precision = c / (c + w)")
ax.set_title("Strategy A: precision and recall of each model × prompt")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); fig.tight_layout(); save(fig, "a_precision_recall"); plt.show()
""")

md("""
### 1.5 Best prompt of each model

Chosen on these results, under each score separately.
""")

code("""
BEST = {}
rows = []
for m in MODELS:
    d = TAp[TAp.model == m]
    for col, _, _ in METRICS:
        r = d.loc[d[col].idxmax()]
        BEST[(m, col)] = r.variant
        rows.append({"model": MNAME[m], "score": col, "best prompt": VNAME[r.variant], "value": r[col],
                     "CI": r[f"{col} CI"], "zero-shot": d.set_index("variant").loc["zero", col],
                     "mean of 9 prompts": d[col].mean(), "abstention of best": r.abstention})
pd.DataFrame(rows).set_index(["model", "score"]).style.format({"value": "{:.3f}", "zero-shot": "{:.3f}",
                                                                "mean of 9 prompts": "{:.3f}", "abstention of best": PCT})
""")

# --------------------------------------------------------------------------------------------- B
md("""
## 2. Strategy B (single-turn agent)

Two variants on the same plan and searches: **search (evidence only)**, where every value must cite retrieved
text, and **search + memory**, where the model may also answer from its own knowledge.

### 2.1 Every model × variant
""")

code("""
TB = full_table(VB, RB, ["model", "variant"])
TB["model"] = pd.Categorical(TB.model, MODELS, ordered=True); TB["variant"] = pd.Categorical(TB.variant, B_VARIANTS, ordered=True)
TB = TB.sort_values(["model", "variant"])
show(TB, ["model", "variant"], QUALITY)
""")

code("""
show(TB, ["model", "variant"], STABLE + EFFORT, grad=False)
""")

md("""
In *search + memory* the plan call and the searches are those of the evidence-only run (replayed), and their
tokens, cost and searches are counted again, so both variants report the cost of the whole pipeline.

### 2.2 Each score per model (one figure per score)
""")

code("""
TBp = TB.astype({"model": str, "variant": str})
BCOLOR = {"search_evidence": BLUE, "search_memory": AQUA}

def bars(t, col, title, name, conds, colors, ylabel_names, xlim=1.0):
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(MODELS) + 1.2))
    h = 0.8 / len(conds)
    for j, c in enumerate(conds):
        d = t[t.cond == c].set_index("model").reindex(MODELS)
        y = np.arange(len(MODELS)) + (j - (len(conds) - 1) / 2) * h
        ax.barh(y, d[col], height=h * 0.9, color=colors[c], label=ylabel_names[c])
        for yy, v in zip(y, d[col]):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(MODELS)), [MNAME[m] for m in MODELS]); ax.invert_yaxis()
    ax.set_xlim(0, xlim); ax.set_title(title); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9); fig.tight_layout(); save(fig, name); plt.show()

for col, title, suf in METRICS + [("abstention", "abstention rate", "abstention"), ("precision", "precision", "precision"), ("recall", "recall", "recall")]:
    bars(TBp.assign(cond=TBp.variant), col, f"Strategy B: {title}", f"b_{suf}", B_VARIANTS, BCOLOR, VNAME)
""")

md("""
### 2.3 What memory adds (paired, same tasks)

*search + memory* − *search (evidence only)*, and the task-by-task counts: tasks right only with memory, right
only with evidence, and answers that changed from abstained to wrong.
""")

code("""
def mcnemar(b, c):
    # exact two-sided McNemar test on the discordant counts
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def taskwise(a, b):
    m = a[["task_id", "prop_key", "outcome"]].merge(b[["task_id", "prop_key", "outcome"]], on=["task_id", "prop_key"], suffixes=("_a", "_b"))
    ca, cb = m.outcome_a == "correct", m.outcome_b == "correct"
    return {"tasks": len(m), "both right": int((ca & cb).sum()), "only first right": int((ca & ~cb).sum()),
            "only second right": int((~ca & cb).sum()), "neither": int((~ca & ~cb).sum()),
            "McNemar p": mcnemar(int((ca & ~cb).sum()), int((~ca & cb).sum())),
            "abstained → wrong": int(((m.outcome_b == "abstained") & (m.outcome_a == "wrong")).sum()),
            "abstained → right": int(((m.outcome_b == "abstained") & ca).sum())}

rows = []
for m in MODELS:
    mem, ev = VB[(VB.model == m) & (VB.variant == "search_memory")], VB[(VB.model == m) & (VB.variant == "search_evidence")]
    p = paired(mem, ev)
    rows.append({"model": MNAME[m], **{f"Δ {k}": p[k][0] for k in SCORES},
                 **{f"CI Δ {k}": f"{p[k][1]:+.3f} to {p[k][2]:+.3f}" for k in SCORES}, **taskwise(mem, ev)})
mem_tab = pd.DataFrame(rows).rename(columns={"only first right": "only with memory", "only second right": "only evidence-only"})
mem_tab.set_index("model").style.format({"Δ F1": "{:+.3f}", "Δ F1 (ign.)": "{:+.3f}", "McNemar p": "{:.3g}"})
""")

# --------------------------------------------------------------------------------------------- A vs B
md("""
## 3. A against B

Four configurations per model, all on the same 243 tasks and voted over 3 runs:

- **A zero-shot**: no examples, no CoT, the A prompt closest to B's prompts;
- **A best**: the model's best A prompt under that score (chosen on these data, so favoured);
- **B evidence**: search, evidence only;
- **B memory**: search + memory.

### 3.1 All metrics side by side
""")

code("""
CONDS = ["A zero-shot", "A best", "B evidence", "B memory"]
CCOLOR = {"A zero-shot": MUTED, "A best": ORANGE, "B evidence": BLUE, "B memory": AQUA}

def cond(m, c, score="F1"):
    if c == "A zero-shot": return VA[(VA.model == m) & (VA.variant == "zero")], RA[(RA.model == m) & (RA.variant == "zero")]
    if c == "A best": return VA[(VA.model == m) & (VA.variant == BEST[(m, score)])], RA[(RA.model == m) & (RA.variant == BEST[(m, score)])]
    v = "search_evidence" if c == "B evidence" else "search_memory"
    return VB[(VB.model == m) & (VB.variant == v)], RB[(RB.model == m) & (RB.variant == v)]

rows = []
for m in MODELS:
    for c in CONDS:
        v, r = cond(m, c)
        s = summary(v)
        label = f"A best ({VNAME[BEST[(m, 'F1')]]})" if c == "A best" else c
        rows.append({"model": m, "configuration": label, "cond": c, **s, **stability(r), **effort(r, s["correct"])})
AB = pd.DataFrame(rows)
show(AB, ["model", "configuration"], QUALITY)
""")

md("""
*A best* here is chosen by F1; the F1 (ign.) figures below use the best prompt under F1 (ign.).
""")

code("""
show(AB, ["model", "configuration"], STABLE + EFFORT, grad=False)
""")

md("""
### 3.2 Each score per model (one figure per score)
""")

code("""
for col, title, suf in METRICS:
    rows = []
    for m in MODELS:
        for c in CONDS:
            rows.append({"model": m, "cond": c, **summary(cond(m, c, col)[0], False)})
    bars(pd.DataFrame(rows), col, f"A against B: {title}", f"ab_{suf}", CONDS, CCOLOR, {c: c for c in CONDS})
for col, title, suf in [("abstention", "abstention rate", "abstention"), ("precision", "precision", "precision"), ("recall", "recall", "recall")]:
    bars(AB, col, f"A against B: {title}", f"ab_{suf}", CONDS, CCOLOR, {c: c for c in CONDS})
""")

md("""
### 3.3 Paired differences B − A (one figure per score)

Point = difference over the 243 tasks, line = 95% paired bootstrap interval. Right of 0: search is better.
""")

code("""
DIFFS = [("B evidence", "A zero-shot"), ("B evidence", "A best"), ("B memory", "A zero-shot"), ("B memory", "A best")]
DCOLOR = {d: col for d, col in zip(DIFFS, [BLUE, "#1c5cab", AQUA, "#0f7a54"])}
diff_rows = []
for col, title, suf in METRICS:
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(MODELS) + 1.2))
    h = 0.8 / len(DIFFS)
    for j, (b, a) in enumerate(DIFFS):
        for i, m in enumerate(MODELS):
            d, lo, hi = paired(cond(m, b, col)[0], cond(m, a, col)[0])[col]
            diff_rows.append({"score": col, "model": MNAME[m], "comparison": f"{b} − {a}", "Δ": d, "lo": lo, "hi": hi})
            y = i + (j - (len(DIFFS) - 1) / 2) * h
            ax.plot([lo, hi], [y, y], color=DCOLOR[(b, a)], linewidth=2)
            ax.scatter([d], [y], s=40, color=DCOLOR[(b, a)], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=f"{b} − {a}" if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(MODELS)), [MNAME[m] for m in MODELS]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Δ {title}"); ax.set_title(f"B − A, {title}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9); fig.tight_layout(); save(fig, f"ab_diff_{suf}"); plt.show()
D = pd.DataFrame(diff_rows)
D["significant"] = np.where(D.lo > 0, "B better", np.where(D.hi < 0, "A better", ""))
D["CI"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(D.lo, D.hi)]
D.pivot_table(index=["model", "comparison"], columns="score", values=["Δ", "CI", "significant"], aggfunc="first").style.format(
    lambda v: f"{v:+.3f}" if isinstance(v, float) else v)
""")

md("""
### 3.4 Task by task

For each model, the 243 tasks split by which configuration got them right (*first* = B, *second* = A), with the
exact McNemar test on the tasks where only one is right. *abstained → wrong*: A abstained and B answered wrongly;
*abstained → right*: A abstained and B answered right.
""")

code("""
rows = []
for m in MODELS:
    for b, a in DIFFS:
        rows.append({"model": MNAME[m], "comparison": f"{b} vs {a}", **taskwise(cond(m, b)[0], cond(m, a)[0])})
tw = pd.DataFrame(rows).rename(columns={"only first right": "only B right", "only second right": "only A right"})
tw.set_index(["model", "comparison"]).style.format({"McNemar p": "{:.3g}"})
""")

md("""
### 3.5 By G stratum (models pooled)

Whether search helps more when the given properties are rare (low) or common (high). *A best* = each model's
best prompt under F1.
""")

code("""
rows = []
for c in CONDS:
    v = pd.concat([cond(m, c)[0] for m in MODELS])
    for (dim, name) in [("G_stratum", "G popularity"), ("H_stratum", "H popularity"), ("node_stratum", "node centrality")]:
        for k, g in v.groupby(dim):
            rows.append({"dimension": name, "stratum": k, "configuration": c, **summary(g, False)})
S = pd.DataFrame(rows)
S["stratum"] = pd.Categorical(S.stratum, ["low", "mid", "high"], ordered=True)
S["configuration"] = pd.Categorical(S.configuration, CONDS, ordered=True)
P = S.pivot_table(index=["dimension", "stratum"], columns="configuration", values=["F1", "F1 (ign.)", "abstention"], observed=True)
P.style.format("{:.3f}").format(PCT, subset=[c for c in P.columns if c[0] == "abstention"])
""")

md("""
### 3.6 Quality against cost (one figure per score)

Each point is a model × configuration; x = cost per 1,000 runs (log scale). Search costs more tokens and calls.
""")

code("""
MARK = {"A zero-shot": "o", "A best": "s", "B evidence": "^", "B memory": "D"}
for col, title, suf in METRICS:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in MODELS:
        for c in CONDS:
            v, r = cond(m, c, col)
            ax.scatter(effort(r, 1)["cost $ / 1k runs"], summary(v, False)[col], s=70, marker=MARK[c], color=MCOLOR[m],
                       edgecolor=SURFACE, linewidth=1.5)
    ax.set_xscale("log"); ax.set_xlabel("cost per 1,000 runs (USD, log scale)"); ax.set_ylabel(title)
    h1 = [plt.Line2D([], [], marker="o", linestyle="", color=MCOLOR[m], label=MNAME[m]) for m in MODELS]
    h2 = [plt.Line2D([], [], marker=MARK[c], linestyle="", color=INK2, label=c) for c in CONDS]
    ax.legend(handles=h1 + h2, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.set_title(f"{title} against cost"); fig.tight_layout(); save(fig, f"ab_cost_{suf}"); plt.show()
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")

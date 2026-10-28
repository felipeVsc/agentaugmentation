"""Builds notebooks/dolap2026_ollama.ipynb.

    venv/bin/python notebooks/build_dolap2026_notebook.py notebooks/dolap2026_discussion.json
    venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/dolap2026_ollama.ipynb

The discussion cells come from the JSON file (written after looking at the results)."""

import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "dolap2026_ollama.ipynb"
disc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
DISC = json.loads(disc_path.read_text()) if disc_path and disc_path.exists() else {}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def discussion(key, placeholder="*Discussion written after the full run.*"):
    md(DISC.get(key, placeholder))


md("""
# Replicating the DOLAP 2026 experiment with open-weight models on Ollama

**What this notebook does.** It runs Strategy A (LLM only, no tools) on the **36 tasks of the published paper**
*On the Efficacy of Using LLMs for Context Driven Entity Augmentation in Property Graphs* (DOLAP 2026), with the
**9 published prompts**, **5 runs** per configuration and **both conditions** (without and with graph context),
using an open-weight model served locally by Ollama. It then scores the answers two ways and compares them with
the six proprietary models of the paper.

**Scope of this run: one model (Qwen3 4B Instruct), to check that the pipeline and the scoring are correct** before
running more models. Other models are added by running them (section 2); every table and chart picks them up.

| | Published run | This run |
|---|---|---|
| Tasks | 36 (9 airports × 4 H/G quadrants) | the same 36, imported unchanged from `test_results.csv` |
| Prompts | 9 (compact, zero, 1/3/5-shot, the same with CoT) | the same 9 files, unchanged |
| Input | `input: prop:value,…` + context | identical text without context; with context, the same procedure (a slice of 3–8 context entries) with a fixed seed, because the published slices were not saved |
| Runs | 5, then vote | 5, then the same vote |
| Models | Gemini 2.5 Pro/Flash, GPT-5/5 mini, Claude Sonnet/Haiku 4.5 | Qwen3 4B Instruct (Ollama, Q4_K_M); more can be added |
| Answers scored | voted answers | voted answers, same tolerance rules |

**Two scorings** (details in `src/evaluation/dolap.py`):

- **Published scoring** reproduces the paper's evaluation notebook: every property the model answered and that has a
  ground-truth value is compared (also properties that were not asked), unanswered properties are ignored, and F1 is
  computed with `y_true` all ones. Precision is therefore 1 by construction and **accuracy = recall = correct / answered**,
  F1 = 2·acc / (1 + acc). This is what Table 3 of the paper reports.
- **Standard scoring** (EVALUATION_SPEC): only the hidden properties H that have a ground truth. Each one is *correct*,
  *wrong* (a confident wrong value) or *abstained* (no value). Precision = correct / answered, recall = correct / |H|.
  This shows what the published scoring hides: abstentions and wrong answers.
""")

code("""
from pathlib import Path
import json, os, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
from evaluation import dolap as D

TASKS = ROOT / "tasks/airports/dolap2026/tasks.csv"
RESULTS = ROOT / "results/airports/dolap2026/strategy_a"
CONFIG = ROOT / "configs/strategy_a_dolap2026.yaml"
NODE_VALUES = ROOT / "data/dolap2026/ground_truth.csv"
PUBLISHED = ROOT / "data/dolap2026/published_results"

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

# Chart style: light surface, recessive axes, text in ink tokens (dataviz reference palette).
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#c9c8c2"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "legend.frameon": False, "figure.dpi": 110,
})
OLLAMA_ORDER = ["qwen3-4b-instruct", "nemotron-mini-4b", "llama3.2-1b"]
PUBLISHED_ORDER = ["Claude Sonnet 4.5", "Claude Haiku 4.5", "GPT-5", "GPT-5 Mini", "Gemini 2.5 Pro", "Gemini 2.5 Flash"]
""")

md("""
## 1. The task set

The 36 published tasks, imported with
`tasks import-published --config configs/airports_dolap2026.yaml --no-graph --results data/dolap2026/test_results.csv --node-values data/dolap2026/ground_truth.csv`.
The recomputed difficulty must equal the published one: this checks that popularity, PageRank normalization and the
specificity formula are the ones of the paper.
""")

code("""
tasks = D.load_tasks(TASKS)
view = tasks[["node_id", "node_stratum", "difficulty_prop", "H_names", "G_drawn", "size_G", "difficulty", "published_difficulty"]].copy()
view["G_drawn"] = view["G_drawn"].map(json.loads)
print(f"{len(tasks)} tasks, {tasks.node_id.nunique()} airports")
print("max |recomputed − published difficulty| =", float((tasks.difficulty - tasks.published_difficulty).abs().max()))
view.sort_values(["node_stratum", "node_id", "difficulty_prop"])
""")

code("""
node_values = D.load_node_values(NODE_VALUES)
h_pairs = [(t.node_id, p) for t in tasks.itertuples() for p in t.H_names]
no_gt = pd.Series([p for n, p in h_pairs if p not in node_values.get(n, {})])
print(f"H properties: {len(h_pairs)}; without a ground-truth value (not scored, as in the paper): {len(no_gt)} -> "
      f"{no_gt.value_counts().to_dict()}")
print("Tasks by number of G values actually given (published: 'G' drawn may include properties the node lacks):",
      tasks["size_G"].value_counts().sort_index().to_dict())
""")

md("""
## 2. Running the models

One model takes about 1 hour on an Apple M4 (3,240 runs: 36 tasks × 18 variants × 5 runs). Every answer is cached,
so re-running is free; set `RUN = True` and list models in `MODELS` to run missing runs or new models
(ids from `configs/strategy_a_dolap2026.yaml`, e.g. `nemotron-mini-4b`, `llama3.2-1b`, `mistral-7b`).
""")

code("""
RUN = False
MODELS = ["qwen3-4b-instruct"]
if RUN:
    for m in MODELS:
        subprocess.run(["strategy-a", "run", "--config", str(CONFIG), "--models", m], check=True)
mpath = RESULTS / "run_manifest.json"
manifest = json.loads(mpath.read_text()) if mpath.exists() else {"model_details": {}}  # written when a model's run ends
details = pd.DataFrame([{"model": k, "ollama tag": v.get("model"), "digest": str(v.get("digest"))[:12],
                         "params": (v.get("details") or {}).get("parameter_size"),
                         "quantization": (v.get("details") or {}).get("quantization_level"),
                         "Ollama": v.get("ollama_version")} for k, v in manifest["model_details"].items()])
details
""")

md("## 3. Loading, voting and scoring")

code("""
runs = D.load_runs(RESULTS)
models_present = [m for m in OLLAMA_ORDER if m in set(runs.model)]
done = runs.groupby("model").size().reindex(models_present)
print("runs per model:", done.to_dict(), "(complete = 3240)")
ollama_votes = D.voted_from_runs(runs, tasks)
published_votes = D.load_published_votes(PUBLISHED, tasks)
votes = pd.concat([ollama_votes, published_votes], ignore_index=True)
sp = D.score_published(votes, node_values)      # one row per compared property (paper's way)
ss = D.score_standard(votes, tasks, node_values) # one row per H property with a ground truth
print(f"configurations: {len(votes)}; compared properties (published scoring): {len(sp)}; scored H properties: {len(ss)}")
cov = D.coverage(votes)
cov[cov.airports < 9]
""")

md("""
The table above lists the model × condition pairs that do **not** cover all 9 airports. In the published files, the
with-context runs of Gemini 2.5 Pro and Gemini 2.5 Flash only contain 4 airports (PAKK, VEGT, YPPH, ZUBJ: the three
most central ones and one of medium centrality). Their with-context results are computed on those airports only and
are not comparable with the other rows.
""")

md("""
### 3.1 Check: the published scoring reproduces Table 3 of the paper

Table 3 lists the best configuration of each proprietary model. Recomputed here from the published voted answers:
""")

code("""
pm = D.published_metrics(sp, ["source", "model", "context", "prompt"]).set_index(["model", "context", "prompt"])
table3 = [("Claude Sonnet 4.5", .80, .66, True, "few5"), ("Claude Haiku 4.5", .54, .37, True, "compact"),
          ("GPT-5", .67, .50, False, "3cot"), ("GPT-5 Mini", .57, .40, False, "few3"),
          ("Gemini 2.5 Pro", .65, .48, True, "5cot"), ("Gemini 2.5 Flash", .61, .44, True, "5cot")]
check = pd.DataFrame([{"model": m, "context": c, "prompt (from effort)": p, "paper F1": f, "paper recall": r,
                       "recomputed F1": pm.loc[(m, c, p), "f1"], "recomputed recall": pm.loc[(m, c, p), "recall"],
                       "compared": int(pm.loc[(m, c, p), "compared"])} for m, f, r, c, p in table3])
sonnet_best = pm.loc["Claude Sonnet 4.5"].sort_values("f1", ascending=False).head(3)
display(check.round(3))
print("Claude Sonnet 4.5, best configurations by published F1:")
sonnet_best.round(3)
""")

discussion("check")

md("""
### 3.2 Format compliance

The published prompts ask for CSV `property,value` only. A run whose answer has no CSV is a format failure: the paper
counted only answers in the requested structure, so these runs contribute nothing (no repair call here either).
""")

code("""
comp = runs.assign(ok=runs.status.eq("ok")).groupby(["model", "variant"]).ok.mean().unstack()
order_cols = [p for p in D.PROMPT_ORDER] + [p + "_ctx" for p in D.PROMPT_ORDER]
comp = comp.reindex(index=models_present, columns=[c for c in order_cols if c in comp.columns])
print("Share of runs answered in CSV format:")
comp.round(2)
""")

md("## 4. Results")
md("""
### 4.1 Best configuration per model (Table 3, extended)

For each model and condition: the prompt with the best **published** F1, as in the paper, and next to it the
**standard** metrics of that same configuration. `compared` is the number of answered properties the published
F1 rests on.
""")

code("""
std = D.standard_metrics(ss, ["source", "model", "context", "prompt"])
both = D.published_metrics(sp, ["source", "model", "context", "prompt"]).merge(
    std, on=["source", "model", "context", "prompt"], suffixes=("_pub", "_std"))
best = (both.sort_values(["f1_pub", "compared"], ascending=False)
            .groupby(["model", "context"]).head(1)
            .sort_values(["source", "model", "context"]))
cols = ["source", "model", "context", "prompt", "compared", "accuracy", "f1_pub",
        "properties", "precision_std", "recall_std", "f1_std", "abstention_rate", "error_rate"]
best_tbl = best[cols].rename(columns={"accuracy": "pub acc (=recall)", "f1_pub": "pub F1", "precision_std": "precision",
                                      "recall_std": "recall", "f1_std": "F1"})
best_tbl.round(3)
""")

md("""
### 4.2 Overall, over all 9 prompts

Choosing the best prompt after seeing the results favours models with more variance (and the best Sonnet configuration
rests on 3 answers). The fairer summary averages over all prompts: micro-averaged standard metrics over every scored
H property, with a 95% bootstrap interval over the 36 tasks.
""")

code("""
rng = np.random.default_rng(0)
def boot_f1(g, n=1000):
    per_task = g.groupby("task_id").outcome.value_counts().unstack(fill_value=0).reindex(columns=["correct", "wrong", "abstained"], fill_value=0)
    arr = per_task.to_numpy()
    out = []
    for _ in range(n):
        c, w, a = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        p = c / (c + w) if c + w else 0.0
        r = c / (c + w + a)
        out.append(2 * p * r / (p + r) if c else 0.0)
    return np.percentile(out, [2.5, 97.5])

overall = D.standard_metrics(ss, ["source", "model", "context"])
ci = ss.groupby(["model", "context"]).apply(boot_f1, include_groups=False)
overall["F1 95% CI"] = [f"[{ci[(m, c)][0]:.2f}, {ci[(m, c)][1]:.2f}]" for m, c in zip(overall.model, overall.context)]
pub_overall = D.published_metrics(sp, ["model", "context"])[["model", "context", "accuracy", "compared"]]
overall = overall.merge(pub_overall, on=["model", "context"]).rename(columns={"accuracy": "pub accuracy"})
overall = overall.sort_values(["source", "context", "f1"], ascending=[True, True, False])
overall[["source", "model", "context", "properties", "correct", "wrong", "abstained", "precision", "recall", "f1",
         "F1 95% CI", "abstention_rate", "error_rate", "pub accuracy", "compared"]].round(3)
""")

code("""
def model_order(df):
    present = set(df.model)
    return [m for m in OLLAMA_ORDER + PUBLISHED_ORDER if m in present]

def outcome_chart(df, title):
    order = model_order(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(order) + 1.6), sharey=True)
    for ax, ctx in zip(axes, [False, True]):
        sub = df[df.context == ctx].set_index("model").reindex(order)
        n = sub["properties"].to_numpy(float)
        left = np.zeros(len(order))
        for col, color, label in [("correct", GOOD, "Correct"), ("wrong", CRITICAL, "Wrong"), ("abstained", NEUTRAL, "Abstained")]:
            share = np.nan_to_num(sub[col].to_numpy(float) / n)
            ax.barh(range(len(order)), share, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2, label=label)
            for i, (l, s) in enumerate(zip(left, share)):
                if s >= 0.08:
                    ax.text(l + s / 2, i, f"{s:.0%}", ha="center", va="center", fontsize=8.5,
                            color="white" if col != "abstained" else INK)
            left += share
        ax.set_title("With context" if ctx else "Without context", fontsize=11)
        ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.grid(axis="y", visible=False)
        ax.set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()  # shared y: flip once, first model on top
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plt.show()

outcome_chart(overall, "Outcome of every hidden property, all prompts (standard scoring)")
""")

md("""
**What counts as wrong.** The paper's tolerance (token overlap ≥ 0.75, relative similarity ≥ 0.75, 10 km for
coordinates) is applied as published. Most wrong answers are genuinely wrong (another airport, another code), but a
part are unit or format mismatches: runway lengths in metres against feet, `distCity` in km against metres,
coordinates written `lat/lon` against `POINT(lon,lat)`. They affect every model alike. A sample:
""")
code("""
w = ss[ss.outcome == "wrong"].merge(votes[["model", "variant", "task_id", "answer"]], on=["model", "variant", "task_id"])
w["predicted"] = [a.get(p) for a, p in zip(w.answer, w.property)]
w["ground truth"] = [node_values[n].get(p) for n, p in zip(w.node, w.property)]
print("Wrong answers by property:", w.property.value_counts().head(8).to_dict())
w[w.property.isin(["runwaysLength", "distCity", "altitude", "location", "iata", "name"])].drop_duplicates(["property", "node"]) \
    [["model", "node", "property", "predicted", "ground truth"]].head(14)
""")

discussion("overall")

md("""
### 4.3 Prompts and effort

Standard F1 per model and prompt (rows: models; columns: prompts ordered by prompt formulation effort). PFE is the
number of tokens of the prompt text (cl100k_base), as in Figure 1 of the paper.
""")

code("""
from strategy_a.config import load_config
from strategy_a.runner import Runner
_r = Runner(load_config(CONFIG), connect=False, log=lambda m: None); _r.prepare()
pfe = pd.Series({v: n for (v, m), n in _r.pfe.items()})  # same for every model with cl100k_base
by_prompt = D.standard_metrics(ss, ["model", "context", "prompt"])
prompts_by_effort = sorted(D.PROMPT_ORDER, key=lambda p: pfe.get(p, 0))
fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(model_order(by_prompt)) + 2), sharey=True)
order = model_order(by_prompt)
for ax, ctx in zip(axes, [False, True]):
    grid = by_prompt[by_prompt.context == ctx].pivot(index="model", columns="prompt", values="f1").reindex(index=order, columns=prompts_by_effort)
    im = ax.imshow(grid.to_numpy(float), cmap=SEQ, vmin=0, vmax=max(0.6, np.nanmax(by_prompt.f1)), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.35 else INK)
    ax.set_xticks(range(len(prompts_by_effort)), [f"{p}\\n{int(pfe.get(p, 0))}" for p in prompts_by_effort], fontsize=8)
    ax.set_yticks(range(len(order)), order); ax.grid(False)
    ax.set_title(("With" if ctx else "Without") + " context — F1 (prompt, PFE tokens)", fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.8, label="standard F1")
plt.show()
""")

code("""
eff = by_prompt.assign(pfe=by_prompt.prompt.map(pfe))
corr = eff.groupby(["model", "context"]).apply(lambda g: g[["pfe", "f1"]].corr(method="spearman").iloc[0, 1], include_groups=False)
print("Spearman correlation between prompt effort (PFE) and F1, per model and condition:")
corr.unstack().reindex(model_order(eff)).round(2)
""")

discussion("prompts")

md("""
### 4.4 Does context help?

Paired comparison per model: the same tasks and prompts, with and without context (standard scoring, all prompts).
""")

code("""
ctx_cmp = overall.pivot(index="model", columns="context", values=["f1", "recall", "abstention_rate", "error_rate"])
ctx_delta = pd.DataFrame({m: ctx_cmp[m][True] - ctx_cmp[m][False] for m in ["f1", "recall", "abstention_rate", "error_rate"]})
ctx_delta = ctx_delta.reindex(model_order(overall))
print("With context minus without context (positive F1 = context helps):")
ctx_delta.round(3)
""")

discussion("context")

md("""
### 4.5 Difficulty

The paper's claim: accuracy falls with difficulty (H/G quadrants, node centrality), and context helps most for the
hardest quadrant. First by quadrant and node stratum, then the correlation between the difficulty score and the
F1 of each task.
""")

code("""
quad = D.standard_metrics(ss, ["source", "model", "context", "difficulty_prop"])
order = model_order(quad)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, ctx in zip(axes, [False, True]):
    sub = quad[quad.context == ctx]
    for m in order:
        s = sub[sub.model == m].set_index("difficulty_prop").reindex(D.QUADRANTS)
        is_local = m in OLLAMA_ORDER
        ax.plot(range(4), s.recall, marker="o", markersize=6 if is_local else 4, linewidth=2 if is_local else 1,
                color=BLUE if is_local else MUTED, alpha=1 if is_local else 0.7)
        last = s.recall.dropna()
        if len(last) and is_local:
            ax.text(3.08, last.iloc[-1], m, fontsize=8, va="center", color=INK2)
    ax.set_xticks(range(4), ["H easy\\nG easy", "H easy\\nG hard", "H hard\\nG easy", "H hard\\nG hard"])
    ax.set_xlim(-0.2, 3.9); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("With context" if ctx else "Without context", fontsize=11)
axes[0].set_ylabel("recall (share of hidden properties found)")
axes[0].plot([], [], color=BLUE, lw=2, marker="o", label="local Ollama models")
axes[0].plot([], [], color=MUTED, lw=1, marker="o", label="published proprietary models")
axes[0].legend(loc="upper right")
fig.suptitle("Recall by H/G difficulty quadrant (standard scoring, all prompts)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); plt.show()
""")
code("""
qt = quad.pivot_table(index=["model", "context"], columns="difficulty_prop", values="recall").reindex(columns=D.QUADRANTS)
print("Recall by quadrant (table view of the chart):")
qt.reindex(model_order(quad), level=0).round(2)
""")

code("""
strata = D.standard_metrics(ss, ["model", "context", "node_stratum"]).pivot_table(index=["model", "context"], columns="node_stratum", values="recall")
print("Recall by node centrality stratum (easy = most central airports):")
strata.reindex(columns=["easy", "medium", "hard"]).reindex(model_order(strata.reset_index()), level=0).round(2)
""")

code("""
tf = D.task_f1(ss)
rho = tf.groupby(["model", "context"]).apply(lambda g: g[["difficulty", "f1"]].corr(method="spearman").iloc[0, 1], include_groups=False)
print("Spearman correlation between task difficulty and task F1 (negative = harder tasks are answered worse):")
rho.unstack().reindex(model_order(tf)).round(2)
""")

discussion("difficulty")

md("""
### 4.6 Stability across the 5 runs

Before voting: for each hidden property, do the 5 runs give the same answer (the same normalized value, or all
abstain)? Low agreement means the vote matters a lot, and the result of one run is not reproducible.
""")

code("""
h_names = tasks.set_index("task_id").H_names.to_dict()
def agreement(g):
    vals = []
    for prop in h_names[g.name[2]]:
        answers = [D.coerce((a or {}).get(prop)) for a in g.answer]
        norm = [None if v is None else (round(v, 1) if isinstance(v, float) else " ".join(D.normalize_tokens(v))) for v in answers]
        vals.append(len(set(norm)) == 1)
    return np.mean(vals)
stab = runs.groupby(["model", "context_flag" if False else "variant", "task_id"]).apply(agreement, include_groups=False).rename("agree").reset_index()
stab["context"] = stab.variant.str.endswith("_ctx")
print("Share of hidden properties on which all 5 runs agree:")
stab.groupby(["model", "context"]).agree.mean().unstack().reindex(models_present).round(2)
""")

md("""
### 4.7 Cost and sustainability

Tokens, latency and energy over all runs of each local model. **The energy figures are estimates and undercount**:
on this Apple M4, CodeCarbon finds no GPU it can measure and estimates CPU and RAM power from load, while Ollama runs
on the Apple GPU (every value is flagged `estimated: true, gpu_measured: false` in `runs.jsonl`). They are not
comparable with the EcoLogits estimates of the paper.
""")

code("""
use = runs.groupby("model").agg(runs=("run_index", "size"), tokens_in=("tokens_in", "sum"), tokens_out=("tokens_out", "sum"),
                                latency_s=("latency_s", "sum"), energy_kwh=("energy_kwh", "sum"), co2e_kg=("co2e_kg", "sum")).reindex(models_present)
correct = ss[ss.source == "ollama"].assign(c=lambda d: d.outcome.eq("correct")).groupby("model").c.sum()
use["mean latency per run (s)"] = use.latency_s / use.runs
use["mean output tokens per run"] = use.tokens_out / use.runs
use["correct H values"] = correct
use["correct per 1k tokens"] = 1000 * use["correct H values"] / (use.tokens_in + use.tokens_out)
use["energy Wh (estimate)"] = 1000 * use.energy_kwh
use[["runs", "tokens_in", "tokens_out", "mean output tokens per run", "mean latency per run (s)",
     "energy Wh (estimate)", "correct H values", "correct per 1k tokens"]].round(3)
""")

code("""
lat = runs.groupby(["model", "variant"]).latency_s.mean().unstack().reindex(index=models_present)
lat = lat[[c for c in order_cols if c in lat.columns]]
fig, ax = plt.subplots(figsize=(12, 3.2))
x = np.arange(len(lat.columns))
w = 0.8 / max(1, len(lat.index))
for i, m in enumerate(lat.index):
    ax.bar(x + i * w - 0.4 + w / 2, lat.loc[m], width=w * 0.9, color=[BLUE, ORANGE, AQUA][i % 3], label=m)
ax.set_xticks(x, lat.columns, rotation=45, ha="right", fontsize=8); ax.grid(axis="x", visible=False)
ax.set_ylabel("seconds per run"); ax.legend(ncol=3, loc="upper left")
ax.set_title("Mean latency per run, by prompt and condition")
fig.tight_layout(); plt.show()
""")

discussion("cost")

md("## 5. Discussion")
discussion("main")
md("## 6. Limitations")
discussion("limits")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "agentaugmentation (venv)", "language": "python", "name": "agentaugmentation"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")

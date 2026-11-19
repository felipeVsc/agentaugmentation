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
using open-weight models served by Ollama. It then scores the answers two ways, **the DOLAP scoring and the scoring
of the current experiments**, and compares them with the six proprietary models of the paper (their published voted
answers, re-scored here).

**Models.** The current experiment lineup (gpt-oss:20b, gemma4:31b, gpt-oss:120b, deepseek-v4.1-flash; mistral-large-3
was retired by Ollama Cloud) with the model settings of the v5 / football runs, plus two earlier runs: Qwen3 4B Instruct
(local, 5 runs) and gemma4:31b (Ollama Cloud, 1 run, a test). A model appears in every table and chart as soon as its
runs exist (section 2).

| | Published run | This run |
|---|---|---|
| Tasks | 36 (9 airports × 4 H/G quadrants) | the same 36, imported unchanged from `test_results.csv` |
| Prompts | 9 (compact, zero, 1/3/5-shot, the same with CoT) | the same 9 files, unchanged |
| Input | `input: prop:value,…` + context | identical text without context; with context, the same procedure (a slice of 3–8 context entries) with a fixed seed, because the published slices were not saved |
| Runs | 5, then vote | 5, then the same vote |
| Models | Gemini 2.5 Pro/Flash, GPT-5/5 mini, Claude Sonnet/Haiku 4.5 | the current lineup (Ollama Cloud), Qwen3 4B (local), gemma4 1-run test |
| Answers scored | voted answers | voted answers, same tolerance rules |

**Two scorings** (code: `src/evaluation/dolap.py` and `src/evaluation/taskset.py`):

- **DOLAP scoring** (`dolap_score` below; called "published" in the code) reproduces the paper's evaluation notebook:
  every property the model answered and that has a ground-truth value is compared, **also properties that were not
  asked** (including values copied from the input), unanswered properties are ignored, the tolerance is lenient
  (token overlap over the *shorter* string), and F1 is sklearn's with `y_true` all ones. So **precision is 1 by
  construction** (there are no negatives to predict), **accuracy = recall = correct / answered**, and
  F1 = 2·acc / (1 + acc). This is what Table 3 of the paper reports, for the best of 18 configurations.
- **Our scoring** (`our_score` below; the function used by every current experiment, `evaluation.taskset.metrics`):
  only the hidden properties H that have a ground truth, with the stricter tolerance of the current runs
  (`dolap.matches_standard`). Each H property is *correct* (c), *wrong* (w, a confident wrong value) or *abstained*
  (a, no value). **Main metrics: F1 = 2c / (2c + 2w + a) and precision P = c / (c + w)**; beside them
  **F1 ign = 2c / (2c + w) = 2P / (1 + P)**, the DOLAP formula applied to the asked properties only;
  recall c / (c + w + a), abstention a / n, error w / n.
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
OLLAMA_ORDER = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-deepseek-v4.1-flash",
                "cloud-gemma4", "qwen3-4b-instruct", "nemotron-mini-4b", "llama3.2-1b"]
LINEUP = OLLAMA_ORDER[:4]
VIOLET, YELLOW = "#8a5cd8", "#d9a514"
MCOLOR = {"cloud-gpt-oss-20b": BLUE, "cloud-gemma4-31b": AQUA, "cloud-gpt-oss-120b": VIOLET,
          "cloud-deepseek-v4.1-flash": YELLOW, "cloud-gemma4": "#7fd3b4", "qwen3-4b-instruct": "#86b6ef"}
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

Each model has 3,240 runs (36 tasks × 18 variants × 5 runs). The lineup runs on Ollama Cloud with the key named in
`configs/strategy_a_dolap2026.yaml` (`api_key_env`), one model at a time; every answer is cached, so re-running is free.
Set `RUN = True` to run the models in `MODELS` (or run `strategy-a run --config configs/strategy_a_dolap2026.yaml
--models <id>` in a terminal), then re-execute the notebook.
""")

code("""
RUN = False
MODELS = LINEUP
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
cover = runs.groupby(["model", "variant"]).task_id.nunique().groupby("model").min()
partial = cover[cover < len(tasks)].index.tolist()
if partial:   # e.g. the 1-task connection test of cloud-gpt-oss-20b, or a model still running
    print("left out until all 36 tasks × 18 variants have runs:", partial)
    runs = runs[~runs.model.isin(partial)]
models_present = [m for m in OLLAMA_ORDER if m in set(runs.model)]
done = runs.groupby("model").size().reindex(models_present)
print("runs per model:", done.to_dict(), "(complete = 3240)")
ollama_votes = D.voted_from_runs(runs, tasks)
published_votes = D.load_published_votes(PUBLISHED, tasks)
votes = pd.concat([ollama_votes, published_votes], ignore_index=True)
sp = D.score_published(votes, node_values)      # one row per compared property (DOLAP scoring)
ss = D.score_standard(votes, tasks, node_values) # one row per H property with a ground truth (our scoring)
print(f"configurations: {len(votes)}; compared properties (published scoring): {len(sp)}; scored H properties: {len(ss)}")
cov = D.coverage(votes)
cov[cov.airports < 9]
""")

md("""
The two scoring functions used in the rest of the notebook. Both take the scored rows above and a grouping.
""")

code("""
from evaluation.taskset import metrics as taskset_metrics   # the scoring of every current experiment

def dolap_score(by):
    \"\"\"DOLAP: every answered property with a ground truth (asked or not), lenient tolerance, y_true all ones.
    precision = 1, accuracy = recall = correct / compared, F1 = 2·acc / (1 + acc).\"\"\"
    return D.published_metrics(sp, by).rename(columns={"f1": "DOLAP F1", "accuracy": "DOLAP accuracy",
                                                        "precision": "DOLAP precision", "compared": "compared"}).drop(columns="recall")

def our_score(by, data=None):
    \"\"\"Ours: the asked H properties, correct / wrong / abstained. Main: F1 and precision; F1 ign beside them.\"\"\"
    rows = []
    for k, g in (ss if data is None else data).groupby(by):
        m = taskset_metrics(g)
        rows.append({**dict(zip(by, k if isinstance(k, tuple) else (k,))), "n": m["properties"], "F1": m["f1"],
                     "precision": m["precision"], "F1 ign": 2 * m["correct"] / (2 * m["correct"] + m["wrong"]) if m["correct"] else 0.0,
                     "recall": m["recall"], "abstention": m["abstention_rate"], "error": m["error_rate"],
                     "correct": m["correct"], "wrong": m["wrong"], "abstained": m["abstained"]})
    return pd.DataFrame(rows)

side = dolap_score(["source", "model"]).merge(our_score(["source", "model"]), on=["source", "model"])
print("All 18 configurations pooled, both scorings side by side:")
side.set_index("model").reindex([m for m in OLLAMA_ORDER + PUBLISHED_ORDER if m in set(side.model)])[
    ["source", "compared", "DOLAP precision", "DOLAP F1", "n", "F1", "precision", "F1 ign", "abstention", "error"]].round(3)
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

For each model and condition: the prompt with the best **DOLAP** F1, as in the paper, and next to it **our**
metrics of that same configuration. `compared` is the number of answered properties the DOLAP F1 rests on.
""")

code("""
std = D.standard_metrics(ss, ["source", "model", "context", "prompt"])
both = D.published_metrics(sp, ["source", "model", "context", "prompt"]).merge(
    std, on=["source", "model", "context", "prompt"], suffixes=("_pub", "_std"))
best = (both.sort_values(["f1_pub", "compared"], ascending=False)
            .groupby(["model", "context"]).head(1)
            .sort_values(["source", "model", "context"]))
best["f1_ign"] = 2 * best["correct"] / (2 * best["correct"] + best["wrong"])
cols = ["source", "model", "context", "prompt", "compared", "accuracy", "f1_pub",
        "properties", "f1_std", "precision_std", "f1_ign", "recall_std", "abstention_rate", "error_rate"]
best_tbl = best[cols].rename(columns={"accuracy": "DOLAP acc (=recall)", "f1_pub": "DOLAP F1", "precision_std": "precision",
                                      "recall_std": "recall", "f1_std": "F1", "f1_ign": "F1 ign"})
best_tbl.round(3)
""")

md("""
### 4.2 Overall, over all 9 prompts

Choosing the best prompt after seeing the results favours models with more variance (and the best Sonnet configuration
rests on 3 answers). The fairer summary pools all prompts: our metrics micro-averaged over every scored H property,
with a 95% bootstrap interval over the 36 tasks (in the table only).
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
overall["F1 ign"] = 2 * overall.correct / (2 * overall.correct + overall.wrong)
overall[["source", "model", "context", "properties", "correct", "wrong", "abstained", "f1", "precision", "F1 ign",
         "recall", "F1 95% CI", "abstention_rate", "error_rate", "pub accuracy", "compared"]].round(3)
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
                color=MCOLOR.get(m, BLUE) if is_local else MUTED, alpha=1 if is_local else 0.7)
        last = s.recall.dropna()
        if len(last) and is_local:
            ax.text(3.08, last.iloc[-1], m, fontsize=8, va="center", color=INK2)
    ax.set_xticks(range(4), ["H easy\\nG easy", "H easy\\nG hard", "H hard\\nG easy", "H hard\\nG hard"])
    ax.set_xlim(-0.2, 3.9); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("With context" if ctx else "Without context", fontsize=11)
axes[0].set_ylabel("recall (share of hidden properties found)")
axes[0].plot([], [], color=BLUE, lw=2, marker="o", label="open-weight models (Ollama)")
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
    ax.bar(x + i * w - 0.4 + w / 2, lat.loc[m], width=w * 0.9, color=MCOLOR.get(m, MUTED), label=m)
ax.set_xticks(x, lat.columns, rotation=45, ha="right", fontsize=8); ax.grid(axis="x", visible=False)
ax.set_ylabel("seconds per run"); ax.legend(ncol=3, loc="upper left")
ax.set_title("Mean latency per run, by prompt and condition")
fig.tight_layout(); plt.show()
""")

discussion("cost")


md("""
## 5. Why are the DOLAP numbers higher?

Table 3 of the paper reports F1 0.54–0.80 for the proprietary models; the current experiments report F1 0.1–0.3 for
the open-weight models. Four things differ at once: **the scoring**, **the choice of configuration**, **the models**
and **the task set**. This section separates them on the same 36 tasks, then compares the task sets.

### 5.1 From the Table 3 number to our F1, one step at a time

For each model, the DOLAP F1 is recomputed while removing one difference at a time, ending at our F1:

1. **Table 3**: the best of the 18 configurations (9 prompts × with/without context), chosen by the DOLAP F1;
2. **all 18 configurations pooled** (no choice after seeing the results);
3. **without values copied from the input**: a property given in G and repeated in the answer is "correct" for free;
4. **asked properties only**: answers to properties that were not asked (the model adds `name`, `city`, `country`
   of the airport it recognised) are dropped;
5. **strict tolerance = F1 ign**: the tolerance of the current runs (text must overlap in both directions, a list
   answer to a single value is wrong, lone coordinates and units compared); F1 ign = 2P / (1 + P);
6. **abstentions counted = F1**: an asked property without an answer is a miss.
""")

code("""
H = tasks.set_index("task_id").H_names.to_dict()
G = tasks.set_index("task_id").G_drawn.map(json.loads).to_dict()
sp["kind"] = ["asked (H)" if p in H[t] else ("copied from the input (G)" if p in G[t] else "not asked")
              for t, p in zip(sp.task_id, sp.property)]

def f1_from_acc(acc):
    return 2 * acc / (1 + acc) if acc > 0 else 0.0

STEPS = ["1 Table 3: best of 18", "2 all 18 pooled", "3 without copied input", "4 asked properties only",
         "5 strict tolerance (F1 ign)", "6 abstentions counted (F1)"]

def ladder(model):
    a, b = sp[sp.model == model], ss[ss.model == model]
    cfg = a.groupby(["context", "prompt"]).correct.agg(["mean", "size"]).sort_values(["mean", "size"], ascending=False)
    c, w, ab = [(b.outcome == o).sum() for o in ("correct", "wrong", "abstained")]
    vals = [f1_from_acc(cfg["mean"].iloc[0]), f1_from_acc(a.correct.mean()),
            f1_from_acc(a[a.kind != "copied from the input (G)"].correct.mean()),
            f1_from_acc(a[a.kind == "asked (H)"].correct.mean()), 2 * c / (2 * c + w), 2 * c / (2 * c + 2 * w + ab)]
    best_ctx, best_prompt = cfg.index[0]
    return {**dict(zip(STEPS, vals)), "best configuration": f"{best_prompt}{' + ctx' if best_ctx else ''}",
            "answers behind it": int(cfg["size"].iloc[0])}

lad_models = [m for m in PUBLISHED_ORDER + OLLAMA_ORDER if m in set(sp.model)]
lad = pd.DataFrame({m: ladder(m) for m in lad_models}).T
lad[STEPS] = lad[STEPS].astype(float)
lad.round(3)
""")

code("""
fig, ax = plt.subplots(figsize=(11, 5.2))
GREYS = dict(zip(PUBLISHED_ORDER, ["#3d3d3a", "#6b6a66", "#2b2b29", "#8a8984", "#55544f", "#a3a29d"]))
for m in lad_models:
    ours = m not in PUBLISHED_ORDER
    y = lad.loc[m, STEPS].to_numpy(float)
    ax.plot(range(len(STEPS)), y, marker="o", ms=6 if ours else 5, lw=2.2 if ours else 1.4,
            color=MCOLOR.get(m, BLUE) if ours else GREYS[m], ls="-" if ours else "--", label=m)

ends = sorted(((lad.loc[m, STEPS[-1]], m) for m in lad_models))
pos = []
for v, m in ends:   # end labels at least 0.025 apart
    pos.append(max(v, pos[-1] + 0.025) if pos else v)
for (v, m), yy in zip(ends, pos):
    ax.text(len(STEPS) - 1 + 0.08, yy, m, fontsize=8, va="center", color=INK2)
ax.set_xticks(range(len(STEPS)), [s[2:].replace(" (", "\\n(") for s in STEPS], fontsize=8.5)
ax.set_xlim(-0.2, len(STEPS) + 0.9); ax.set_ylim(0, 0.85); ax.grid(axis="x", visible=False)
ax.set_ylabel("F1 (steps 1-5: DOLAP formula, step 6: ours)")
ax.legend(ncol=2, fontsize=8, loc="upper right", title="dashed = published proprietary models", title_fontsize=8)
ax.set_title("From Table 3 to our F1 on the same 36 tasks")
fig.tight_layout(); plt.show()
""")

md("""
### 5.2 Is precision really 1?

The paper states that precision is steadily equal to 1. That is a property of the formula, not of the models: the
evaluation calls sklearn with `y_true` all ones (every compared property "should be correct") and `y_pred` = 1 for a
correct answer, 0 for a wrong one. With no negative in `y_true` there can be no false positive, so precision is 1 as
soon as one answer is correct; a wrong answer only counts as a false negative, and recall equals accuracy.
""")

code("""
y_pred = [1, 1, 0, 0, 0]      # 2 correct, 3 wrong answers
y_true = [1] * len(y_pred)    # the DOLAP notebook: every compared property is a positive
tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))   # what sklearn counts
fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))   # always 0: y_true has no 0
fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))   # every wrong answer
P, R = tp / (tp + fp), tp / (tp + fn)
print(f"sklearn's formulas, y_true all ones: TP {tp}, FP {fp}, FN {fn} -> precision {P}, recall {R}, F1 {2*P*R/(P+R):.3f} = 2·0.4/(1+0.4)")
c, w = 2, 3
print("ours (a wrong answer is a false positive): precision", c / (c + w), "| F1 ign = 2c/(2c+w) =", round(2 * c / (2 * c + w), 3))
prec = our_score(["source", "model"]).set_index("model").reindex(lad_models)[["source", "precision", "F1 ign", "F1", "error", "abstention"]]
print("\\nPrecision when a wrong answer counts against the model (all 18 configurations, asked properties):")
prec.round(3)
""")

md("""
### 5.3 Same tasks, same scoring: how much is the model?

With the scoring and the configuration choice held fixed (step 6, all configurations pooled), the remaining
differences on the 36 tasks are the models. The open-weight models of the current lineup appear here once their runs
exist; until then gemma4:31b (1-run test) and Qwen3 4B stand in.
""")

code("""
same = our_score(["source", "model"]).set_index("model").reindex(lad_models)
fig, ax = plt.subplots(figsize=(10, 0.42 * len(same) + 1.4))
y = np.arange(len(same))
for i, (m, r) in enumerate(same.iterrows()):
    col = GREYS.get(m) if m in PUBLISHED_ORDER else MCOLOR.get(m, BLUE)
    ax.barh(i - 0.18, r["F1"], 0.34, color=col)
    ax.barh(i + 0.18, r["precision"], 0.34, color=col, alpha=0.45)
    ax.text(r["F1"] + 0.005, i - 0.18, f"F1 {r['F1']:.2f}", va="center", fontsize=8, color=INK2)
    ax.text(r["precision"] + 0.005, i + 0.18, f"P {r['precision']:.2f}  (F1 ign {r['F1 ign']:.2f})", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, same.index); ax.invert_yaxis(); ax.set_xlim(0, 0.9); ax.grid(axis="y", visible=False)
ax.set_xlabel("our scoring, all 18 configurations pooled (solid: F1, light: precision)")
ax.set_title("The 36 DOLAP tasks under our scoring")
fig.tight_layout(); plt.show()
""")

md("""
### 5.4 Different task sets: the 36 DOLAP tasks against airports v5

The current airports results come from another task set (v5, built by Algorithm 2 of the new draft), with other
prompts (Figure 2) and 3 runs. Its composition differs from the DOLAP set in ways that lower every model's score.
""")

code("""
V5 = {v: ROOT / f"results/airports/{v}/scored/scored_voted.csv" for v in ("v5_ctx", "v5_noctx")}
ID = {"name", "iata", "icao", "city"}
cent = pd.read_csv(ROOT / "tasks/airports/v5_ctx/intermediate/centrality.csv", low_memory=False)
cent = cent[(cent.element_kind == "node") & (cent.element_type == "Airport")].copy()
cent["rank"] = cent.raw_centrality.rank(ascending=False, method="min").astype(int)
rank = dict(zip(cent.icao, cent["rank"]))

def composition(name, t, h_names, g_names, scored):
    return {"task set": name, "tasks": len(t), "airports": t.node_id.nunique(),
            "airport rank (of 7,698), median": int(np.median([rank.get(n, np.nan) for n in t.node_id.unique()])),
            "worst airport rank": int(max(rank.get(n, 0) for n in t.node_id.unique())),
            "mean |H|": round(h_names.map(len).mean(), 2), "mean |G|": round(g_names.map(len).mean(), 2),
            "G names the airport (name/IATA/ICAO/city)": round(g_names.map(lambda g: bool(set(g) & ID)).mean(), 2),
            "share of scored H that is commercial_flights": round((scored.property == "commercial_flights").mean(), 2)}

comp = [composition("DOLAP 36", tasks, tasks.H_names, tasks.G_drawn.map(json.loads), ss)]
v5 = {}
for v, path in V5.items():
    t = pd.read_csv(ROOT / f"tasks/airports/{v}/tasks.csv")
    sc = pd.read_csv(path)
    v5[v] = sc[sc.strategy.eq("A")]
    comp.append(composition(f"airports {v}", t, t.H.map(lambda s: [x["property_name"] for x in json.loads(s)]),
                            t.G.map(lambda s: [x["property_name"] for x in json.loads(s)]), v5[v]))
pd.DataFrame(comp).set_index("task set").T
""")

code("""
rows = []
for v, sc in v5.items():
    ids = {r.task_id for r in pd.read_csv(ROOT / f"tasks/airports/{v}/tasks.csv").itertuples()
           if set(x["property_name"] for x in json.loads(r.G)) & ID}
    for label, sub in [("all", sc), ("without commercial_flights", sc[sc.property != "commercial_flights"]),
                       ("G names the airport", sc[sc.task_id.isin(ids)]), ("G does not", sc[~sc.task_id.isin(ids)])]:
        m = our_score(["model"], data=sub).assign(**{"task set": v, "subset": label})
        rows.append(m)
v5m = pd.concat(rows)
print("Airports v5, Strategy A, all 9 prompts pooled (our scoring):")
v5m.pivot_table(index=["task set", "subset"], columns="model", values="F1", sort=False).round(3)
""")

code("""
print("F1 ign (the DOLAP formula on asked properties) on v5, same subsets:")
v5m.pivot_table(index=["task set", "subset"], columns="model", values="F1 ign", sort=False).round(3)
""")

code("""
props = sorted(set(ss.property) | set(v5["v5_ctx"].property))
def recall_by_prop(df):
    return df.assign(c=df.outcome.eq("correct")).groupby("property").c.mean()
pp = pd.DataFrame({
    "DOLAP 36, proprietary": recall_by_prop(ss[ss.source == "published"]),
    "DOLAP 36, open-weight": recall_by_prop(ss[ss.source == "ollama"]),
    "v5 ctx, lineup": recall_by_prop(v5["v5_ctx"]),
    "v5 noctx, lineup": recall_by_prop(v5["v5_noctx"]),
    "share in DOLAP 36": ss[ss.model == ss.model.iloc[0]].property.value_counts(normalize=True),
    "share in v5 ctx": v5["v5_ctx"][v5["v5_ctx"].model == v5["v5_ctx"].model.iloc[0]].property.value_counts(normalize=True),
}).reindex(props)
print("Recall per hidden property (all configurations pooled), and how much of each task set it makes up:")
pp.round(2)
""")

discussion("gap")

md("## 6. Discussion")
discussion("main")
md("## 7. Limitations")
discussion("limits")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "agentaugmentation (venv)", "language": "python", "name": "agentaugmentation"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")

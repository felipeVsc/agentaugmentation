"""Builds notebooks/energy_effort_report.ipynb: energy consumption and effort (tokens, token amplification) of the
airports v5 runs (Strategy A, Strategy B, both) and of Strategy A on football (idmix), then of both datasets together.

    venv/bin/python scripts/energy_effort_extract.py          # per-run data from the runs.jsonl files
    venv/bin/python notebooks/build_energy_effort_notebook.py
    venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/energy_effort_report.ipynb

The discussion cells come from notebooks/energy_effort_report_discussion.json (written after looking at the results).
Figures are also saved to notebooks/figures/energy_effort/ (PNG and PDF)."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "energy_effort_report.ipynb"
DISC_PATH = ROOT / "notebooks" / "energy_effort_report_discussion.json"
DISC = json.loads(DISC_PATH.read_text()) if DISC_PATH.exists() else {}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def discussion(key, placeholder="*Discussion written after the run.*"):
    md(DISC.get(key, placeholder))


# --------------------------------------------------------------------------------------------- 0. intro
md(r"""
# Energy consumption and effort: airports and football

Energy and effort of the experiments run so far, in five parts:

1. **Airports, Strategy A** (LLM alone, 9 prompts of Figure 2), with and without context.
2. **Airports, Strategy B** (agent with web search; search only, and search + memory), context on.
3. **Airports, Strategy A vs Strategy B.**
4. **Football, Strategy A**, with and without context (Strategy B on football is still running and is not included).
5. **Both datasets**, Strategy A.

**Data.** Every ok run of the 5 models (gpt-oss:20b, gemma4:31b, gpt-oss:120b, mistral-large-3, deepseek-v4.1-flash),
3 runs per task:

| Dataset | Set | Tasks | Strategy |
|---|---|---|---|
| airports v5 | context on | 243 | A (9 prompts), B search, B search + memory |
| airports v5 | context off | 180 (tasks that have a context-free version) | A (9 prompts) |
| football v1 idmix | context on | 239 | A (9 prompts) |
| football v1 idmix | context off | 233 | A (9 prompts) |

**Energy** (as in `v5_eis.ipynb` and the paper draft):

- **LLM energy (EC)**: the EcoLogits estimate of each LLM call, summed over the run's calls (repair calls included),
  with its global warming potential (**GWP**, g CO₂e) and water consumption footprint (**WCF**, mL).
  deepseek-v4.1-flash has no EcoLogits entry; its estimate uses a stand-in of similar total size.
- **Run energy** `E_run = N_calls · E_single + N_searches · e_search`, with `e_search = 0.3 Wh` per web search (Google,
  2009; §3.4 varies it). Strategy A makes no search, so its run energy is its LLM energy. Searches are counted whether or
  not our search cache answered them (a user without the cache would make them). GWP and WCF of a search are those of the
  run's own LLM calls per Wh.
- All values are **estimates**, not measurements at Ollama.

**Effort** (METRICS.md §6), from the `effort` block of every run:

- **PFE**: tokens of the template (instructions and examples, without placeholders). Strategy B: the plan and synthesis
  templates together.
- **$T_{user}$ (human tokens)**: the tokens a person writes for one task = template + known properties (input) + context + target names.
- **KLM**: time to type and send the prompt (keystroke-level model): `M + N_chars·K + H + P + K`, with K = 0.28 s,
  M = 1.35 s, H = 0.4 s, P = 1.1 s, assuming the whole prompt is typed.
- **$T_{process}$**: the tokens the model processes in a run: input and output, reasoning included, over all LLM calls.
- **Token amplification** $= |T_{process}| \,/\, |T_{user}|$: how much work the system does for each token of human
  effort. For Strategy B it includes the retrieved evidence the synthesis call reads.
- **Output ratio** $= |T_{out}| \,/\, |T_{user}|$ (output tokens, reasoning included): a secondary measure, the part of
  $T_{process}$ the model generates.

Every F1 is also given with abstentions ignored, `F1 (ign.) = 2c / (2c + w)`, next to `F1 = 2c / (2c + 2w + a)`.
""")

# --------------------------------------------------------------------------------------------- setup
code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from evaluation.eis import agent_energy      # noqa: E402

E_SEARCH_WH = 0.3
K, M_, H, P = 0.28, 1.35, 0.4, 1.1          # KLM operator times (s)
FIG = ROOT / "notebooks/figures/energy_effort"
FIG.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd8", "#d9a514"
NEUTRAL = "#c9c8c2"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DCOLOR = {"airports": "#86b6ef", "football": "#1c5cab"}           # dataset: two steps of one hue
SCOLOR = {"on": "#1c5cab", "off": "#86b6ef"}                      # context on / off
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "legend.frameon": False, "figure.dpi": 110,
})

MODELS = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-mistral-large-3", "cloud-deepseek-v4.1-flash"]
MNAME = {"cloud-gpt-oss-20b": "gpt-oss:20b", "cloud-gemma4-31b": "gemma4:31b", "cloud-gpt-oss-120b": "gpt-oss:120b",
         "cloud-mistral-large-3": "mistral-large-3", "cloud-deepseek-v4.1-flash": "deepseek-v4.1-flash"}
MN = [MNAME[m] for m in MODELS]
MCOLOR = dict(zip(MODELS, [BLUE, AQUA, VIOLET, ORANGE, YELLOW]))
PROMPTS = ["compact", "zero", "zerocot", "few_1", "fewcot_1", "few_3", "fewcot_3", "few_5", "fewcot_5"]
PNAME = {"compact": "compact", "zero": "zero-shot", "few_1": "1-shot", "few_3": "3-shot", "few_5": "5-shot",
         "zerocot": "zero-shot + CoT", "fewcot_1": "1-shot + CoT", "fewcot_3": "3-shot + CoT", "fewcot_5": "5-shot + CoT",
         "search_evidence": "B: search", "search_memory": "B: search + memory"}
SETNAME = {"ctx": "on", "noctx": "off"}


PLAIN = FuncFormatter(lambda v, _: f"{v:g}")   # log axes labelled 0.1, 2, 5 rather than 10⁻¹, 2×10⁰
MINOR25 = FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}".rstrip("0").rstrip(".")[-1:] in ("2", "5") else "")


def logaxis(ax, which="y", minor=False):
    # log scale with plain labels (0.001, 0.01, 1, 2, 5), minor labels at 2 and 5 when the range is short
    for a in which:
        axis = ax.xaxis if a == "x" else ax.yaxis
        (ax.set_xscale if a == "x" else ax.set_yscale)("log")
        axis.set_major_formatter(PLAIN)
        axis.set_minor_formatter(MINOR25 if minor else FuncFormatter(lambda v, _: ""))


def fig1(w=9, h=5):
    # every saved figure is a single chart with its own title, axis labels and legend
    return plt.subplots(figsize=(w, h))


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.show()


def model_handles():
    return [plt.Line2D([], [], ls="", marker="o", color=MCOLOR[m], ms=7, label=MNAME[m]) for m in MODELS]


def heading(text):
    return plt.Line2D([], [], ls="", marker="", label=text)


def legend(ax, groups, **kw):
    # one legend with a heading per group, e.g. [("model", handles), ("approach", handles)]
    h = []
    for i, (title, hs) in enumerate(groups):
        h += ([heading("")] if i else []) + [heading(title)] + list(hs)
    lg = ax.legend(handles=h, fontsize=8, handletextpad=0.4, **kw)
    for t, hh in zip(lg.get_texts(), h):
        if hh.get_marker() in ("", "None", None) and t.get_text():
            t.set_fontweight("bold"); t.set_position((-14, 0))
    return lg


def model_legend(ax, **kw):
    return legend(ax, [("model", model_handles())], **kw)


runs = pd.read_csv(ROOT / "results/energy_effort/runs.csv")
acc = pd.read_csv(ROOT / "results/energy_effort/accuracy.csv")
runs["human_tokens"] = runs[[f"{p}_tokens" for p in ("template", "input", "context", "targets")]].sum(axis=1)
runs["human_chars"] = runs[[f"{p}_chars" for p in ("template", "input", "context", "targets")]].sum(axis=1)
runs["klm_s"] = M_ + runs.human_chars * K + H + P + K
runs["amp_out"] = runs.tokens_out / runs.human_tokens
runs["amp_proc"] = (runs.tokens_in + runs.tokens_out) / runs.human_tokens
runs["EC_search"] = runs.searches * E_SEARCH_WH
runs["EC_run"] = runs.EC_wh + runs.EC_search
scale = np.where(runs.EC_wh > 0, runs.EC_run / runs.EC_wh, 1.0)   # GWP and WCF of a search: those of the run's LLM calls per Wh
runs["GWP_run"], runs["WCF_run"] = runs.GWP_g * scale, runs.WCF_ml * scale
# check the run energy against the evaluation's formula (METRICS.md §4.2) on a sample
s = runs[runs.searches > 0].head(200)
chk = [1000 * agent_energy(e / 1000, n, t, E_SEARCH_WH / (e / n)) for e, n, t in zip(s.EC_wh, s.llm_calls, s.searches)]
assert np.allclose(chk, s.EC_run), "run energy differs from evaluation.eis.agent_energy"
print(runs.groupby(["dataset", "set", "strategy"]).size().rename("ok runs").to_frame().T.to_string())
""")

code(r"""
def configs(dataset, set_, strategy):
    # one row per model × variant: means per run, accuracy and energy per correct answer
    r = runs[(runs.dataset == dataset) & (runs.set == set_) & runs.strategy.isin(strategy if isinstance(strategy, list) else [strategy])]
    c = (r.groupby(["model", "variant"])
          .agg(runs=("task_id", "size"), tasks=("task_id", "nunique"), llm_calls=("llm_calls", "mean"), searches=("searches", "mean"),
               searches_real=("searches_real", "mean"), tokens_in=("tokens_in", "mean"), tokens_out=("tokens_out", "mean"),
               tokens_reasoning=("tokens_reasoning", "mean"), evidence_tokens=("evidence_tokens", "mean"),
               EC_llm=("EC_wh", "mean"), EC_search=("EC_search", "mean"), EC_run=("EC_run", "mean"),
               GWP=("GWP_run", "mean"), WCF=("WCF_run", "mean"), EC_total=("EC_run", "sum"), EC_llm_total=("EC_wh", "sum"),
               PFE=("template_tokens", "mean"), human_tokens=("human_tokens", "mean"), klm_s=("klm_s", "mean"),
               amp_out=("amp_out", "mean"), amp_proc=("amp_proc", "mean"), latency_s=("latency_s", "median"),
               cost_usd=("cost_usd", "sum"))
          .reset_index())
    a = acc[(acc.dataset == dataset) & (acc.set == set_)]
    c = c.merge(a[["model", "variant", "correct", "wrong", "abstained", "f1", "f1_ign", "abstention"]], on=["model", "variant"], how="left")
    c["Wh_per_correct"] = c.EC_total / c.correct.replace(0, np.nan)
    c["Wh_per_correct_llm"] = c.EC_llm_total / c.correct.replace(0, np.nan)
    c["search_share"] = c.EC_search / c.EC_run
    return c

def best_prompt(c):
    # each model's Strategy A prompt with the highest F1 (as in the other reports)
    a = c[c.variant.isin(PROMPTS)]
    return a.loc[a.groupby("model").f1.idxmax()].set_index("model").variant.to_dict()

def order(df, col="model"):
    return df.assign(_o=df[col].map({m: i for i, m in enumerate(MODELS)})).sort_values(["_o"] + ([ "variant"] if "variant" in df else [])).drop(columns="_o")

FMT = {"llm_calls": "{:.2f}", "searches": "{:.2f}", "searches_real": "{:.2f}", "tokens_in": "{:.0f}", "tokens_out": "{:.0f}",
       "tokens_reasoning": "{:.0f}", "evidence_tokens": "{:.0f}", "EC_llm": "{:.4f}", "EC_search": "{:.3f}", "EC_run": "{:.4f}",
       "GWP": "{:.4f}", "WCF": "{:.3f}", "PFE": "{:.0f}", "human_tokens": "{:.0f}", "klm_s": "{:.0f}", "amp_out": "{:.2f}",
       "amp_proc": "{:.2f}", "Wh_per_correct": "{:.3f}", "Wh_per_correct_llm": "{:.3f}", "search_share": "{:.0%}",
       "f1": "{:.3f}", "f1_ign": "{:.3f}", "abstention": "{:.0%}", "latency_s": "{:.1f}", "cost_usd": "${:.2f}"}
COLS = {"llm_calls": "LLM calls", "searches": "searches", "searches_real": "searches (not cached)", "tokens_in": "tokens in",
        "tokens_out": "tokens out", "tokens_reasoning": "of which reasoning", "evidence_tokens": "evidence tokens",
        "EC_llm": "EC LLM (Wh)", "EC_search": "EC searches (Wh)", "EC_run": "E_run (Wh)", "GWP": "GWP (g CO₂e)",
        "WCF": "WCF (mL)", "PFE": "PFE", "human_tokens": "|T_user| (human tokens)", "klm_s": "KLM (s)", "amp_out": "output ratio |T_out|/|T_user|",
        "amp_proc": "token ampl. |T_process|/|T_user|", "Wh_per_correct": "Wh per correct", "Wh_per_correct_llm": "Wh per correct (LLM only)",
        "search_share": "searches' share of E_run", "f1": "F1", "f1_ign": "F1 (ign.)", "abstention": "abstention",
        "latency_s": "median latency (s)", "cost_usd": "cost (USD)"}

def show(df, cols, index=("model",), caption=None):
    d = df.copy()
    if "model" in d:
        d["model"] = d.model.map(MNAME)
    if "variant" in d:
        d["variant"] = d.variant.map(lambda v: PNAME.get(v, v))
    d = d[list(index) + cols].rename(columns=COLS)
    st = d.style.hide(axis="index").format({COLS[k]: v for k, v in FMT.items() if k in cols})
    if caption:
        st = st.set_caption(caption)
    display(st)

C = {(d, s, "A"): configs(d, s, "A") for d in ("airports", "football") for s in ("ctx", "noctx")}
C[("airports", "ctx", "B")] = configs("airports", "ctx", ["B", "B_memory"])   # both B variants
""")

# --------------------------------------------------------------------------------------------- shared A section
A_SECTION = r"""
DS = "@DS@"
ca = {s: C[(DS, s, "A")] for s in ("ctx", "noctx")}
per_model = pd.concat({SETNAME[s]: c.groupby("model")[["llm_calls", "tokens_in", "tokens_out", "tokens_reasoning", "EC_run", "GWP", "WCF",
                                                       "amp_proc", "Wh_per_correct", "f1", "f1_ign"]].mean() for s, c in ca.items()}, names=["context"])
per_model = per_model.reset_index()
show(order(per_model), ["llm_calls", "tokens_in", "tokens_out", "tokens_reasoning", "EC_run", "GWP", "WCF", "amp_proc", "Wh_per_correct", "f1", "f1_ign"],
     index=("context", "model"), caption=f"{DS}, Strategy A: means over the 9 prompts, per model")
"""

A_PROMPTS = r"""
eff = ca["ctx"].groupby("variant")[["PFE", "human_tokens", "klm_s"]].mean().reindex(PROMPTS)
amp = ca["ctx"].pivot(index="variant", columns="model", values="amp_proc").reindex(PROMPTS)[MODELS].rename(columns=MNAME)
ec = ca["ctx"].pivot(index="variant", columns="model", values="EC_run").reindex(PROMPTS)[MODELS].rename(columns=MNAME)
f = ca["ctx"].groupby("variant")[["f1", "f1_ign"]].mean().reindex(PROMPTS).rename(columns={"f1": "mean F1", "f1_ign": "mean F1 (ign.)"})
t = eff.join(amp.add_prefix("ampl. ")).join(f).rename(index=PNAME).rename(columns={"PFE": "PFE", "human_tokens": "|T_user|", "klm_s": "KLM (s)"})
display(t.style.format("{:.2f}").format("{:.0f}", subset=["PFE", "|T_user|", "KLM (s)"]).format("{:.3f}", subset=["mean F1", "mean F1 (ign.)"])
         .set_caption(f"{DS}, context on: effort per prompt and token amplification |T_process|/|T_user| per model"))
display(ec.rename(index=PNAME).style.format("{:.4f}").background_gradient(cmap=SEQ, axis=None)
          .set_caption(f"{DS}, context on: energy per run (Wh), model × prompt"))
"""

A_FIGS = r"""
TITLE = DS.capitalize()
x = np.arange(len(MODELS)); w = 0.38
fig, ax = fig1()
for i, s in enumerate(["ctx", "noctx"]):
    v = ca[s].groupby("model").EC_run.mean().reindex(MODELS)
    ax.bar(x + (i - 0.5) * w, v, w - 0.04, color=SCOLOR[SETNAME[s]], label=f"context {SETNAME[s]} ({ca[s].tasks.max()} tasks)")
logaxis(ax); ax.set_xticks(x, MN, rotation=20); ax.set_ylabel("Wh per run (log), mean over the 9 prompts"); ax.legend(fontsize=8)
ax.set_title(f"{TITLE}, Strategy A: energy per run, context on vs off")
save(fig, f"{DS}_A_energy_per_run")

# context on only: mean over the 9 prompts (bar) and each prompt (dots)
c = ca["ctx"]
fig, ax = fig1()
v = c.groupby("model").EC_run.mean().reindex(MODELS)
ax.bar(x, v, 0.6, color=[MCOLOR[m] for m in MODELS], alpha=0.35)
ax.bar([0], [np.nan], color=NEUTRAL, label="mean over the 9 prompts (bar)")   # legend swatch in a neutral colour
for j, m in enumerate(MODELS):
    d = c[c.model == m]
    ax.scatter(np.full(len(d), j), d.EC_run, s=30, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1, zorder=3)
ax.scatter([], [], s=30, color=INK2, label="one prompt")
logaxis(ax); ax.set_xticks(x, [f"{MNAME[m]}\n{v[m]:.4f} Wh" for m in MODELS]); ax.set_ylabel("Wh per run (log)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"{TITLE}, Strategy A, context on ({c.tasks.max()} tasks): energy per run")
save(fig, f"{DS}_A_energy_per_run_ctx")

fig, ax = fig1()
for m in MODELS:
    d = ca["ctx"][ca["ctx"].model == m].set_index("variant").reindex(PROMPTS)
    ax.plot(range(len(PROMPTS)), d.amp_proc, marker="o", ms=6, lw=2, color=MCOLOR[m], label=MNAME[m])
logaxis(ax, minor=True); ax.set_xticks(range(len(PROMPTS)), [PNAME[p] for p in PROMPTS], rotation=35, ha="right")
ax.set_ylabel("token amplification $|T_{process}|\\,/\\,|T_{user}|$ (log)")
ax.set_title(f"{TITLE}, Strategy A, context on: token amplification per prompt")
ax.legend(fontsize=8, title="model", title_fontsize=8, loc="best")
save(fig, f"{DS}_A_amplification_per_prompt")

fig, ax = fig1()
e = ca["ctx"].groupby("variant")[["PFE", "human_tokens"]].mean().reindex(PROMPTS)
ax.bar(range(len(PROMPTS)), e.human_tokens, 0.6, color=NEUTRAL, label="task (input, context, target names)")
ax.bar(range(len(PROMPTS)), e.PFE, 0.6, color=BLUE, label="template = PFE")
ax.set_xticks(range(len(PROMPTS)), [PNAME[p] for p in PROMPTS], rotation=35, ha="right")
ax.set_ylabel("$|T_{user}|$: tokens a person writes per task (mean)")
ax.set_title(f"{TITLE}, Strategy A, context on: prompt effort $|T_{{user}}|$"); ax.legend(fontsize=8, loc="upper left")
save(fig, f"{DS}_A_prompt_effort")

for col, lab, fn in (("f1", "F1", "f1"), ("f1_ign", "F1 (abstentions ignored)", "f1ign")):
    fig, ax = fig1()
    for m in MODELS:
        d = ca["ctx"][ca["ctx"].model == m]
        ax.scatter(d.EC_run, d[col], s=42, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1, alpha=0.9, zorder=3)
        c = d[d.variant == "compact"]   # compact is the outlier prompt for several models: name it
        ax.annotate("compact", (c.EC_run.iloc[0], c[col].iloc[0]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    logaxis(ax, "x"); ax.set_xlabel("Wh per run (log)"); ax.set_ylabel(f"{lab}, voted over 3 runs")
    ax.set_title(f"{TITLE}, Strategy A, context on: {lab} vs energy\none point per model × prompt (9 prompts)", fontsize=11)
    model_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, f"{DS}_A_{fn}_vs_energy")
"""

A_CTX = r"""
pair = ca["ctx"].merge(ca["noctx"], on=["model", "variant"], suffixes=("_on", "_off"))
t = pd.DataFrame({"model": pair.model, "variant": pair.variant,
                  "tokens in, on": pair.tokens_in_on, "tokens in, off": pair.tokens_in_off,
                  "context tokens (human)": pair.human_tokens_on - pair.human_tokens_off,
                  "Wh/run on": pair.EC_run_on, "Wh/run off": pair.EC_run_off, "on ÷ off": pair.EC_run_on / pair.EC_run_off,
                  "ampl. on": pair.amp_proc_on, "ampl. off": pair.amp_proc_off})
t = t.groupby("model").mean(numeric_only=True).reindex(MODELS).rename(index=MNAME)
display(t.style.format("{:.4f}", subset=["Wh/run on", "Wh/run off"]).format("{:.0f}", subset=["tokens in, on", "tokens in, off", "context tokens (human)"])
         .format("{:.2f}", subset=["on ÷ off", "ampl. on", "ampl. off"]).set_caption(f"{DS}: context on vs off, means over the 9 prompts (all tasks of each set)"))
"""


def a_section(ds, num, title, key):
    md(f"""
## {num}. {title}

### {num}.1 Energy and tokens per model

Means over the 9 prompts. *Wh per correct* = energy of all 3 runs of every task ÷ properties the voted answers get right.
""")
    code(A_SECTION.replace("@DS@", ds))
    md(f"### {num}.2 Effort and energy per prompt")
    code(A_PROMPTS)
    code(A_FIGS)
    md(f"""
### {num}.3 Context on vs off

The context-off set has its own tasks (for airports, the 180 tasks that have a context-free version), so these are
means over each set, not paired tasks.
""")
    code(A_CTX)
    discussion(key)


# --------------------------------------------------------------------------------------------- 1. airports A
a_section("airports", 1, "Airports, Strategy A", "airports_a")

# --------------------------------------------------------------------------------------------- 2. airports B
md(r"""
## 2. Airports, Strategy B

Context on, 243 tasks, 3 runs each. *Search + memory* replays the plan and searches of *search* (no new search) and lets
the synthesis call use the model's own knowledge; its searches are those of the replayed run, counted again, since a
user without our cache would make them.

### 2.1 Energy, tokens and effort per model
""")

code(r"""
cb = C[("airports", "ctx", "B")]
show(order(cb), ["llm_calls", "searches", "searches_real", "tokens_in", "evidence_tokens", "tokens_out", "tokens_reasoning",
                 "EC_llm", "EC_search", "EC_run", "search_share", "GWP", "WCF"], index=("model", "variant"),
     caption="Airports, Strategy B: means per run")
show(order(cb), ["PFE", "human_tokens", "klm_s", "amp_out", "amp_proc", "Wh_per_correct", "Wh_per_correct_llm", "f1", "f1_ign", "abstention"],
     index=("model", "variant"), caption="Airports, Strategy B: effort, amplification, energy per correct answer and accuracy")
""")

code(r"""
d = cb[cb.variant == "search_evidence"].set_index("model").reindex(MODELS)
y = np.arange(len(MODELS))
fig, ax = fig1()
ax.barh(y, d.EC_llm, 0.6, color=BLUE, label="LLM calls (EcoLogits)")
ax.barh(y, d.EC_search, 0.6, left=d.EC_llm, color=NEUTRAL, label=f"web searches ({E_SEARCH_WH} Wh each)")
for i, (l, s_) in enumerate(zip(d.EC_llm, d.EC_search)):
    ax.text(l + s_ + 0.02, i, f"{l + s_:.2f} Wh ({s_ / (l + s_):.0%} searches)", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, MN); ax.invert_yaxis(); ax.set_xlabel("Wh per run (mean)"); ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, (d.EC_llm + d.EC_search).max() * 1.45)
ax.set_title("Airports, Strategy B (search), context on: energy per run by source")
save(fig, "airports_B_energy_by_source")

fig, ax = fig1(10, 5)
parts = pd.DataFrame({"template (plan + synthesis)": d.PFE, "task (input, targets)": d.human_tokens - d.PFE,
                      "retrieved evidence": d.evidence_tokens, "other model input": d.tokens_in - d.evidence_tokens - d.human_tokens,
                      "model output": d.tokens_out})
left = np.zeros(len(MODELS))
for col, colr in zip(parts.columns, [BLUE, "#86b6ef", NEUTRAL, "#e6e5e0", ORANGE]):
    ax.barh(y, parts[col].clip(lower=0), 0.6, left=left, color=colr, label=col); left += parts[col].clip(lower=0).to_numpy()
ax.set_yticks(y, MN); ax.invert_yaxis(); ax.set_xlabel("tokens per run, plan and synthesis calls together (mean)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5)
ax.set_title("Airports, Strategy B (search), context on: where the agent's tokens come from")
save(fig, "airports_B_token_sources")

fig, ax = fig1()
x = np.arange(len(MODELS)); w = 0.38
for i, (var, colr) in enumerate((("search_evidence", "#1c5cab"), ("search_memory", "#86b6ef"))):
    v = cb[cb.variant == var].set_index("model").amp_proc.reindex(MODELS)
    ax.bar(x + (i - 0.5) * w, v, w - 0.04, color=colr, label=PNAME[var])
    for xx, vv in zip(x + (i - 0.5) * w, v):
        ax.text(xx, vv + 0.08, f"{vv:.2f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, MN, rotation=20); ax.set_ylabel("token amplification $|T_{process}|\\,/\\,|T_{user}|$")
ax.set_ylim(0, cb.amp_proc.max() * 1.15); ax.legend(fontsize=8, title="Strategy B variant", title_fontsize=8, loc="upper left")
ax.set_title("Airports, Strategy B, context on: token amplification per model")
save(fig, "airports_B_token_amplification")
""")
md(r"""
*Other model input* is what the two calls read beyond the person's text and the evidence: the plan's output passed
to the synthesis call, the search results' titles and URLs, and the template counted twice (plan and synthesis each
include their template). It is clipped at 0 where the effort count is larger.
""")
discussion("airports_b")

# --------------------------------------------------------------------------------------------- 3. airports A vs B
md(r"""
## 3. Airports, Strategy A vs Strategy B

Context on, the same 243 tasks. For each model: Strategy A's zero-shot prompt and its best prompt (highest F1), and
Strategy B's two variants.
""")

code(r"""
ca_ = C[("airports", "ctx", "A")]
BEST = best_prompt(ca_)
rows = []
for m in MODELS:
    for lab, d in (("A: zero-shot", ca_[(ca_.model == m) & (ca_.variant == "zero")]),
                   (f"A: best ({PNAME[BEST[m]]})", ca_[(ca_.model == m) & (ca_.variant == BEST[m])]),
                   ("B: search", cb[(cb.model == m) & (cb.variant == "search_evidence")]),
                   ("B: search + memory", cb[(cb.model == m) & (cb.variant == "search_memory")])):
        rows.append(d.assign(approach=lab))
ab = pd.concat(rows)
show(ab.assign(variant=ab.approach), ["llm_calls", "searches", "tokens_in", "tokens_out", "EC_llm", "EC_run", "amp_out", "amp_proc",
                                      "Wh_per_correct", "f1", "f1_ign"], index=("model", "variant"),
     caption="Airports, context on: Strategy A (zero-shot, best prompt) and Strategy B")
""")

code(r"""
best = ca_[ca_.apply(lambda r: r.variant == BEST[r.model], axis=1)].set_index("model").reindex(MODELS)
bs = cb[cb.variant == "search_evidence"].set_index("model").reindex(MODELS)
bm = cb[cb.variant == "search_memory"].set_index("model").reindex(MODELS)
ratio = pd.DataFrame({
    "E_run: B ÷ A best": bs.EC_run / best.EC_run,
    "LLM energy only: B ÷ A best": bs.EC_llm / best.EC_llm,
    "tokens processed: B ÷ A best": (bs.tokens_in + bs.tokens_out) / (best.tokens_in + best.tokens_out),
    "tokens out: B ÷ A best": bs.tokens_out / best.tokens_out,
    "|T_user|: B ÷ A best": bs.human_tokens / best.human_tokens,
    "Wh per correct: B ÷ A best": bs.Wh_per_correct / best.Wh_per_correct,
    "Wh per correct: B + memory ÷ A best": bm.Wh_per_correct / best.Wh_per_correct,
    "ΔF1 (B − A best)": bs.f1 - best.f1, "ΔF1 ign. (B − A best)": bs.f1_ign - best.f1_ign,
}).rename(index=MNAME)
display(ratio.style.set_caption("Airports, context on: Strategy B (search) relative to each model's best Strategy A prompt").format("{:.1f}×").format("{:+.3f}", subset=["ΔF1 (B − A best)", "ΔF1 ign. (B − A best)"]))
""")

code(r"""
APPS = ["A: zero-shot", "A: best", "B: search", "B: search + memory"]
AMARK = dict(zip(APPS, ["o", "s", "D", "^"]))
ab["app"] = ab.approach.str.replace(r" \(.*\)", "", regex=True)
def approach_legend(ax, extra=(), models=True, **kw):
    # colour = model (left out where the axis already names the models), marker = approach
    h = [plt.Line2D([], [], ls="", marker=AMARK[a], color=INK2, ms=7, label=a) for a in APPS] + list(extra)
    legend(ax, ([("model", model_handles())] if models else []) + [("approach", h)], **kw)

fig, ax = fig1(10, 5.5)
for j, m in enumerate(MODELS):
    d = ab[ab.model == m]
    for k, a in enumerate(APPS):
        r = d[d.app == a].iloc[0]
        yy = j + (k - 1.5) * 0.17
        ax.scatter(r.EC_run, yy, marker=AMARK[a], s=60, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if a.startswith("B"):   # the same run without its searches: hollow marker of the same shape
            ax.scatter(r.EC_llm, yy, marker=AMARK[a], s=50, facecolor="none", edgecolor=MCOLOR[m], linewidth=1.2, zorder=3)
            ax.plot([r.EC_llm, r.EC_run], [yy, yy], color=MCOLOR[m], lw=0.8, alpha=0.5, zorder=2)
logaxis(ax, "x"); ax.set_yticks(range(len(MODELS)), MN); ax.invert_yaxis(); ax.set_xlabel("Wh per run (log)")
ax.grid(axis="y", visible=False)
hollow = plt.Line2D([], [], ls="", marker="D", markerfacecolor="none", markeredgecolor=INK2, ms=7, label="B, LLM calls only (hollow)")
approach_legend(ax, [hollow], models=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6)   # colour = the model on the y axis
ax.set_title("Airports, context on: energy per run, Strategy A vs Strategy B")
save(fig, "airports_AB_energy_per_run")

fig, ax = fig1(9, 5.5)
for m in MODELS:
    d = ab[ab.model == m]
    for a in APPS:
        r = d[d.app == a].iloc[0]
        ax.scatter(r.amp_proc, r.amp_out, marker=AMARK[a], s=60, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
logaxis(ax, "xy", minor=True)
ax.set_xlabel("token amplification $|T_{process}|\\,/\\,|T_{user}|$ (log)")
ax.set_ylabel("output ratio $|T_{out}|\\,/\\,|T_{user}|$ (log)")
approach_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Airports, context on: token amplification and output ratio, A vs B")
save(fig, "airports_AB_amplification")

# accuracy against token amplification: the 9 A prompts (dots) and the two B variants, per model
for col, lab, fn in (("f1", "F1", "f1"), ("f1_ign", "F1 (abstentions ignored)", "f1ign")):
    fig, ax = fig1(9, 5.5)
    for m in MODELS:
        d = ca_[ca_.model == m]
        ax.scatter(d.amp_proc, d[col], s=26, color=MCOLOR[m], alpha=0.75, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for var, mk in (("search_evidence", "D"), ("search_memory", "^")):
            r = cb[(cb.model == m) & (cb.variant == var)].iloc[0]
            ax.scatter(r.amp_proc, r[col], marker=mk, s=70, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_xlim(1, None); ax.set_ylim(0, None)
    ax.set_xlabel("token amplification $|T_{process}|\\,/\\,|T_{user}|$"); ax.set_ylabel(f"{lab}, voted over 3 runs")
    legend(ax, [("model", model_handles()),
                ("approach", [plt.Line2D([], [], ls="", marker="o", color=INK2, ms=5, label="A: one prompt"),
                              plt.Line2D([], [], ls="", marker="D", color=INK2, ms=7, label="B: search"),
                              plt.Line2D([], [], ls="", marker="^", color=INK2, ms=7, label="B: search + memory")])],
           loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Airports, context on: {lab} against token amplification, A vs B")
    save(fig, f"airports_AB_{fn}_vs_amplification")

for col, lab, fn in (("f1", "F1", "f1"), ("f1_ign", "F1 (abstentions ignored)", "f1ign")):
    fig, ax = fig1(9, 5.5)
    for m in MODELS:
        d = ca_[ca_.model == m]
        ax.scatter(d.EC_run, d[col], s=18, color=MCOLOR[m], alpha=0.4, edgecolor="none")
        for a in APPS:
            r = ab[(ab.model == m) & (ab.app == a)].iloc[0]
            ax.scatter(r.EC_run, r[col], marker=AMARK[a], s=70, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    logaxis(ax, "x"); ax.set_xlabel("Wh per run (log)"); ax.set_ylabel(f"{lab}, voted over 3 runs")
    small = plt.Line2D([], [], ls="", marker="o", color=INK2, alpha=0.4, ms=4, label="A: other prompts")
    approach_legend(ax, [small], loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Airports, context on: {lab} vs energy per run, A vs B", fontsize=11)
    save(fig, f"airports_AB_{fn}_vs_energy")
""")

md(r"""
### 3.1 Sensitivity to the energy of a search

The energy of a web search is not reported by the provider; the ratio B ÷ A best per run as a function of it
(0 = LLM calls only).
""")

code(r"""
es = np.logspace(-3, 0, 60)
fig, ax = fig1(9, 5)
for m in MODELS:
    ax.plot(es, (bs.loc[m].EC_llm + bs.loc[m].searches * es) / best.loc[m].EC_run, color=MCOLOR[m], lw=2, label=MNAME[m])
ax.axvline(E_SEARCH_WH, color=INK2, ls="--", lw=1, label=f"value used in this report ({E_SEARCH_WH} Wh)")
logaxis(ax, "xy")
ax.set_xlabel("energy of one web search (Wh, log)"); ax.set_ylabel("E_run of B (search) ÷ E_run of A best prompt (log)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Airports, context on: B ÷ A energy per run vs the energy of a search")
save(fig, "airports_AB_search_sensitivity")
""")
discussion("airports_ab")

# --------------------------------------------------------------------------------------------- 4. football A
a_section("football", 4, "Football, Strategy A", "football_a")

# --------------------------------------------------------------------------------------------- 5. both datasets
md(r"""
## 5. Both datasets, Strategy A

Strategy A on airports (v5) and football (v1 idmix), context on unless said otherwise. The prompts and models are the
same, and both task sets give 1–5 known properties (3 on average) and ask for one. They differ in where those properties
sit: on airports most are context (neighbouring nodes, ~37 tokens per task), on football most are the player's own
properties (~29 tokens) with little context (~4 tokens).
""")

code(r"""
both = pd.concat([C[(d, s, "A")].assign(dataset=d, context=SETNAME[s]) for d in ("airports", "football") for s in ("ctx", "noctx")])
pm = (both.groupby(["context", "dataset", "model"])[["tokens_in", "tokens_out", "tokens_reasoning", "human_tokens", "EC_run", "GWP", "WCF",
                                                     "amp_out", "amp_proc", "Wh_per_correct", "f1", "f1_ign"]].mean().reset_index())
show(order(pm[pm.context == "on"]), ["tokens_in", "tokens_out", "tokens_reasoning", "human_tokens", "EC_run", "GWP", "WCF", "amp_out",
                                     "amp_proc", "Wh_per_correct", "f1", "f1_ign"], index=("dataset", "model"),
     caption="Strategy A, context on: means over the 9 prompts")
tot = (both.groupby(["dataset", "context"]).agg(runs=("runs", "sum"), EC_total_wh=("EC_total", "sum"), cost_usd=("cost_usd", "sum"),
                                                correct=("correct", "sum")).reset_index())
tot["Wh per correct (all configs)"] = tot.EC_total_wh / tot.correct
display(tot.rename(columns={"EC_total_wh": "total energy (Wh)", "cost_usd": "cost (USD, list prices)", "correct": "correct answers (voted, summed)"})
          .style.hide(axis="index").format({"total energy (Wh)": "{:.1f}", "cost (USD, list prices)": "${:.2f}", "Wh per correct (all configs)": "{:.3f}",
                                             "runs": "{:,.0f}", "correct answers (voted, summed)": "{:,.0f}"}))
""")

code(r"""
on = pm[pm.context == "on"]
x = np.arange(len(MODELS)); w = 0.38
for col, ylab, title, fn in (
        ("EC_run", "Wh per run (log)", "energy per run", "both_A_energy_per_run"),
        ("amp_proc", "token amplification $|T_{process}|\\,/\\,|T_{user}|$ (log)", "token amplification", "both_A_amplification"),
        ("Wh_per_correct", "Wh per correct answer (log)\nall 3 runs of every task ÷ correct voted answers", "energy per correct answer",
         "both_A_wh_per_correct")):
    fig, ax = fig1()
    for i, d in enumerate(["airports", "football"]):
        v = on[on.dataset == d].set_index("model")[col].reindex(MODELS)
        ax.bar(x + (i - 0.5) * w, v, w - 0.04, color=DCOLOR[d], label=d)
    ax.set_xticks(x, MN, rotation=20); logaxis(ax); ax.set_ylabel(ylab); ax.legend(fontsize=8, title="dataset", title_fontsize=8)
    ax.set_title(f"Strategy A, context on: {title} per model, airports vs football\n(mean over the 9 prompts)", fontsize=11)
    save(fig, fn)

# token amplification on a linear scale with the values, as airports_B_token_amplification
fig, ax = fig1()
for i, d in enumerate(["airports", "football"]):
    v = on[on.dataset == d].set_index("model").amp_proc.reindex(MODELS)
    ax.bar(x + (i - 0.5) * w, v, w - 0.04, color=DCOLOR[d], label=d)
    for xx, vv in zip(x + (i - 0.5) * w, v):
        ax.text(xx, vv + 0.03, f"{vv:.2f}", ha="center", fontsize=8, color=INK2)
ax.axhline(1, color=INK2, lw=1, ls="--", zorder=0, label="1× = only what the user wrote")
ax.set_xticks(x, MN, rotation=20); ax.set_ylabel("token amplification $|T_{process}|\\,/\\,|T_{user}|$")
ax.set_ylim(0, on.amp_proc.max() * 1.15); ax.legend(fontsize=8, title="dataset", title_fontsize=8, loc="upper right")
ax.set_title("Strategy A, context on: token amplification per model, airports vs football\n(mean over the 9 prompts)", fontsize=11)
save(fig, "both_A_token_amplification")

# energy per run by prompt, one panel per model (own x scale), both datasets
fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
y = np.arange(len(PROMPTS)); h = 0.4
for ax, m in zip(axs.flat, MODELS):
    for i, d in enumerate(["airports", "football"]):
        v = both[(both.dataset == d) & (both.context == "on") & (both.model == m)].set_index("variant").EC_run.reindex(PROMPTS)
        ax.barh(y + (i - 0.5) * h, v, h - 0.04, color=DCOLOR[d], label=d)
    ax.set_yticks(y, [PNAME[p] for p in PROMPTS]); ax.invert_yaxis()
    ax.set_xlabel("Wh per run"); ax.set_title(MNAME[m], fontsize=11, loc="center")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False); ax.xaxis.set_major_locator(plt.MaxNLocator(5))
for ax in axs[:, 0]:
    ax.set_ylabel("prompt technique")
axs.flat[-1].axis("off")
axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(), title="dataset", loc="center", fontsize=10, title_fontsize=10)
fig.suptitle("Strategy A, context on: energy per run (Wh) by prompt and model, airports vs football\n"
             "each panel has its own x scale", fontsize=12, fontweight="bold")
save(fig, "both_A_energy_by_prompt_model")

for col, lab, fn in (("f1", "F1", "f1"), ("f1_ign", "F1 (abstentions ignored)", "f1ign")):
    fig, ax = fig1(9, 5.5)
    for d, mk in (("airports", "o"), ("football", "s")):
        t = both[(both.dataset == d) & (both.context == "on")]
        for m in MODELS:
            tt = t[t.model == m]
            ax.scatter(tt.EC_run, tt[col], marker=mk, s=40, color=MCOLOR[m], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       alpha=0.9 if d == "football" else 0.55)
    logaxis(ax, "x"); ax.set_xlabel("Wh per run (log)"); ax.set_ylabel(f"{lab}, voted over 3 runs")
    legend(ax, [("model", model_handles()),
                ("dataset", [plt.Line2D([], [], ls="", marker="o", color=INK2, alpha=0.55, ms=7, label="airports"),
                             plt.Line2D([], [], ls="", marker="s", color=INK2, ms=7, label="football")])],
           loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Strategy A, context on: {lab} vs energy, both datasets\none point per model × prompt", fontsize=11)
    save(fig, f"both_A_{fn}_vs_energy")
""")

code(r"""
# amplification per prompt, both datasets (mean over models), and the share of reasoning in the output
ap = both[both.context == "on"].groupby(["dataset", "variant"]).agg(amp=("amp_proc", "mean"), human=("human_tokens", "mean"),
                                                                     out=("tokens_out", "mean"), reas=("tokens_reasoning", "mean")).reset_index()
x = np.arange(len(PROMPTS)); w = 0.38
for col, ylab, title, fn in (("amp", "token amplification $|T_{process}|\\,/\\,|T_{user}|$\n(mean over the 5 models)",
                              "token amplification per prompt", "both_A_amplification_per_prompt"),
                             ("human", "$|T_{user}|$: tokens a person writes per task", "prompt effort $|T_{user}|$ per prompt",
                              "both_A_human_tokens_per_prompt")):
    fig, ax = fig1()
    for i, d in enumerate(["airports", "football"]):
        v = ap[ap.dataset == d].set_index("variant").reindex(PROMPTS)
        ax.bar(x + (i - 0.5) * w, v[col], w - 0.04, color=DCOLOR[d], label=d)
    ax.set_xticks(x, [PNAME[p] for p in PROMPTS], rotation=35, ha="right"); ax.set_ylabel(ylab)
    ax.set_title(f"Strategy A, context on: {title}, airports vs football", fontsize=11)
    ax.legend(fontsize=8, title="dataset", title_fontsize=8, loc="upper right" if col == "amp" else "upper left")
    save(fig, fn)
reas = both[both.context == "on"].groupby(["dataset", "model"]).apply(lambda g: g.tokens_reasoning.mean() / g.tokens_out.mean()).unstack(0).reindex(MODELS)
display(reas.rename(index=MNAME).style.format("{:.0%}").set_caption("Share of reasoning tokens in the output, context on"))
""")
discussion("both")

md("## 6. Summary")
discussion("summary")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")

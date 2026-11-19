"""Builds the paper reports, with as many charts as the data supports and a short paragraph under each one:

- football_a:  notebooks/paper_report_football_strategy_a.ipynb   Strategy A on football (context on and off);
- football_b:  notebooks/paper_report_football_strategy_b.ipynb   Strategy B on football (search, search + memory);
- football_ab: notebooks/paper_report_football_a_vs_b.ipynb       Strategy A against Strategy B on football;
- both:        notebooks/paper_report_both_datasets.ipynb         both strategies on airports and football.

    venv/bin/python notebooks/build_paper_reports_notebook.py football_a      # or football_b, football_ab, both
    venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/paper_report_football_strategy_a.ipynb
    venv/bin/jupyter nbconvert --to html notebooks/paper_report_football_strategy_a.ipynb

Data, scoring and chart helpers: notebooks/paper_report_lib.py. Figures: notebooks/figures/paper_reports/<mode>/ (PNG
and PDF). The paragraphs come from notebooks/paper_report_<mode>_discussion.json (written after looking at the
results). Every F1 comes with the F1 ignoring abstentions."""
import json
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

MODE = sys.argv[1] if len(sys.argv) > 1 else "football_a"
OUTNAME = {"football_a": "paper_report_football_strategy_a", "football_b": "paper_report_football_strategy_b",
           "football_ab": "paper_report_football_a_vs_b", "both": "paper_report_both_datasets"}[MODE]
ROOT = Path(__file__).resolve().parents[1]
disc_path = ROOT / "notebooks" / f"paper_report_{MODE}_discussion.json"
DISC = json.loads(disc_path.read_text()) if disc_path.exists() else {}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip("\n")))


def discussion(key):
    md(DISC.get(key, "*Discussion written after the run.*"))


def chart(key, title, src):
    md(f"### {title}")
    code(src)
    discussion(key)


SETUP = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() if Path.cwd().name == "notebooks" else Path.cwd() / "notebooks"))
from paper_report_lib import *
FOLDER = "{MODE}"
"""

SCORING = """
**Scoring.** Each task hides one property. The answer of a configuration (model, prompt or approach) is the vote over
its 3 runs (text: most frequent value; numbers: mean; abstained only when all 3 runs abstain), and is **correct**,
**wrong** or **abstained**. **F1** = 2c / (2c + 2w + a) counts abstentions against recall; **F1 (abstentions ignored)**
= 2c / (2c + w) scores only the answers given (the paper's Table 3 scoring). Intervals are 95% bootstrap intervals
over tasks (1,000 resamples); paired differences resample the same tasks for both sides; task-by-task comparisons use
an exact McNemar test on "correct".
"""

# ============================================================================================ football_a
if MODE == "football_a":
    md("""
    # Strategy A on the football graph (context on and off)

    **Strategy A** asks the model directly for the hidden property of a `Player` node, with the 9 prompts of
    Figure 2 (compact, zero-shot, 1/3/5-shot, each with and without chain of thought). 5 models, 3 runs each.
    Two task sets: **context on** (`football-v1-ctx-idmix`, 239 tasks: the given properties G may include
    properties of neighbouring nodes) and **context off** (`football-v1-noctx-idmix`, 233 tasks: G only from the
    player). In the 81 high-G context-on tasks, the player's name and/or team were added to G with probability 0.38
    (*identifier mix*), so the effect of the identifiers can be measured.
    """)
    md(SCORING)
    code(SETUP + """
v = load_voted(("football", "ctx"), ("football", "noctx"))
A = v[v.strategy == "A"].copy()
runs = load_runs(("football", "ctx"), ("football", "noctx"))
runsA = runs[runs.strategy == "A"]
eff = load_effort()
effA = eff[(eff.dataset == "football") & (eff.strategy == "A")]
print(A.groupby("set").task_id.nunique().to_dict(), "tasks;", len(A), "voted rows;", len(runsA), "scored runs")
""")
    md("## 1. Overall results")
    chart("overall_table", "Scores per model and set, all 9 prompts pooled, and each model's best prompt", """
pool = boot(A, ["set", "mname"])
display(pool[["set", "mname", "tasks", "correct", "wrong", "abstained", "precision", "recall", "f1", "f1_lo", "f1_hi",
              "f1_ign", "f1_ign_lo", "f1_ign_hi", "abstention"]])
mp = metrics(A, ["set", "mname", "approach"])
best = pd.concat([mp.loc[mp.groupby(["set", "mname"])[s].idxmax(), ["set", "mname", "approach", s]].rename(
    columns={"approach": f"best prompt ({n})", s: n}).set_index(["set", "mname"]) for s, n in SCORES], axis=1)
display(best)
""")
    chart("pooled_bars", "F1 per model, context on vs off (9 prompts pooled)", """
t = boot(A, ["set", "model"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (s, n) in zip(axes, SCORES):
    grouped_bars(ax, t, "model", MODELS, "set", ["ctx", "noctx"], s, SCOLOR, names={**MNAME, "ctx": "context on", "noctx": "context off"})
    ax.set_title(n); ax.set_xticklabels(MN, rotation=15)
axes[0].set_ylabel("score (9 prompts pooled)"); axes[0].legend(loc="upper left")
fig.suptitle("Football, Strategy A: score per model, context on vs off")
save(fig, FOLDER, "a_pooled_by_model")
""")
    for s_, sn in (("ctx", "context on"), ("noctx", "context off")):
        chart(f"heat_{s_}", f"F1 per model and prompt, {sn}", f"""
t = metrics(A[A.set == "{s_}"], ["model", "approach"])
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    pv = t.pivot(index="model", columns="approach", values=s).reindex(index=MODELS, columns=PROMPTS)
    heat(ax, pv, fmt="{{:.0%}}" if s == "abstention" else "{{:.2f}}", xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN)
    ax.set_title(n)
fig.suptitle("Football, Strategy A, {sn}: F1, F1 (abstentions ignored) and abstention per model and prompt")
save(fig, FOLDER, "a_heat_{s_}")
""")
    chart("outcomes", "Outcomes per model and prompt, context on", """
t = metrics(A[A.set == "ctx"], ["model", "approach"])
fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=True)
for ax, m in zip(axes, MODELS):
    stacked_outcomes(ax, t[t.model == m], "approach", PROMPTS, names=PNAME)
    ax.set_title(MNAME[m])
axes[-1].legend(loc="lower right", bbox_to_anchor=(1, -0.32), ncol=3)
fig.suptitle("Football, Strategy A, context on: share of correct, wrong and abstained answers")
save(fig, FOLDER, "a_outcomes_ctx")
""")
    chart("pr_scatter", "Precision against recall, every model and prompt", """
t = metrics(A, ["set", "model", "approach"])
fig, ax = plt.subplots(figsize=(8.5, 6))
rr = np.linspace(0.01, 1, 200)
for f in (0.1, 0.2, 0.3, 0.4):
    p_ = f * rr / (2 * rr - f); ok = (p_ > 0) & (p_ <= 1)
    ax.plot(rr[ok], p_[ok], color=GRID, lw=1, zorder=1); ax.text(0.445, f * 0.445 / (0.89 - f) + 0.01, f"F1 = {f}", fontsize=7, color=MUTED, ha="right")
for m in MODELS:
    for s_, fc in (("ctx", MCOLOR[m]), ("noctx", "none")):
        d = t[(t.model == m) & (t.set == s_)]
        ax.scatter(d.recall, d.precision, s=40, facecolor=fc, edgecolor=MCOLOR[m], linewidth=1.3, zorder=3)
ax.set_xlabel("recall (correct / all tasks)"); ax.set_ylabel("precision (correct / answered)"); ax.set_xlim(0, 0.45); ax.set_ylim(0, 1.02)
h = model_handles() + [plt.Line2D([], [], ls="", marker="o", color=INK2, label="context on"),
                       plt.Line2D([], [], ls="", marker="o", mfc="none", color=INK2, label="context off")]
ax.legend(handles=h, loc="upper right", fontsize=8)
ax.set_title("Football, Strategy A: precision vs recall (one dot per model and prompt)")
save(fig, FOLDER, "a_precision_recall")
""")
    md("## 2. Prompts")
    chart("shots", "Effect of the number of examples and of chain of thought (context on)", """
t = metrics(A[(A.set == "ctx") & (A.approach != "compact")], ["model", "approach"])
t["shots"] = t.approach.map(SHOTS); t["cot"] = t.approach.str.contains("cot")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
for ax, (s, n) in zip(axes, SCORES):
    for m in MODELS:
        for cot, ls in ((False, "-"), (True, "--")):
            d = t[(t.model == m) & (t.cot == cot)].sort_values("shots")
            ax.plot(d.shots, d[s], ls=ls, marker="o", ms=4, color=MCOLOR[m], lw=1.6)
    ax.set_xticks([0, 1, 3, 5]); ax.set_xlabel("examples in the prompt"); ax.set_title(n)
axes[0].set_ylabel("score")
axes[1].legend(handles=model_handles() + [plt.Line2D([], [], color=INK2, label="without CoT"),
                                          plt.Line2D([], [], color=INK2, ls="--", label="with CoT")], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle("Football, Strategy A, context on: score vs number of examples, with and without CoT")
save(fig, FOLDER, "a_shots_cot")
""")
    chart("prompt_rank", "Prompt ranking across models", """
t = metrics(A, ["set", "model", "approach"])
t["rank"] = t.groupby(["set", "model"]).f1.rank(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, s_ in zip(axes, ["ctx", "noctx"]):
    d = t[t.set == s_]
    for j, p in enumerate(PROMPTS):
        dd = d[d.approach == p]
        ax.scatter(np.full(len(dd), j) + np.linspace(-0.2, 0.2, len(dd)), dd.f1, c=[MCOLOR[m] for m in dd.model], s=36, zorder=3, edgecolor=SURFACE)
        ax.hlines(dd.f1.mean(), j - 0.3, j + 0.3, color=INK, lw=2)
    ax.set_xticks(range(len(PROMPTS))); ax.set_xticklabels([PNAME[p] for p in PROMPTS], rotation=35, ha="right")
    ax.set_title("context " + ("on" if s_ == "ctx" else "off")); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("F1 (bar = mean over the 5 models)"); axes[1].legend(handles=model_handles(), fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle("Football, Strategy A: F1 of each prompt, per model")
save(fig, FOLDER, "a_prompt_rank")
display(t.pivot_table(index="approach", columns="set", values="rank").reindex(PROMPTS).rename(columns={"ctx": "mean rank, on", "noctx": "mean rank, off"}))
""")
    md("## 3. Context on vs off")
    chart("ctx_paired", "Paired effect of context (task pairs whose context differs, voted answers)", """
rows = []
pairs = context_pairs("football")   # context-off task -> context-on task, only pairs whose G differs
off = A[A.set == "noctx"].assign(task_id=lambda x: x.task_id.map(pairs)).dropna(subset=["task_id"])
on = A[(A.set == "ctx") & A.task_id.isin(pairs.values())]
for m in MODELS:
    for p in PROMPTS:
        x = off[(off.model == m) & (off.approach == p)].set_index("task_id").outcome
        y = on[(on.model == m) & (on.approach == p)].set_index("task_id").outcome
        for s, n in SCORES:
            d, lo, hi = paired_diff(x, y, s)
            rows.append({"model": m, "approach": p, "score": s, "diff": d, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0})
pdf = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, (s, n) in zip(axes, SCORES):
    pv = pdf[pdf.score == s].pivot(index="model", columns="approach", values="diff").reindex(index=MODELS, columns=PROMPTS)
    heat(ax, pv, fmt="{:+.2f}", cmap=DIV, center=0, xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN)
    ax.set_title(f"{n}: context on − off")
fig.suptitle(f"Football, Strategy A: effect of context on the {len(pairs)} task pairs whose context differs (blue = context helps)")
save(fig, FOLDER, "a_context_effect")
print("model × prompt cells with a 95% interval excluding 0:", pdf.groupby("score").significant.sum().to_dict(), "of", len(MODELS) * len(PROMPTS))
pooled = []
for m in MODELS + [None]:
    x = off[(off.model == m) if m else off.model.notna()].set_index(["task_id", "model", "approach"]).outcome
    y = on[(on.model == m) if m else on.model.notna()].set_index(["task_id", "model", "approach"]).outcome
    r = {"model": MNAME.get(m, "all models")}
    for s, n in SCORES:
        d, lo, hi = paired_diff(x, y, s); r[f"Δ{n}"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    pooled.append(r)
display(pd.DataFrame(pooled))
""")
    md("## 4. Task factors")
    chart("strata", "Score per stratum (context on, 9 prompts pooled)", """
for s, n in SCORES:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (f, fn) in zip(axes, FACTORS.items()):
        t = boot(A[A.set == "ctx"], ["model", f])
        grouped_bars(ax, t, f, STRATA, "model", MODELS, s, MCOLOR, names=MNAME, labels=False)
        ax.set_title(fn); ax.set_xlabel("stratum")
    axes[0].set_ylabel(n); axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"Football, Strategy A, context on: {n} per stratum")
    save(fig, FOLDER, f"a_strata_{s}")
""")
    chart("difficulty", "Score against task difficulty (context on, tertiles)", """
d = A[A.set == "ctx"].copy()
d["difficulty bin"] = pd.qcut(d.difficulty, 3, labels=["easy third", "middle third", "hard third"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
for ax, (s, n) in zip(axes, SCORES):
    t = metrics(d, ["model", "difficulty bin"])
    for m in MODELS:
        tt = t[t.model == m].set_index("difficulty bin").reindex(["easy third", "middle third", "hard third"])
        ax.plot(tt.index.astype(str), tt[s], marker="o", color=MCOLOR[m], lw=2)
    ax.set_title(n); ax.set_xlabel("difficulty of the task (tertiles)")
axes[0].set_ylabel("score (9 prompts pooled)"); axes[1].legend(handles=model_handles(), fontsize=8)
fig.suptitle("Football, Strategy A, context on: score vs difficulty")
save(fig, FOLDER, "a_difficulty")
""")
    chart("identifiers", "Name and team of the player in the given properties (high-G tasks, context on)", """
d = A[(A.set == "ctx") & (A.G_stratum == "high")]
conds = ["none", "team", "name", "both"]
CN = {"none": "neither", "team": "team only", "name": "name only", "both": "name and team"}
CC = dict(zip(conds, ["#cde2fb", "#86b6ef", "#3987e5", "#0d366b"]))
t = boot(d, ["model", "condition"])
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    grouped_bars(ax, t, "model", MODELS, "condition", conds, s, CC, names={**MNAME, **CN}, err=s != "abstention", labels=False)
    ax.set_title(n); ax.set_xticklabels(MN, rotation=15)
axes[0].legend(fontsize=8); pct(axes[2])
print(d.drop_duplicates("task_id").condition.value_counts().rename("tasks").to_frame().T)
fig.suptitle("Football, Strategy A, high-G tasks: effect of giving the player's name and/or team")
save(fig, FOLDER, "a_identifiers")
""")
    chart("properties", "Score per hidden property (context on, 9 prompts pooled)", """
d = A[A.set == "ctx"]
t = metrics(d, ["property", "model"])
ntask = d.drop_duplicates("task_id").property.value_counts()
props = ntask[ntask >= 4].index.tolist()
pv = t[t.property.isin(props)].pivot(index="property", columns="model", values="f1_ign").reindex(index=props, columns=MODELS)
pa = t[t.property.isin(props)].pivot(index="property", columns="model", values="abstention").reindex(index=props, columns=MODELS)
fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(props) + 1.5))
heat(axes[0], pv, xlabels=MN, ylabels=[f"{p} ({ntask[p]})" for p in props]); axes[0].set_title("F1 (abstentions ignored)")
heat(axes[1], pa, fmt="{:.0%}", xlabels=MN, ylabels=[""] * len(props)); axes[1].set_title("abstention")
fig.suptitle("Football, Strategy A, context on: per hidden property (tasks in brackets; properties with ≥ 4 tasks)")
save(fig, FOLDER, "a_properties")
""")
    chart("popularity", "Score against the popularity pop(p) of the hidden property", """
pop = popularity("football").set_index("property").popularity
d = A[A.set == "ctx"]
t = metrics(d, ["property"]).assign(pop=lambda x: x.property.map(pop), n=lambda x: x.tasks / 45)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (s, n) in zip(axes, SCORES):
    ax.scatter(t["pop"], t[s], s=12 + 6 * t.n, color=BLUE, alpha=0.75, edgecolor=SURFACE)
    for _, r in t.nlargest(8, "n").iterrows():
        ax.annotate(r.property, (r["pop"], r[s]), fontsize=7, color=INK2, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("pop(p): share of players with a value"); ax.set_title(n)
axes[0].set_ylabel("score (all models and prompts pooled; size = tasks)")
fig.suptitle("Football, Strategy A, context on: score vs popularity of the hidden property")
save(fig, FOLDER, "a_popularity")
""")
    chart("numeric_error", "How wrong are the wrong numbers?", """
d = A[A.set == "ctx"].copy()
d["rel_error"] = numeric_error(d)
d = d.dropna(subset=["rel_error"])
fig, ax = plt.subplots(figsize=(9, 4))
data = [d[d.model == m].rel_error.clip(1e-3, 1e3) for m in MODELS]
bp = ax.boxplot(data, tick_labels=MN, showfliers=False, patch_artist=True, widths=0.55)
for patch, m in zip(bp["boxes"], MODELS):
    patch.set_facecolor(MCOLOR[m]); patch.set_alpha(0.55)
logaxis(ax, "y"); ax.axhline(0.25, color=MUTED, ls=":", lw=1); ax.text(0.5, 0.26, "≈ scoring tolerance (similarity ≥ 0.75)", fontsize=7, color=MUTED)
ax.set_ylabel("|answer − truth| / |truth| (log)"); ax.grid(axis="x", visible=False)
ax.set_title("Football, Strategy A, context on: relative error of the wrong numeric answers")
save(fig, FOLDER, "a_numeric_error")
display(d.groupby("mname").rel_error.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]])
d["zero"] = pd.to_numeric(d.answer.str.strip('"'), errors="coerce") == 0
print("share of the wrong numeric answers that are 0:", d.groupby("mname").zero.mean().round(3).to_dict())
""")
    md("## 5. Stability and effort")
    chart("stability", "Agreement of the 3 runs", """
ag = agreement(runsA[runsA.set == "ctx"], ["model", "approach"])
fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
heat(axes[0], ag.pivot(index="model", columns="approach", values="same outcome in all runs").reindex(index=MODELS, columns=PROMPTS),
     fmt="{:.0%}", xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN)
axes[0].set_title("tasks whose 3 runs have the same outcome")
heat(axes[1], ag.pivot(index="model", columns="approach", values="f1_sd").reindex(index=MODELS, columns=PROMPTS),
     fmt="{:.3f}", xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN)
axes[1].set_title("sd of the F1 of the single runs")
fig.suptitle("Football, Strategy A, context on: stability over runs")
save(fig, FOLDER, "a_stability")
""")
    chart("effort", "Energy, tokens and token amplification per prompt (context on)", """
e = effA[effA.set == "ctx"].groupby(["model", "approach"]).agg(EC=("EC_run", "mean"), out=("tokens_out", "mean"), inn=("tokens_in", "mean"),
    proc=("tokens_proc", "sum"), user=("tokens_user", "sum"), lat=("latency_s", "mean")).reset_index()
e["amp"] = e.proc / e.user
fig, axes = plt.subplots(1, 3, figsize=(17, 3.8))
for ax, (c, n, f) in zip(axes, [("EC", "energy per run (Wh)", "{:.4f}"), ("out", "output tokens per run", "{:.0f}"), ("amp", "token amplification |T_process| / |T_user|", "{:.2f}")]):
    heat(ax, e.pivot(index="model", columns="approach", values=c).reindex(index=MODELS, columns=PROMPTS), fmt=f,
         xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN); ax.set_title(n)
fig.suptitle("Football, Strategy A, context on: effort per model and prompt")
save(fig, FOLDER, "a_effort")
""")
    chart("energy_f1", "Score against energy", """
e = effA[effA.set == "ctx"].groupby(["model", "approach"]).EC_run.mean().rename("EC").reset_index()
t = metrics(A[A.set == "ctx"], ["model", "approach"]).merge(e, on=["model", "approach"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (s, n) in zip(axes, SCORES):
    for m in MODELS:
        d = t[t.model == m]
        ax.scatter(d.EC, d[s], color=MCOLOR[m], s=40, edgecolor=SURFACE, zorder=3)
    logaxis(ax, "x"); ax.set_xlabel("energy per run (Wh, log)"); ax.set_title(n)
axes[0].set_ylabel("score (one dot per prompt)"); axes[1].legend(handles=model_handles(), fontsize=8)
fig.suptitle("Football, Strategy A, context on: score vs energy per run")
save(fig, FOLDER, "a_energy_vs_f1")
""")

# ============================================================================================ football_b
if MODE == "football_b":
    md("""
    # Strategy B on the football graph (context on)

    **Strategy B** is a single-turn agent: the model plans up to 3 web searches (Ollama web search), reads the top
    results (evidence capped at 3,000 tokens) and answers with citations. Two variants on the same retrieval:
    **search** (`search_evidence`: answer from the evidence only) and **search + memory** (`search_memory`: the
    synthesis call replayed on the same evidence, now allowing the model's own knowledge). 5 models, 3 runs each,
    239 context-on tasks (`football-v1-ctx-idmix`).
    """)
    md(SCORING)
    code(SETUP + """
v = load_voted(("football", "ctx"))
B = v[v.strategy.isin(["B", "B_memory"])].copy()
runs = load_runs(("football", "ctx"))
runsB = runs[runs.strategy.isin(["B", "B_memory"])]
det = load_b_detail("football")
dom = load_domains("football")
eff = load_effort()
effB = eff[(eff.dataset == "football") & eff.strategy.isin(["B", "B_memory"])]
V = ["search_evidence", "search_memory"]
VC = {"search_evidence": "#86b6ef", "search_memory": "#1c5cab"}
print(B.groupby("approach").size().to_dict(), "voted rows;", len(runsB), "scored runs;", len(det), "run details")
""")
    md("## 1. Overall results")
    chart("overall_table", "Scores per model and variant", """
t = boot(B, ["approach", "mname"])
table(t[["approach", "mname", "tasks", "correct", "wrong", "abstained", "precision", "recall", "f1", "f1_lo", "f1_hi", "f1_ign", "f1_ign_lo", "f1_ign_hi", "abstention", "error"]],
      FOLDER, "b_scores", "Table B1. Scores per model and variant (voted over 3 runs; 95% bootstrap intervals)")
""")
    chart("bars", "F1 per model, search vs search + memory", """
t = boot(B, ["approach", "model"])
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    grouped_bars(ax, t, "model", MODELS, "approach", V, s, VC, names={**MNAME, **PNAME}, err=s != "abstention", fmt="{:.0%}" if s == "abstention" else "{:.2f}")
    ax.set_title(n); ax.set_xticklabels(MN, rotation=15)
axes[0].legend(loc="upper left", fontsize=8); pct(axes[2])
fig.suptitle("Football, Strategy B: score and abstention per model")
save(fig, FOLDER, "b_bars")
w = t.assign(model=t.model.map(MNAME), approach=t.approach.map(VNAME)).pivot(index="model", columns="approach", values=["f1", "f1_ign", "abstention"]).reindex(MN)
w.columns = [f"{ {'f1': 'F1', 'f1_ign': 'F1 ign', 'abstention': 'abstention'}[a]} · {b}" for a, b in w.columns]
for n in ("F1", "F1 ign"):
    w[f"Δ{n} (memory − search)"] = w[f"{n} · search + memory"] - w[f"{n} · search"]
table(w, FOLDER, "b_scores_wide", "Table B2. F1, F1 ign and abstention side by side", pct=[c for c in w.columns if c.startswith("abstention")])
""")
    chart("outcomes", "Outcomes per model and variant", """
t = metrics(B, ["model", "approach"])
t["cfg"] = t.model.map(MNAME) + " · " + t.approach.map({"search_evidence": "search", "search_memory": "+ memory"})
fig, ax = plt.subplots(figsize=(9, 5))
stacked_outcomes(ax, t, "cfg", [f"{MNAME[m]} · {x}" for m in MODELS for x in ("search", "+ memory")])
ax.legend(loc="lower right", bbox_to_anchor=(1, -0.18), ncol=3)
ax.set_title("Football, Strategy B: share of correct, wrong and abstained answers")
save(fig, FOLDER, "b_outcomes")
o = t.assign(**{"correct %": t.correct / t.tasks, "wrong %": t.wrong / t.tasks, "abstained %": t.abstained / t.tasks,
                "wrong per correct": t.wrong / t.correct.replace(0, np.nan)})
table(o[["model", "approach", "tasks", "correct", "wrong", "abstained", "correct %", "wrong %", "abstained %", "wrong per correct"]],
      FOLDER, "b_outcomes", "Table B3. Outcomes of the voted answers", pct=["correct %", "wrong %", "abstained %"], digits=2)
""")
    chart("pr", "Precision against recall: what memory changes", """
t = metrics(B, ["model", "approach"]).set_index(["model", "approach"])
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in MODELS:
    a, b = t.loc[(m, "search_evidence")], t.loc[(m, "search_memory")]
    ax.annotate("", (b.recall, b.precision), (a.recall, a.precision), arrowprops=dict(arrowstyle="->", color=MCOLOR[m], lw=1.6))
    ax.scatter(a.recall, a.precision, s=50, facecolor="none", edgecolor=MCOLOR[m], linewidth=1.5, zorder=3)
    ax.scatter(b.recall, b.precision, s=50, color=MCOLOR[m], zorder=3)
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, None); ax.set_ylim(0, None)
ax.legend(handles=model_handles() + [plt.Line2D([], [], ls="", marker="o", mfc="none", color=INK2, label="search"),
                                     plt.Line2D([], [], ls="", marker="o", color=INK2, label="search + memory")], fontsize=8)
ax.set_title("Football, Strategy B: precision vs recall (arrow: search → search + memory)")
save(fig, FOLDER, "b_precision_recall")
pr = t[["precision", "recall"]].unstack("approach")
pr.columns = [f"{a} · {VNAME[b]}" for a, b in pr.columns]
pr["Δprecision"] = pr["precision · search + memory"] - pr["precision · search"]
pr["Δrecall"] = pr["recall · search + memory"] - pr["recall · search"]
table(pr.reindex(MODELS), FOLDER, "b_precision_recall", "Table B4. Precision and recall, and what memory changes")
""")
    md("## 2. What memory adds")
    chart("transitions", "Task-by-task transitions from search to search + memory", """
k = ["model", "task_id"]
j = B[B.approach == "search_evidence"].set_index(k).outcome.rename("search").to_frame().join(
    B[B.approach == "search_memory"].set_index(k).outcome.rename("memory"))
O = ["correct", "wrong", "abstained"]
fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
for ax, m in zip(axes, MODELS):
    ct = pd.crosstab(j.loc[m].search, j.loc[m].memory).reindex(index=O, columns=O, fill_value=0)
    heat(ax, ct, fmt="{:.0f}", xlabels=O, ylabels=O, cbar=False); ax.set_title(MNAME[m])
    ax.set_xlabel("search + memory"); ax.set_ylabel("search" if m == MODELS[0] else "")
fig.suptitle("Football, Strategy B: outcome with search (rows) vs with search + memory (columns), tasks")
save(fig, FOLDER, "b_memory_transitions")
rows = []
for m in MODELS:
    x, y = j.loc[m].search, j.loc[m].memory
    r = {"model": MNAME[m], **paired(x, y)}
    for s, n in SCORES:
        d, lo, hi = paired_diff(x, y, s); r[f"Δ{n}"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    rows.append(r)
table(pd.DataFrame(rows).rename(columns={"only first right": "only search right", "only second right": "only memory right"}),
      FOLDER, "b_memory_paired", "Table B5. Search vs search + memory, task by task (exact McNemar on correct; paired bootstrap Δ)")
tr = []
for m in MODELS:
    ct = pd.crosstab(j.loc[m].search, j.loc[m].memory).reindex(index=O, columns=O, fill_value=0)
    for a in O:
        tr.append({"model": m, "with search": a, **{f"→ {b} with memory": int(ct.loc[a, b]) for b in O}, "tasks": int(ct.loc[a].sum())})
table(pd.DataFrame(tr), FOLDER, "b_memory_transitions", "Table B6. Transitions of each task from search (row) to search + memory (columns)")
""")
    chart("citations", "Answers with and without a citation (search + memory)", """
k = ["variant", "model", "task_id", "run_index"]
rr = runsB.merge(det[k + ["answer_cited", "answer_uncited", "entity_match", "evidence_passed", "evidence_tokens", "cited_domains"]], on=k, how="left")
mem = rr[(rr.variant == "search_memory") & (rr.outcome != "abstained")].copy()
mem["source"] = np.where(mem.answer_cited, "cites the evidence", "no citation (memory)")
t = metrics(mem, ["model", "source"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
grouped_bars(axes[0], t.assign(share=t.tasks / t.groupby("model").tasks.transform("sum")), "model", MODELS, "source",
             ["cites the evidence", "no citation (memory)"], "share", {"cites the evidence": "#86b6ef", "no citation (memory)": ORANGE},
             names=MNAME, err=False, fmt="{:.0%}")
axes[0].set_title("share of the answers (single runs)"); pct(axes[0])
grouped_bars(axes[1], t, "model", MODELS, "source", ["cites the evidence", "no citation (memory)"], "precision",
             {"cites the evidence": "#86b6ef", "no citation (memory)": ORANGE}, names=MNAME, err=False)
axes[1].set_title("precision of those answers")
for ax in axes: ax.set_xticklabels(MN, rotation=15)
axes[0].legend(fontsize=8)
fig.suptitle("Football, Strategy B search + memory: answers grounded in the evidence vs from memory")
save(fig, FOLDER, "b_citations")
t["share"] = t.tasks / t.groupby("model").tasks.transform("sum")
table(t[["model", "source", "tasks", "share", "correct", "wrong", "precision"]].rename(columns={"tasks": "answers"}),
      FOLDER, "b_citations", "Table B7. Search + memory: answers that cite the evidence vs answers from memory (single runs)")
""")
    md("## 3. Retrieval")
    chart("entity", "Did the planner find the right player?", """
k = ["variant", "model", "task_id", "run_index"]
d = det[det.variant == "search_evidence"].merge(runsB[k + ["outcome", "has_name", "has_team", "G_stratum"]], on=k)
t = d.groupby("model").agg(named=("entity_named", "mean"), match=("entity_match", "mean")).reindex(MODELS)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(MODELS))
axes[0].bar(x - 0.2, t.named, 0.38, color="#86b6ef", label="names an entity")
axes[0].bar(x + 0.2, t.match, 0.38, color="#1c5cab", label="names the right player")
axes[0].set_xticks(x); axes[0].set_xticklabels(MN, rotation=15); pct(axes[0]); axes[0].legend(fontsize=8)
axes[0].set_title("plans that identify a player (single runs)")
g = d.groupby(["has_name", "entity_match"]).outcome.value_counts(normalize=True).unstack().fillna(0)
nn = d.groupby(["has_name", "entity_match"]).size()
g.index = [f"{'name given' if a else 'no name'} · {'right player' if b else 'no / wrong player'} (n = {nn[(a, b)]})" for a, b in g.index]
left = np.zeros(len(g))
for o in ("correct", "wrong", "abstained"):
    axes[1].barh(range(len(g)), g[o], left=left, color=OCOLOR[o], label=o, edgecolor=SURFACE); left += g[o].to_numpy()
axes[1].set_yticks(range(len(g))); axes[1].set_yticklabels(g.index); pct(axes[1], "x"); axes[1].legend(fontsize=8, loc="lower right")
axes[1].set_title("outcome by name given and player found (all models)")
fig.suptitle("Football, Strategy B search: entity identification in the search plan")
save(fig, FOLDER, "b_entity")
t2 = d.groupby("model").agg(runs=("task_id", "size"), named=("entity_named", "mean"), match=("entity_match", "mean"),
                            answered=("answered", "mean")).reindex(MODELS).reset_index()
table(t2, FOLDER, "b_entity_by_model", "Table B8. Search plans that name an entity / the right player, and runs that answer")
e2 = metrics(d.assign(name=np.where(d.has_name, "name given", "no name"), player=np.where(d.entity_match, "right player", "no / wrong player")),
             ["name", "player"])
e2["share of runs"] = e2.tasks / e2.groupby("name").tasks.transform("sum")
table(e2[["name", "player", "tasks", "share of runs", "correct", "wrong", "abstained", "precision", "abstention"]].rename(columns={"tasks": "runs"}),
      FOLDER, "b_entity_outcomes", "Table B9. Outcome by name given and player found (search, single runs, all models)", pct=["share of runs"])
""")
    chart("domains", "Where the evidence comes from", """
dd = dom[dom.variant == "search_evidence"].groupby("domain")[["passed", "cited"]].sum().sort_values("passed", ascending=False)
top = dd.head(15)
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(top))
ax.barh(y, top.passed, color="#86b6ef", label="in the evidence of a run")
ax.barh(y, top.cited, color="#1c5cab", label="cited by an answer")
ax.set_yticks(y); ax.set_yticklabels(top.index); ax.invert_yaxis(); ax.legend(fontsize=8); ax.grid(axis="y", visible=False)
ax.set_xlabel("runs (all models)")
ax.set_title("Football, Strategy B search: the 15 most frequent evidence domains")
save(fig, FOLDER, "b_domains")
print("share of evidence slots from the top 5 domains:", round(dd.passed.head(5).sum() / dd.passed.sum(), 3), "; distinct domains:", len(dd))
td = dd.head(20).reset_index()
td["share of evidence"] = td.passed / dd.passed.sum()
td["cited / passed"] = td.cited / td.passed
table(td.rename(columns={"passed": "runs with it in the evidence", "cited": "runs citing it"}), FOLDER, "b_domains",
      "Table B10. The 20 most frequent evidence domains (search, all models)", pct=["share of evidence", "cited / passed"])
""")
    chart("domain_precision", "Precision of the answers by cited domain", """
k = ["variant", "model", "task_id", "run_index"]
rr = runsB.merge(det[k + ["cited_domains"]], on=k)
rr = rr[(rr.outcome != "abstained") & rr.cited_domains.notna()]
ex = rr.assign(domain=rr.cited_domains.str.split(";")).explode("domain")
t = metrics(ex, ["domain"]).query("tasks >= 25").sort_values("tasks", ascending=False).head(12)
fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(t))
ax.barh(y, t.precision, color=BLUE)
for yy, p_, n_ in zip(y, t.precision, t.tasks):
    ax.text(p_ + 0.01, yy, f"{p_:.2f} (n = {n_})", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(t.domain); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.set_xlabel("precision of the answers that cite the domain (single runs, both variants)")
ax.set_title("Football, Strategy B: precision by cited domain (domains cited ≥ 25 times)")
save(fig, FOLDER, "b_domain_precision")
table(t[["domain", "tasks", "correct", "wrong", "precision"]].rename(columns={"tasks": "answers citing it"}), FOLDER, "b_domain_precision",
      "Table B11. Precision of the answers by cited domain (single runs, both variants, domains cited ≥ 25 times)")
""")
    chart("evidence", "Amount of evidence and answering", """
k = ["variant", "model", "task_id", "run_index"]
d = det[det.variant == "search_evidence"].merge(runsB[k + ["outcome"]], on=k)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
t = d.groupby("model").agg(ev=("evidence_passed", "mean"), tok=("evidence_tokens", "mean"), trunc=("evidence_truncated", "mean"), ans=("answered", "mean")).reindex(MODELS)
x = np.arange(len(MODELS))
axes[0].bar(x, t.tok, color=[MCOLOR[m] for m in MODELS])
for xx, a, b in zip(x, t.tok, t.trunc):
    axes[0].text(xx, a + 30, f"{b:.0%} truncated", ha="center", fontsize=7.5, color=INK2)
axes[0].set_xticks(x); axes[0].set_xticklabels(MN, rotation=15); axes[0].set_ylabel("evidence tokens per run (cap 3,000)")
axes[0].set_title("evidence passed to the synthesis"); axes[0].grid(axis="x", visible=False)
d["bin"] = pd.cut(d.evidence_passed, [-1, 2, 4, 6, 20], labels=["0–2", "3–4", "5–6", "7+"])
g = d.groupby(["model", "bin"], observed=True).answered.mean().reset_index()
for m in MODELS:
    gg = g[g.model == m].set_index("bin").reindex(["0–2", "3–4", "5–6", "7+"]); axes[1].plot(gg.index.astype(str), gg.answered, marker="o", color=MCOLOR[m], lw=2)
pct(axes[1]); axes[1].set_xlabel("evidence items passed"); axes[1].set_ylabel("runs that answer"); axes[1].legend(handles=model_handles(), fontsize=8)
axes[1].set_title("answer rate vs evidence items")
fig.suptitle("Football, Strategy B search: evidence and answering")
save(fig, FOLDER, "b_evidence")
table(t.rename(columns={"ev": "evidence items", "tok": "evidence tokens", "trunc": "truncated", "ans": "answered"}).reset_index(),
      FOLDER, "b_evidence", "Table B12. Evidence passed to the synthesis per run (search)", digits=1)
ab = d.groupby(["model", "bin"], observed=True).agg(runs=("task_id", "size"), answered=("answered", "mean")).reset_index()
ab = ab.pivot(index="model", columns="bin", values="answered").reindex(MODELS)
ab.columns = [f"answer rate, {c} items" for c in ab.columns]
table(ab, FOLDER, "b_answer_rate_by_evidence", "Table B13. Share of runs that answer, by number of evidence items", pct=list(ab.columns))
""")
    md("## 4. Task factors")
    chart("strata", "Score per stratum", """
for s, n in SCORES:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (f, fn) in zip(axes, FACTORS.items()):
        t = boot(B, ["approach", f])
        grouped_bars(ax, t, f, STRATA, "approach", V, s, VC, names=PNAME)
        ax.set_title(fn); ax.set_xlabel("stratum")
    axes[0].set_ylabel(f"{n} (5 models pooled)"); axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"Football, Strategy B: {n} per stratum")
    save(fig, FOLDER, f"b_strata_{s}")
st = pd.concat([boot(B, ["approach", f]).rename(columns={f: "stratum"}).assign(factor=fn) for f, fn in FACTORS.items()])
table(st[["factor", "stratum", "approach", "tasks", "correct", "wrong", "abstained", "f1", "f1_lo", "f1_hi", "f1_ign", "f1_ign_lo", "f1_ign_hi", "abstention"]],
      FOLDER, "b_strata", "Table B14. Scores per stratum (5 models pooled)")
sm = pd.concat([metrics(B, ["model", "approach", f]).rename(columns={f: "stratum"}).assign(factor=fn) for f, fn in FACTORS.items()])
sm = sm.assign(model=sm.model.map(MNAME), approach=sm.approach.map(VNAME)).pivot_table(index=["factor", "model"], columns=["approach", "stratum"], values="f1")
sm.columns = [f"F1 · {a} · {b}" for a, b in sm.columns]
table(sm[[c for a in ("search", "search + memory") for c in (f"F1 · {a} · low", f"F1 · {a} · mid", f"F1 · {a} · high")]],
      FOLDER, "b_strata_by_model", "Table B15. F1 per stratum and model")
""")
    chart("identifiers", "Name and team in the given properties (high-G tasks)", """
d = B[B.G_stratum == "high"]
conds = ["none", "team", "name", "both"]
CN = {"none": "neither", "team": "team only", "name": "name only", "both": "name and team"}
t = boot(d, ["approach", "condition"])
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    grouped_bars(ax, t, "condition", conds, "approach", V, s, VC, names={**PNAME, **CN}, err=s != "abstention")
    ax.set_title(n)
axes[0].legend(fontsize=8); pct(axes[2])
fig.suptitle("Football, Strategy B, high-G tasks (5 models pooled): effect of the player's name and team")
save(fig, FOLDER, "b_identifiers")
t["condition"] = t.condition.map(CN)
table(t[["condition", "approach", "tasks", "correct", "wrong", "abstained", "f1", "f1_lo", "f1_hi", "f1_ign", "f1_ign_lo", "f1_ign_hi", "abstention"]],
      FOLDER, "b_identifiers", "Table B16. High-G tasks: name and team of the player (5 models pooled)")
tm = metrics(d.assign(name=np.where(d.has_name, "name given", "no name")), ["model", "approach", "name"])
tm = tm.assign(model=tm.model.map(MNAME), approach=tm.approach.map(VNAME)).pivot_table(index=["model", "approach"], columns="name", values=["f1", "f1_ign", "abstention"])
tm.columns = [f"{ {'f1': 'F1', 'f1_ign': 'F1 ign', 'abstention': 'abstention'}[a]} · {b}" for a, b in tm.columns]
table(tm, FOLDER, "b_identifiers_by_model", "Table B17. High-G tasks: with and without the player's name, per model", pct=[c for c in tm.columns if c.startswith("abstention")])
""")
    chart("difficulty", "Score against difficulty", """
d = B.copy(); d["bin"] = pd.qcut(d.difficulty, 3, labels=["easy third", "middle third", "hard third"])
t = boot(d, ["approach", "bin"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (s, n) in zip(axes, SCORES):
    grouped_bars(ax, t, "bin", ["easy third", "middle third", "hard third"], "approach", V, s, VC, names=PNAME); ax.set_title(n)
axes[0].legend(fontsize=8)
fig.suptitle("Football, Strategy B (5 models pooled): score vs difficulty tertile")
save(fig, FOLDER, "b_difficulty")
table(t[["bin", "approach", "tasks", "correct", "wrong", "abstained", "f1", "f1_lo", "f1_hi", "f1_ign", "f1_ign_lo", "f1_ign_hi", "abstention"]].rename(columns={"bin": "difficulty"}),
      FOLDER, "b_difficulty", "Table B18. Scores per difficulty tertile (5 models pooled)")
""")
    chart("properties", "Score per hidden property", """
t = metrics(B, ["property", "approach"])
ntask = B.drop_duplicates("task_id").property.value_counts()
props = ntask[ntask >= 4].index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(props) + 1.5))
for ax, (c, n, f) in zip(axes, [("f1_ign", "F1 (abstentions ignored)", "{:.2f}"), ("abstention", "abstention", "{:.0%}")]):
    pv = t[t.property.isin(props)].pivot(index="property", columns="approach", values=c).reindex(index=props, columns=V)
    heat(ax, pv, fmt=f, xlabels=["search", "+ memory"], ylabels=[f"{p} ({ntask[p]})" for p in props] if c == "f1_ign" else [""] * len(props))
    ax.set_title(n)
fig.suptitle("Football, Strategy B (5 models pooled): per hidden property (tasks in brackets)")
save(fig, FOLDER, "b_properties")
pw = t[t.property.isin(props)].assign(approach=lambda x: x.approach.map(VNAME)).pivot(index="property", columns="approach",
     values=["correct", "wrong", "abstained", "f1", "f1_ign", "abstention"]).reindex(props)
pw.columns = [f"{a} · {b}" for a, b in pw.columns]
pw.insert(0, "tasks", ntask.reindex(props).values)
table(pw, FOLDER, "b_properties", "Table B19. Per hidden property (5 models pooled; properties with ≥ 4 tasks)", pct=[c for c in pw.columns if c.startswith("abstention")])
""")
    chart("numeric_error", "Relative error of the wrong numeric answers", """
d = B.copy(); d["rel_error"] = numeric_error(d); d = d.dropna(subset=["rel_error"])
fig, ax = plt.subplots(figsize=(10, 4))
data, labels, cols = [], [], []
for m in MODELS:
    for a in V:
        data.append(d[(d.model == m) & (d.approach == a)].rel_error.clip(1e-3, 1e3)); labels.append(f"{MNAME[m]}\\n{'search' if a == V[0] else '+ memory'}"); cols.append(VC[a])
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.55)
for p_, c in zip(bp["boxes"], cols): p_.set_facecolor(c)
logaxis(ax, "y"); ax.axhline(0.25, color=MUTED, ls=":"); ax.set_ylabel("|answer − truth| / |truth| (log)"); ax.tick_params(axis="x", labelsize=7)
ax.set_title("Football, Strategy B: relative error of the wrong numeric answers"); ax.grid(axis="x", visible=False)
save(fig, FOLDER, "b_numeric_error")
d["zero"] = pd.to_numeric(d.answer.str.strip('"'), errors="coerce") == 0
ne = d.groupby(["model", "approach"]).agg(wrong_numeric=("rel_error", "size"), q25=("rel_error", lambda x: x.quantile(0.25)),
     median=("rel_error", "median"), q75=("rel_error", lambda x: x.quantile(0.75)), zero=("zero", "mean")).reset_index()
table(ne.rename(columns={"wrong_numeric": "wrong numeric answers", "q25": "rel. error 25%", "median": "rel. error median",
                         "q75": "rel. error 75%", "zero": "share that are 0"}), FOLDER, "b_numeric_error",
      "Table B20. Relative error of the wrong numeric answers", pct=["share that are 0"], digits=2)
""")
    md("## 5. Stability and effort")
    chart("stability", "Agreement of the 3 runs", """
ag = agreement(runsB, ["model", "approach"])
fig, ax = plt.subplots(figsize=(8, 3.6))
grouped_bars(ax, ag, "model", MODELS, "approach", V, "same outcome in all runs", VC, names={**MNAME, **PNAME}, err=False, fmt="{:.0%}")
pct(ax); ax.set_xticklabels(MN, rotation=15); ax.legend(fontsize=8)
ax.set_title("Football, Strategy B: tasks whose 3 runs have the same outcome")
save(fig, FOLDER, "b_stability")
table(ag, FOLDER, "b_stability", "Table B21. Agreement of the 3 runs and sd of the single-run F1")
""")
    chart("effort", "Energy, searches and tokens", """
e = effB.groupby(["model", "approach"]).agg(EC_llm=("EC_llm", "mean"), EC_run=("EC_run", "mean"), searches=("searches", "mean"),
    tokens_in=("tokens_in", "mean"), tokens_out=("tokens_out", "mean"), latency=("latency_s", "mean"), proc=("tokens_proc", "sum"), user=("tokens_user", "sum")).reset_index()
e["amp"] = e.proc / e.user
e["search share of energy"] = 1 - e.EC_llm / e.EC_run
table(e.drop(columns=["proc", "user"]).rename(columns={"EC_llm": "Wh, LLM calls", "EC_run": "Wh, run", "searches": "searches per run",
      "tokens_in": "input tokens", "tokens_out": "output tokens", "latency": "latency (s)", "amp": "token amplification"}),
      FOLDER, "b_effort", "Table B22. Effort per run (means; the memory variant reuses the retrieval, its searches are counted again)",
      pct=["search share of energy"], digits=3)
d = e[e.approach == "search_evidence"].set_index("model").reindex(MODELS)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
x = np.arange(len(MODELS))
axes[0].bar(x, d.EC_llm, color="#1c5cab", label="LLM calls"); axes[0].bar(x, d.EC_run - d.EC_llm, bottom=d.EC_llm, color="#cde2fb", label=f"searches ({E_SEARCH_WH} Wh each)")
axes[0].set_ylabel("Wh per run"); axes[0].set_title("energy per run (search)"); axes[0].legend(fontsize=8)
axes[1].bar(x, d.tokens_in, color="#86b6ef", label="input"); axes[1].bar(x, d.tokens_out, bottom=d.tokens_in, color="#0d366b", label="output")
axes[1].set_title("tokens per run (search)"); axes[1].legend(fontsize=8, loc="lower right")
axes[2].bar(x, d.searches, color=[MCOLOR[m] for m in MODELS]); axes[2].set_title("web searches per run (search)")
for ax in axes: ax.set_xticks(x); ax.set_xticklabels(MN, rotation=15); ax.grid(axis="x", visible=False)
fig.suptitle("Football, Strategy B search: effort per run")
save(fig, FOLDER, "b_effort")
""")

# ============================================================================================ football_ab
if MODE == "football_ab":
    md("""
    # Strategy A against Strategy B on the football graph (context on)

    Both strategies on the same 239 context-on tasks (`football-v1-ctx-idmix`), the same 5 models and 3 runs:
    **A zero-shot**, **A mean** (the 9 prompts pooled), **A best** (each model's best prompt by F1 on these tasks,
    an optimistic upper bound for A), **B search** and **B search + memory**.
    """)
    md(SCORING)
    code(SETUP + """
v = load_voted(("football", "ctx"))
runs = load_runs(("football", "ctx"))
eff = load_effort(); eff = eff[(eff.dataset == "football") & (eff.set == "ctx")]
mp = metrics(v[v.strategy == "A"], ["model", "approach"])
BEST = mp.loc[mp.groupby("model").f1.idxmax()].set_index("model").approach.to_dict()
BEST_IGN = mp.loc[mp.groupby("model").f1_ign.idxmax()].set_index("model").approach.to_dict()
def cfg(r):
    if r.strategy == "A":
        return "A best" if r.approach == BEST[r.model] else ("A zero-shot" if r.approach == "zero" else None)
    return "B search" if r.approach == "search_evidence" else "B search + memory"
v["cfg"] = v.apply(cfg, axis=1)
C = ["A zero-shot", "A best", "B search", "B search + memory"]
CC = {"A zero-shot": "#cde2fb", "A best": "#86b6ef", "A mean": "#c9c8c2", "B search": "#eb6834", "B search + memory": "#a8380c"}
pd.DataFrame({"best prompt (F1)": BEST, "best prompt (F1 ign)": BEST_IGN}).rename(index=MNAME)
""")
    md("## 1. Headline comparison")
    chart("bars", "Score per model and approach", """
t = pd.concat([boot(v[v.cfg.notna()], ["model", "cfg"]), boot(v[v.strategy == "A"], ["model"]).assign(cfg="A mean")])
order = ["A zero-shot", "A mean", "A best", "B search", "B search + memory"]
fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
for ax, (s, n) in zip(axes, SCORES):
    grouped_bars(ax, t, "model", MODELS, "cfg", order, s, CC, names=MNAME, labels=True); ax.set_title(n)
axes[0].legend(fontsize=8, loc="upper left", ncol=2); axes[0].set_ylabel("score")
fig.suptitle("Football: Strategy A vs Strategy B per model")
save(fig, FOLDER, "ab_bars")
display(t.assign(model=t.model.map(MNAME)).pivot(index="model", columns="cfg", values=["f1", "f1_ign", "abstention"]).round(3))
""")
    chart("forest", "Paired differences B − A (same tasks)", """
rows = []
for m in MODELS:
    for a in ("A zero-shot", "A best"):
        for b in ("B search", "B search + memory"):
            x = v[(v.model == m) & (v.cfg == a)].set_index("task_id").outcome
            y = v[(v.model == m) & (v.cfg == b)].set_index("task_id").outcome
            for s, n in SCORES:
                d, lo, hi = paired_diff(x, y, s)
                rows.append({"model": m, "pair": f"{b} − {a}", "score": n, "diff": d, "lo": lo, "hi": hi, **paired(x, y)})
pdf = pd.DataFrame(rows)
pairs = pdf.pair.unique().tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (s, n) in zip(axes, SCORES):
    d = pdf[pdf.score == n]
    for i, p in enumerate(pairs):
        for j, m in enumerate(MODELS):
            r = d[(d.pair == p) & (d.model == m)].iloc[0]
            y = i * 6 + j
            ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", color=MCOLOR[m], ms=5, capsize=2)
    ax.axvline(0, color=INK2, lw=1); ax.set_title(n); ax.set_xlabel("difference (B − A)")
    ax.set_yticks([i * 6 + 2 for i in range(len(pairs))]); ax.set_yticklabels(pairs); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
axes[1].legend(handles=model_handles(), fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle("Football: paired difference B − A with 95% interval (right of 0: B better)")
save(fig, FOLDER, "ab_forest")
display(pdf[pdf.score == "F1"][["model", "pair", "diff", "lo", "hi", "both right", "only first right", "only second right", "neither right", "p (McNemar)"]]
        .assign(model=lambda x: x.model.map(MNAME)).rename(columns={"only first right": "only A right", "only second right": "only B right"}))
""")
    chart("taskwise", "Task-by-task: who gets each task right", """
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, b in zip(axes, ("B search", "B search + memory")):
    rows = []
    for m in MODELS:
        x = v[(v.model == m) & (v.cfg == "A best")].set_index("task_id").outcome
        y = v[(v.model == m) & (v.cfg == b)].set_index("task_id").outcome
        rows.append({"model": m, **paired(x, y)})
    t = pd.DataFrame(rows).set_index("model").reindex(MODELS)
    parts = [("both right", AQUA), ("only first right", "#86b6ef"), ("only second right", ORANGE)]
    left = np.zeros(len(t))
    for c, col in parts:
        ax.barh(range(len(t)), t[c], left=left, color=col, label=c.replace("first", "A best").replace("second", b), edgecolor=SURFACE)
        for yy, l_, w_ in zip(range(len(t)), left, t[c]):
            if w_ >= 4: ax.text(l_ + w_ / 2, yy, int(w_), ha="center", va="center", fontsize=7.5)
        left += t[c].to_numpy()
    for yy, p_ in enumerate(t["p (McNemar)"]):
        ax.text(left[yy] + 1, yy, f"p = {p_:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(MN); ax.invert_yaxis(); ax.set_title(f"A best vs {b}")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3); ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("tasks answered correctly"); ax.grid(axis="y", visible=False)
fig.suptitle("Football: tasks right by A best, by B, or by both (exact McNemar p)")
save(fig, FOLDER, "ab_taskwise")
""")
    chart("pr", "Precision against recall, all approaches", """
t = metrics(v[v.cfg.notna()], ["model", "cfg"])
MK = {"A zero-shot": "s", "A best": "o", "B search": "D", "B search + memory": "^"}
fig, ax = plt.subplots(figsize=(8, 5.5))
for _, r in t.iterrows():
    ax.scatter(r.recall, r.precision, marker=MK[r.cfg], s=60, color=MCOLOR[r.model], edgecolor=SURFACE, zorder=3)
for m in MODELS:
    d = t[t.model == m].set_index("cfg").reindex(C)
    ax.plot(d.recall, d.precision, color=MCOLOR[m], lw=0.8, alpha=0.5)
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, None); ax.set_ylim(0, None)
h = model_handles() + [plt.Line2D([], [], ls="", marker="", label="")] + [plt.Line2D([], [], ls="", marker=mk, color=INK2, label=c) for c, mk in MK.items()]
ax.legend(handles=h, fontsize=8, loc="upper right")
ax.set_title("Football: precision vs recall of A and B")
save(fig, FOLDER, "ab_precision_recall")
""")
    md("## 2. Where each strategy works")
    chart("strata", "Score per stratum", """
for s, n in SCORES:
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    for ax, (f, fn) in zip(axes, FACTORS.items()):
        t = boot(v[v.cfg.notna()], ["cfg", f])
        grouped_bars(ax, t, f, STRATA, "cfg", C, s, CC, labels=False); ax.set_title(fn); ax.set_xlabel("stratum")
    axes[0].set_ylabel(f"{n} (5 models pooled)"); axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"Football: {n} per stratum, A vs B")
    save(fig, FOLDER, f"ab_strata_{s}")
""")
    chart("identifiers", "Name and team in the given properties (high-G tasks)", """
d = v[v.cfg.notna() & (v.G_stratum == "high")].copy()
d["name given"] = np.where(d.has_name, "name given", "no name")
t = boot(d, ["cfg", "name given"])
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    grouped_bars(ax, t, "cfg", C, "name given", ["no name", "name given"], s, {"no name": "#cde2fb", "name given": "#1c5cab"}, err=s != "abstention")
    ax.set_title(n); ax.set_xticklabels(C, rotation=15)
axes[0].legend(fontsize=8); pct(axes[2])
fig.suptitle("Football, high-G tasks (5 models pooled): effect of the player's name on A and B")
save(fig, FOLDER, "ab_identifiers")
""")
    chart("difficulty", "Score against difficulty", """
d = v[v.cfg.notna()].copy(); d["bin"] = pd.qcut(d.difficulty, 3, labels=["easy third", "middle third", "hard third"])
t = boot(d, ["cfg", "bin"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (s, n) in zip(axes, SCORES):
    grouped_bars(ax, t, "bin", ["easy third", "middle third", "hard third"], "cfg", C, s, CC, labels=False); ax.set_title(n)
axes[0].legend(fontsize=8)
fig.suptitle("Football (5 models pooled): score vs difficulty, A vs B")
save(fig, FOLDER, "ab_difficulty")
""")
    chart("properties", "Score per hidden property and approach", """
t = metrics(v[v.cfg.notna()], ["property", "cfg"])
ntask = v.drop_duplicates("task_id").property.value_counts(); props = ntask[ntask >= 4].index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 0.32 * len(props) + 1.5))
for ax, (s, n) in zip(axes, SCORES):
    pv = t[t.property.isin(props)].pivot(index="property", columns="cfg", values=s).reindex(index=props, columns=C)
    heat(ax, pv, xlabels=C, ylabels=[f"{p} ({ntask[p]})" for p in props] if s == "f1" else [""] * len(props)); ax.set_title(n)
fig.suptitle("Football (5 models pooled): score per hidden property and approach")
save(fig, FOLDER, "ab_properties")
""")
    chart("complementarity", "Complementarity: an oracle that takes whichever is right", """
rows = []
for m in MODELS:
    a = v[(v.model == m) & (v.cfg == "A best")].set_index("task_id").outcome
    b = v[(v.model == m) & (v.cfg == "B search + memory")].set_index("task_id").outcome
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    orc = np.where((j.a == "correct") | (j.b == "correct"), "correct", np.where((j.a == "wrong") | (j.b == "wrong"), "wrong", "abstained"))
    # B first, A when B abstains: a deployable combination
    fb = np.where(j.b != "abstained", j.b, j.a)
    for name, o in (("A best", j.a), ("B search + memory", j.b), ("B, then A if B abstains", pd.Series(fb)), ("oracle (either right)", pd.Series(orc))):
        s = metrics(pd.DataFrame({"outcome": o, "k": 0}), ["k"]).iloc[0]
        rows.append({"model": m, "cfg": name, "f1": s.f1, "f1_ign": s.f1_ign, "recall": s.recall})
t = pd.DataFrame(rows)
O = ["A best", "B search + memory", "B, then A if B abstains", "oracle (either right)"]
OC = dict(zip(O, ["#86b6ef", "#a8380c", VIOLET, NEUTRAL]))
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, (s, n) in zip(axes, SCORES):
    grouped_bars(ax, t, "model", MODELS, "cfg", O, s, OC, names=MNAME, err=False); ax.set_title(n); ax.set_xticklabels(MN, rotation=15)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Football: A and B combined (oracle = upper bound of a perfect selector)")
save(fig, FOLDER, "ab_complementarity")
""")
    chart("same_answer", "When both answer, do they give the same value?", """
rows = []
for m in MODELS:
    a = v[(v.model == m) & (v.cfg == "A best")].set_index("task_id")
    b = v[(v.model == m) & (v.cfg == "B search + memory")].set_index("task_id")
    j = a[["answer", "outcome"]].join(b[["answer", "outcome"]], lsuffix="_a", rsuffix="_b")
    j = j[(j.outcome_a != "abstained") & (j.outcome_b != "abstained")]
    same = j.answer_a.astype(str).str.lower() == j.answer_b.astype(str).str.lower()
    rows.append({"model": MNAME[m], "both answered": len(j), "same answer": same.mean(),
                 "precision when same": (j[same].outcome_a == "correct").mean(), "precision when different (A)": (j[~same].outcome_a == "correct").mean(),
                 "precision when different (B)": (j[~same].outcome_b == "correct").mean()})
t = pd.DataFrame(rows); display(t)
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(t))
for i, (c, col) in enumerate([("precision when same", AQUA), ("precision when different (A)", "#86b6ef"), ("precision when different (B)", "#a8380c")]):
    ax.bar(x + (i - 1) * 0.27, t[c], 0.25, color=col, label=c)
ax.set_xticks(x); ax.set_xticklabels([f"{m}\\n({s:.0%} same, n = {n})" for m, s, n in zip(t.model, t["same answer"], t["both answered"])], fontsize=8)
ax.legend(fontsize=8); ax.grid(axis="x", visible=False); ax.set_ylabel("precision")
ax.set_title("Football: agreement of A best and B search + memory as a confidence signal")
save(fig, FOLDER, "ab_agreement")
""")
    md("## 3. Cost of the gain")
    chart("energy", "Score against energy and token amplification", """
eff["cfg"] = None
for m in MODELS:
    eff.loc[(eff.model == m) & (eff.strategy == "A") & (eff.approach == BEST[m]), "cfg"] = "A best"
eff.loc[(eff.strategy == "A") & (eff.approach == "zero") & eff.cfg.isna(), "cfg"] = "A zero-shot"
eff.loc[eff.approach == "search_evidence", "cfg"] = "B search"; eff.loc[eff.approach == "search_memory", "cfg"] = "B search + memory"
e = eff[eff.cfg.notna()].groupby(["model", "cfg"]).agg(EC=("EC_run", "mean"), proc=("tokens_proc", "sum"), user=("tokens_user", "sum"),
                                                       latency=("latency_s", "mean")).reset_index()
e["amp"] = e.proc / e.user
t = metrics(v[v.cfg.notna()], ["model", "cfg"]).merge(e, on=["model", "cfg"])
MK = {"A zero-shot": "s", "A best": "o", "B search": "D", "B search + memory": "^"}
for xc, xn, fn in (("EC", "energy per run (Wh, log)", "ab_energy"), ("amp", "token amplification |T_process| / |T_user| (log)", "ab_amplification")):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (s, n) in zip(axes, SCORES):
        for _, r in t.iterrows():
            ax.scatter(r[xc], r[s], marker=MK[r.cfg], s=60, color=MCOLOR[r.model], edgecolor=SURFACE, zorder=3)
        logaxis(ax, "x"); ax.set_xlabel(xn); ax.set_title(n)
    h = model_handles() + [plt.Line2D([], [], ls="", marker="", label="")] + [plt.Line2D([], [], ls="", marker=mk, color=INK2, label=c) for c, mk in MK.items()]
    axes[1].legend(handles=h, fontsize=8, loc="best")
    fig.suptitle(f"Football: score vs {xn.split(' (')[0]}")
    save(fig, FOLDER, fn)
display(e.assign(model=e.model.map(MNAME)).pivot(index="model", columns="cfg", values=["EC", "amp", "latency"]).round(4))
""")
    chart("stability", "Run agreement, A vs B", """
r = runs.copy()
r["cfg"] = r.apply(cfg, axis=1)
ag = agreement(r[r.cfg.notna()], ["model", "cfg"])
fig, ax = plt.subplots(figsize=(10, 3.8))
grouped_bars(ax, ag, "model", MODELS, "cfg", C, "same outcome in all runs", CC, names=MNAME, err=False, fmt="{:.0%}")
pct(ax); ax.set_xticklabels(MN, rotation=15); ax.legend(fontsize=8, ncol=2)
ax.set_title("Football: tasks whose 3 runs have the same outcome")
save(fig, FOLDER, "ab_stability")
""")

# ============================================================================================ both
if MODE == "both":
    md("""
    # Both strategies on both datasets: airports and football

    **Airports** (`airports-v5`, 243 context-on / 180 context-off tasks, hidden properties of `Airport` nodes) and
    **football** (`football-v1-idmix`, 239 / 233 tasks, hidden properties of `Player` nodes). Same 5 models,
    3 runs, same prompts (Strategy A: 9 prompts) and the same agent (Strategy B: search, and search + memory, run on
    the context-on sets). This report puts the two datasets side by side to separate what holds in general from
    what depends on the graph.
    """)
    md(SCORING)
    code(SETUP + """
v = load_voted(("airports", "ctx"), ("airports", "noctx"), ("football", "ctx"), ("football", "noctx"))
runs = load_runs(("airports", "ctx"), ("football", "ctx"))
eff = load_effort()
DS = ["airports", "football"]
ctx = v[v.set == "ctx"].copy()
mp = metrics(ctx[ctx.strategy == "A"], ["dataset", "model", "approach"])
BEST = mp.loc[mp.groupby(["dataset", "model"]).f1.idxmax()].set_index(["dataset", "model"]).approach.to_dict()
def cfg(r):
    if r.strategy == "A":
        return "A best" if r.approach == BEST[(r.dataset, r.model)] else ("A zero-shot" if r.approach == "zero" else None)
    return "B search" if r.approach == "search_evidence" else "B search + memory"
ctx["cfg"] = ctx.apply(cfg, axis=1)
C = ["A zero-shot", "A best", "B search", "B search + memory"]
CC = {"A zero-shot": "#cde2fb", "A best": "#86b6ef", "A mean": "#c9c8c2", "B search": "#eb6834", "B search + memory": "#a8380c"}
print(v.groupby(["dataset", "set"]).task_id.nunique().to_dict())
pd.DataFrame(BEST, index=["best A prompt (F1)"]).T.unstack(0).droplevel(0, axis=1).rename(index=MNAME)
""")
    md("## 1. Headline numbers")
    chart("headline", "Score per model, approach and dataset", """
t = pd.concat([boot(ctx[ctx.cfg.notna()], ["dataset", "model", "cfg"]), boot(ctx[ctx.strategy == "A"], ["dataset", "model"]).assign(cfg="A mean")])
order = ["A zero-shot", "A mean", "A best", "B search", "B search + memory"]
for s, n in SCORES:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.3), sharey=True)
    for ax, d in zip(axes, DS):
        grouped_bars(ax, t[t.dataset == d], "model", MODELS, "cfg", order, s, CC, names=MNAME); ax.set_title(d)
        ax.set_xticklabels(MN, rotation=10)
    axes[0].set_ylabel(n); axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle(f"{n} per model and approach, airports vs football (context on)")
    save(fig, FOLDER, f"both_headline_{s}")
tab = t.assign(model=t.model.map(MNAME)).pivot_table(index=["dataset", "model"], columns="cfg", values=["f1", "f1_ign"]).round(3)
display(tab)
""")
    chart("delta", "Football − airports, per model and approach", """
t = pd.concat([metrics(ctx[ctx.cfg.notna()], ["dataset", "model", "cfg"]), metrics(ctx[ctx.strategy == "A"], ["dataset", "model"]).assign(cfg="A mean")])
order = ["A zero-shot", "A mean", "A best", "B search", "B search + memory"]
fig, axes = plt.subplots(1, 3, figsize=(17, 3.6))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    pv = t.pivot_table(index="model", columns=["dataset", "cfg"], values=s)
    dlt = (pv["football"] - pv["airports"]).reindex(index=MODELS, columns=order)
    heat(ax, dlt, fmt="{:+.2f}", cmap=DIV, center=0, xlabels=order, ylabels=MN); ax.set_title(f"{n}: football − airports")
fig.suptitle("Change from airports to football (blue = higher on football)")
save(fig, FOLDER, "both_delta")
""")
    chart("pr", "Precision against recall: every configuration on both datasets", """
t = metrics(ctx, ["dataset", "model", "approach"])
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, d in zip(axes, DS):
    for _, r in t[t.dataset == d].iterrows():
        mk = "D" if r.approach == "search_evidence" else "^" if r.approach == "search_memory" else "o"
        ax.scatter(r.recall, r.precision, marker=mk, s=70 if mk != "o" else 26, color=MCOLOR[r.model], edgecolor=INK if mk != "o" else SURFACE,
                   linewidth=1 if mk != "o" else 0.5, alpha=1 if mk != "o" else 0.75, zorder=3 if mk != "o" else 2)
    ax.set_title(d); ax.set_xlabel("recall")
axes[0].set_ylabel("precision")
h = model_handles() + [plt.Line2D([], [], ls="", marker="", label="")] + [plt.Line2D([], [], ls="", marker=mk, color=INK2, label=l) for mk, l in
                                                                          (("o", "A (one dot per prompt)"), ("D", "B search"), ("^", "B search + memory"))]
axes[1].legend(handles=h, fontsize=8)
fig.suptitle("Precision vs recall of every configuration (context on)")
save(fig, FOLDER, "both_precision_recall")
""")
    md("## 2. Strategy A: prompts, models and context")
    chart("variance", "What matters more: the model or the prompt?", """
t = metrics(v[v.strategy == "A"], ["dataset", "set", "model", "approach"])
rows = []
for (d, s_), g in t.groupby(["dataset", "set"]):
    for sc, n in SCORES:
        pm = g.pivot(index="model", columns="approach", values=sc)
        rows.append({"dataset": d, "set": s_, "score": n, "span over models (mean over prompts)": pm.mean(axis=1).max() - pm.mean(axis=1).min(),
                     "span over prompts (mean over models)": pm.mean(axis=0).max() - pm.mean(axis=0).min()})
r = pd.DataFrame(rows); display(r)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, (sc, n) in zip(axes, SCORES):
    d = r[r.score == n]; x = np.arange(len(d))
    ax.bar(x - 0.2, d["span over models (mean over prompts)"], 0.38, color="#1c5cab", label="span over models")
    ax.bar(x + 0.2, d["span over prompts (mean over models)"], 0.38, color="#86b6ef", label="span over prompts")
    ax.set_xticks(x); ax.set_xticklabels([f"{a}\\ncontext {'on' if b == 'ctx' else 'off'}" for a, b in zip(d.dataset, d.set)]); ax.set_title(n); ax.grid(axis="x", visible=False)
axes[0].legend(fontsize=8); axes[0].set_ylabel("max − min of the mean score")
fig.suptitle("Strategy A: spread of the score over models vs over prompts")
save(fig, FOLDER, "both_model_vs_prompt")
""")
    chart("prompt_consistency", "Is the best prompt the same on both datasets?", """
t = metrics(v[(v.strategy == "A") & (v.set == "ctx")], ["dataset", "model", "approach"])
pv = t.pivot_table(index=["model", "approach"], columns="dataset", values=["f1", "f1_ign"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (s, n) in zip(axes, SCORES):
    d = pv[s].reset_index()
    for m in MODELS:
        dd = d[d.model == m]; ax.scatter(dd.airports, dd.football, color=MCOLOR[m], s=36, edgecolor=SURFACE, zorder=3)
    lim = [0, max(d.airports.max(), d.football.max()) * 1.05]; ax.plot(lim, lim, color=GRID, lw=1)
    rho = d[["airports", "football"]].rank().corr().iloc[0, 1]
    ax.set_xlabel("airports"); ax.set_ylabel("football"); ax.set_title(f"{n} (Spearman ρ = {rho:.2f})")
axes[1].legend(handles=model_handles(), fontsize=8)
fig.suptitle("Strategy A, context on: score of each model × prompt on airports vs football")
save(fig, FOLDER, "both_prompt_consistency")
rk = t.assign(rank=t.groupby(["dataset", "model"]).f1.rank(ascending=False)).pivot_table(index="approach", columns="dataset", values="rank").reindex(PROMPTS)
display(rk.rename(index=PNAME).round(1).rename(columns=lambda c: f"mean rank of the prompt, {c}"))
""")
    chart("context", "Effect of context on both datasets (task pairs whose context differs, 9 prompts pooled)", """
rows = []
for d in DS:
    a = v[(v.dataset == d) & (v.strategy == "A")]
    pairs = context_pairs(d)   # only the pairs whose context-on G has context properties
    off = a[a.set == "noctx"].assign(task_id=lambda x: x.task_id.map(pairs)).dropna(subset=["task_id"])
    on = a[(a.set == "ctx") & a.task_id.isin(pairs.values())]
    for m in MODELS:
        x = off[off.model == m].set_index(["task_id", "approach"]).outcome
        y = on[on.model == m].set_index(["task_id", "approach"]).outcome
        for s, n in SCORES:
            dd, lo, hi = paired_diff(x, y, s); rows.append({"dataset": d, "model": m, "score": n, "diff": dd, "lo": lo, "hi": hi, "task pairs": len(pairs)})
pdf = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, (s, n) in zip(axes, SCORES):
    for i, d in enumerate(DS):
        dd = pdf[(pdf.dataset == d) & (pdf.score == n)].set_index("model").reindex(MODELS)
        y = np.arange(len(MODELS)) + (i - 0.5) * 0.3
        ax.errorbar(dd["diff"], y, xerr=[dd["diff"] - dd.lo, dd.hi - dd["diff"]], fmt="o", color=DCOLOR[d], label=d, capsize=2)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MN); ax.invert_yaxis(); ax.set_title(f"{n}: context on − off"); ax.grid(axis="y", visible=False)
axes[1].legend(fontsize=8)
fig.suptitle("Strategy A: paired effect of context on the task pairs whose context differs (right of 0: context helps)")
save(fig, FOLDER, "both_context_effect")
display(pdf.assign(model=pdf.model.map(MNAME)).round(3))
""")
    md("## 3. Strategy B and A vs B")
    chart("ab_forest", "B − A on both datasets", """
rows = []
for d in DS:
    for m in MODELS:
        for a_ in ("A zero-shot", "A best"):
            for b_ in ("B search", "B search + memory"):
                x = ctx[(ctx.dataset == d) & (ctx.model == m) & (ctx.cfg == a_)].set_index("task_id").outcome
                y = ctx[(ctx.dataset == d) & (ctx.model == m) & (ctx.cfg == b_)].set_index("task_id").outcome
                for s, n in SCORES:
                    dd, lo, hi = paired_diff(x, y, s); rows.append({"dataset": d, "model": m, "pair": f"{b_} − {a_}", "score": n, "diff": dd, "lo": lo, "hi": hi, **paired(x, y)})
pdf = pd.DataFrame(rows); pairs = pdf.pair.unique().tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (s, n) in zip(axes, SCORES):
    for i, p in enumerate(pairs):
        for j, m in enumerate(MODELS):
            for k, d in enumerate(DS):
                r = pdf[(pdf.pair == p) & (pdf.model == m) & (pdf.dataset == d) & (pdf.score == n)].iloc[0]
                y = i * 12 + j * 2 + k * 0.8
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o" if d == "airports" else "s", color=MCOLOR[m],
                            mfc=MCOLOR[m] if d == "football" else "white", ms=4.5, capsize=1.5)
    ax.axvline(0, color=INK2, lw=1); ax.set_title(n); ax.set_xlabel("B − A")
    ax.set_yticks([i * 12 + 4.4 for i in range(len(pairs))]); ax.set_yticklabels(pairs); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
h = model_handles() + [plt.Line2D([], [], ls="", marker="o", mfc="white", color=INK2, label="airports"), plt.Line2D([], [], ls="", marker="s", color=INK2, label="football")]
axes[1].legend(handles=h, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.suptitle("Paired difference B − A with 95% interval, both datasets (right of 0: B better)")
save(fig, FOLDER, "both_ab_forest")
display(pdf[pdf.score == "F1"].assign(model=lambda x: x.model.map(MNAME))[["dataset", "model", "pair", "diff", "lo", "hi", "only first right", "only second right", "p (McNemar)"]]
        .rename(columns={"only first right": "only A right", "only second right": "only B right"}).round(3))
""")
    chart("memory", "What memory adds to B on both datasets", """
rows = []
for d in DS:
    for m in MODELS:
        x = ctx[(ctx.dataset == d) & (ctx.model == m) & (ctx.cfg == "B search")].set_index("task_id").outcome
        y = ctx[(ctx.dataset == d) & (ctx.model == m) & (ctx.cfg == "B search + memory")].set_index("task_id").outcome
        for s, n in SCORES + [("abstention", "abstention")]:
            if s == "abstention":
                dd = (y == "abstained").mean() - (x == "abstained").mean(); lo = hi = np.nan
            else:
                dd, lo, hi = paired_diff(x, y, s)
            rows.append({"dataset": d, "model": m, "score": n, "diff": dd, "lo": lo, "hi": hi})
pdf = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
for ax, (s, n) in zip(axes, SCORES + [("abstention", "abstention")]):
    t = pdf[pdf.score == n].rename(columns={"diff": "val", "lo": "val_lo", "hi": "val_hi"})
    grouped_bars(ax, t, "model", MODELS, "dataset", DS, "val", DCOLOR, names=MNAME, err=s != "abstention", fmt="{:+.2f}")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_title(f"{n}: memory − search"); ax.set_xticklabels(MN, rotation=15)
axes[0].legend(fontsize=8); pct(axes[2])
fig.suptitle("Strategy B: change from search to search + memory, both datasets")
save(fig, FOLDER, "both_memory_gain")
""")
    chart("retrieval", "Retrieval on both datasets", """
rows = []
for d in DS:
    det = load_b_detail(d); det = det[det.variant == "search_evidence"]
    for m in MODELS:
        x = det[det.model == m]
        rows.append({"dataset": d, "model": m, "plan names an entity": x.entity_named.mean(), "plan names the right node": x.entity_match.mean(),
                     "run answers": x.answered.mean(), "evidence truncated": x.evidence_truncated.mean()})
t = pd.DataFrame(rows); display(t.assign(model=t.model.map(MNAME)).round(3))
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
for ax, c in zip(axes, ["plan names an entity", "plan names the right node", "run answers"]):
    grouped_bars(ax, t, "model", MODELS, "dataset", DS, c, DCOLOR, names=MNAME, err=False, fmt="{:.0%}")
    ax.set_title(c); pct(ax); ax.set_xticklabels(MN, rotation=15)
axes[0].legend(fontsize=8)
fig.suptitle("Strategy B search: entity identification and answering (single runs)")
save(fig, FOLDER, "both_retrieval")
""")
    chart("domains", "Concentration of the evidence sources", """
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, d in zip(axes, DS):
    dom = load_domains(d); dd = dom[dom.variant == "search_evidence"].groupby("domain")[["passed", "cited"]].sum().sort_values("passed", ascending=False).head(10)
    y = np.arange(len(dd))
    ax.barh(y, dd.passed, color=DCOLOR[d], label="in the evidence"); ax.barh(y, dd.cited, color=INK2, label="cited")
    ax.set_yticks(y); ax.set_yticklabels(dd.index); ax.invert_yaxis(); ax.set_title(d); ax.grid(axis="y", visible=False); ax.set_xlabel("runs")
axes[0].legend(fontsize=8)
fig.suptitle("Strategy B search: the 10 most frequent evidence domains")
save(fig, FOLDER, "both_domains")
""")
    md("## 4. Task factors on both datasets")
    chart("strata", "Score per stratum, A best and B + memory", """
for s, n in SCORES:
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    for ax, (f, fn) in zip(axes, FACTORS.items()):
        for d, ls in zip(DS, ("-", "--")):
            for c_ in ("A best", "B search + memory"):
                t = metrics(ctx[(ctx.dataset == d) & (ctx.cfg == c_)], [f]).set_index(f).reindex(STRATA)
                ax.plot(STRATA, t[s], ls=ls, marker="o", lw=2, color=CC[c_], label=f"{d}, {c_}")
        ax.set_title(fn); ax.set_xlabel("stratum")
    axes[0].set_ylabel(f"{n} (5 models pooled)"); axes[-1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"{n} per stratum on both datasets")
    save(fig, FOLDER, f"both_strata_{s}")
""")
    chart("difficulty", "Score against difficulty (within-dataset tertiles)", """
d = ctx[ctx.cfg.notna()].copy()
d["bin"] = d.groupby("dataset").difficulty.transform(lambda x: pd.qcut(x, 3, labels=["easy third", "middle third", "hard third"]).astype(str))
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
B3 = ["easy third", "middle third", "hard third"]
for ax, (s, n) in zip(axes, SCORES):
    for ds, ls in zip(DS, ("-", "--")):
        for c_ in C:
            t = metrics(d[(d.dataset == ds) & (d.cfg == c_)], ["bin"]).set_index("bin").reindex(B3)
            ax.plot(B3, t[s], ls=ls, marker="o", color=CC[c_], lw=2, label=f"{ds}, {c_}")
    ax.set_title(n)
axes[1].legend(fontsize=7, ncol=2)
fig.suptitle("Score vs difficulty tertile (solid: airports, dashed: football; 5 models pooled)")
save(fig, FOLDER, "both_difficulty")
""")
    chart("popularity", "Score against pop(p) of the hidden property", """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, c_ in zip(axes, ["A best", "B search + memory"]):
    for d in DS:
        pop = popularity(d).set_index("property").popularity
        t = metrics(ctx[(ctx.dataset == d) & (ctx.cfg == c_)], ["property"]).assign(pop=lambda x: x.property.map(pop))
        ax.scatter(t["pop"], t.f1_ign, s=10 + 2 * t.tasks / 5, color=DCOLOR[d], edgecolor=INK2, linewidth=0.4, alpha=0.85, label=d)
    ax.set_xlabel("pop(p)"); ax.set_title(c_)
axes[0].set_ylabel("F1 (abstentions ignored), per property; size = tasks"); axes[0].legend(fontsize=8)
fig.suptitle("Score of each hidden property vs its popularity (5 models pooled)")
save(fig, FOLDER, "both_popularity")
""")
    md("## 5. Effort on both datasets")
    chart("energy", "Energy per run and score", """
e = eff[eff.set == "ctx"].copy()
e["cfg"] = None
for (d, m), p in BEST.items():
    e.loc[(e.dataset == d) & (e.model == m) & (e.strategy == "A") & (e.approach == p), "cfg"] = "A best"
e.loc[(e.strategy == "A") & (e.approach == "zero") & e.cfg.isna(), "cfg"] = "A zero-shot"
e.loc[e.approach == "search_evidence", "cfg"] = "B search"; e.loc[e.approach == "search_memory", "cfg"] = "B search + memory"
g = e[e.cfg.notna()].groupby(["dataset", "model", "cfg"]).agg(EC=("EC_run", "mean"), proc=("tokens_proc", "sum"), user=("tokens_user", "sum"), latency=("latency_s", "mean")).reset_index()
g["amp"] = g.proc / g.user
t = metrics(ctx[ctx.cfg.notna()], ["dataset", "model", "cfg"]).merge(g, on=["dataset", "model", "cfg"])
MK = {"A zero-shot": "s", "A best": "o", "B search": "D", "B search + memory": "^"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, d in zip(axes, DS):
    for _, r in t[t.dataset == d].iterrows():
        ax.scatter(r.EC, r.f1, marker=MK[r.cfg], s=60, color=MCOLOR[r.model], edgecolor=SURFACE, zorder=3)
    logaxis(ax, "x"); ax.set_xlabel("energy per run (Wh, log)"); ax.set_title(d)
axes[0].set_ylabel("F1")
h = model_handles() + [plt.Line2D([], [], ls="", marker="", label="")] + [plt.Line2D([], [], ls="", marker=mk, color=INK2, label=c) for c, mk in MK.items()]
axes[1].legend(handles=h, fontsize=8)
fig.suptitle("F1 vs energy per run, both datasets")
save(fig, FOLDER, "both_energy_f1")
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, (c, n) in zip(axes, [("EC", "energy per run (Wh, log)"), ("amp", "token amplification (log)")]):
    pv = t.pivot_table(index="cfg", columns="dataset", values=c, aggfunc="median").reindex(C)
    x = np.arange(len(C))
    for i, d in enumerate(DS):
        ax.bar(x + (i - 0.5) * 0.38, pv[d], 0.36, color=DCOLOR[d], label=d)
    logaxis(ax, "y"); ax.set_xticks(x); ax.set_xticklabels(C); ax.set_title(n + ", median over models"); ax.grid(axis="x", visible=False)
axes[0].legend(fontsize=8)
fig.suptitle("Effort per approach on both datasets")
save(fig, FOLDER, "both_effort")
display(t.assign(model=t.model.map(MNAME)).pivot_table(index=["dataset", "model"], columns="cfg", values=["EC", "amp"]).round(4))
""")
    chart("stability", "Run agreement on both datasets", """
r = runs.copy()
r["cfg"] = r.apply(cfg, axis=1)
ag = agreement(r[r.cfg.notna()], ["dataset", "cfg", "model"]).groupby(["dataset", "cfg"])["same outcome in all runs"].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 3.6))
grouped_bars(ax, ag, "cfg", C, "dataset", DS, "same outcome in all runs", DCOLOR, err=False, fmt="{:.0%}")
pct(ax); ax.legend(fontsize=8); ax.set_title("Tasks whose 3 runs have the same outcome (mean over the 5 models)")
display(ag.round(3))
save(fig, FOLDER, "both_stability")
""")

md("## Takeaways")
discussion("takeaways")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / f"{OUTNAME}.ipynb"
nbf.write(nb, out)
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")

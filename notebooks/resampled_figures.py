"""Figures and tables of the airports evaluation on the 108 resampled tasks (data/new-diff-airports_ressampled.csv).

    venv/bin/python notebooks/resampled_figures.py

Every source is filtered to the 108 task ids: the voted answers (results/airports/v5_ctx/scored/scored_voted.csv,
Strategy A with 9 prompts, Strategy B search and search + memory, 5 models, 3 runs) and the per-run effort and energy
(results/energy_effort/runs.csv). Writes PNG and PDF figures, CSV tables and facts.json to
notebooks/figures/paper_reports/airports_resampled/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import paper_report_lib as L  # noqa: E402

ROOT = L.ROOT
FOLDER = "airports_resampled"
OUT = L.figdir(FOLDER)
(OUT / "tables").mkdir(exist_ok=True)
SEL = pd.read_csv(ROOT / "data/new-diff-airports_ressampled.csv").rename(columns={"taskID": "task_id"})
IDS = set(SEL.task_id)
DIFF = SEL.set_index("task_id").newDiffMeasure
MODELS, MN = L.MODELS, L.MNAME
APP = ["A mean", "A best", "B search", "B search + memory"]
ACOL = {"A mean": "#86b6ef", "A best": "#1c5cab", "B search": "#eb6834", "B search + memory": "#f4a582"}
FACTS = {}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def table(df, name):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)


# ------------------------------------------------------------------------------------------------ data
v = pd.read_csv(ROOT / "results/airports/v5_ctx/scored/scored_voted.csv", low_memory=False)
v = v[v.task_id.isin(IDS) & v.model.isin(MODELS)].copy()
v["approach"] = np.where(v.strategy == "A", "A", np.where(v.variant == "search_memory", "B search + memory", "B search"))
v["diff"] = v.task_id.map(DIFF)
assert v.task_id.nunique() == 108, v.task_id.nunique()
A = v[v.approach == "A"]

# per model and prompt (Strategy A)
ap = L.metrics(A, ["model", "variant"])
best = ap.sort_values("f1", ascending=False).groupby("model").head(1).set_index("model").variant
FACTS["best_prompt"] = {MN[m]: L.PNAME[best[m]] for m in MODELS}

# the four approaches per model; A mean = mean of the 9 per-prompt values
rows = []
for m in MODELS:
    pm = ap[ap.model == m]
    rows.append({"model": m, "approach": "A mean", **pm[["f1", "precision", "f1_ign", "recall", "abstention", "error"]].mean().to_dict(),
                 "tasks": pm.tasks.iloc[0]})
    rows.append({"model": m, "approach": "A best", **pm[pm.variant == best[m]].iloc[0].drop(["model", "variant"]).to_dict()})
    for b in ["B search", "B search + memory"]:
        rows.append({"model": m, "approach": b, **L.metrics(v[(v.model == m) & (v.approach == b)], ["model"]).iloc[0].drop("model").to_dict()})
sc = pd.DataFrame(rows)
FACTS["scores"] = {f"{MN[r.model]} | {r.approach}": {k: round(float(getattr(r, k)), 3) for k in ["f1", "precision", "f1_ign", "abstention", "error"]}
                   for r in sc.itertuples()}

# best A config per model (for "A best" in other figures)
def a_best_rows(df):
    return df[(df.approach == "A") & (df.variant == df.model.map(best))].assign(approach="A best")


core = pd.concat([a_best_rows(v), v[v.approach != "A"]])   # A best, B search, B search + memory

# effort and energy per run
e = L.load_effort()
e = e[(e.dataset == "airports") & (e.set == "ctx") & e.task_id.isin(IDS) & e.model.isin(MODELS)].copy()
e["approach"] = np.where(e.strategy == "A", "A", np.where(e.variant == "search_memory", "B search + memory", "B search"))
e["config"] = np.where(e.approach == "A", e.variant, e.approach)
e["amp"] = e.tokens_proc / e.tokens_user   # token amplification |T_process| / |T_user| (Section 4.3.2), per run
e["tokens_out_visible"] = e.tokens_out - e.tokens_reasoning.fillna(0)

# Energy per run (Section 4.3.3): EcoLogits energy of the run's LLM calls + 0.3 Wh per web search. Energy only, in Wh:
# no GWP / water and no normalisation (GWP and water rank the configurations like energy; see the page's definitions).
cfg = e.groupby(["model", "approach", "config"]).agg(EC=("EC_run", "mean"),
                                                     tok_in=("tokens_in", "mean"), tok_out=("tokens_out", "mean"),
                                                     tok_reason=("tokens_reasoning", "mean"), evidence=("evidence_tokens", "mean"),
                                                     searches=("searches", "mean"), runs=("task_id", "size"),
                                                     amp=("amp", "mean"), t_user=("tokens_user", "mean"), t_proc=("tokens_proc", "mean")).reset_index()
acc_cfg = L.metrics(v.assign(config=np.where(v.approach == "A", v.variant, v.approach)), ["model", "config"])
cfg = cfg.merge(acc_cfg[["model", "config", "f1", "precision", "f1_ign", "abstention", "correct"]], on=["model", "config"])
cfg["Wh_per_correct"] = cfg.EC * cfg.runs / cfg.correct     # all runs of the configuration / correct voted answers


def per_app(col, how="mean"):
    out = []
    for m in MODELS:
        c = cfg[cfg.model == m]
        out.append({"model": m, "approach": "A mean", col: c[c.approach == "A"][col].mean()})
        out.append({"model": m, "approach": "A best", col: c[c.config == best[m]][col].iloc[0]})
        for b in ["B search", "B search + memory"]:
            out.append({"model": m, "approach": b, col: c[c.config == b][col].iloc[0]})
    return pd.DataFrame(out)


for col in ["EC", "Wh_per_correct", "amp", "t_user", "t_proc", "tok_in", "tok_out", "tok_reason", "evidence"]:
    sc = sc.merge(per_app(col), on=["model", "approach"])
sc.insert(1, "model name", sc.model.map(MN))
table(sc.drop(columns="model"), "scores_by_approach")

# ------------------------------------------------------------------------------------------------ tables
t1 = ap.pivot(index="variant", columns="model", values="f1").reindex(index=L.PROMPTS, columns=MODELS)
t1p = ap.pivot(index="variant", columns="model", values="precision").reindex(index=L.PROMPTS, columns=MODELS)
t1a = ap.pivot(index="variant", columns="model", values="abstention").reindex(index=L.PROMPTS, columns=MODELS)
table(ap.assign(model=ap.model.map(MN), prompt=ap.variant.map(L.PNAME)).drop(columns="variant"), "strategy_a_per_prompt")
json.dump({"f1": t1.rename(index=L.PNAME, columns=MN).round(3).to_dict(),
           "precision": t1p.rename(index=L.PNAME, columns=MN).round(3).to_dict(),
           "abstention": t1a.rename(index=L.PNAME, columns=MN).round(3).to_dict()},
          open(OUT / "tables/strategy_a_per_prompt.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ F1 accuracy
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, (col, title) in zip(axes, [("f1", "F1"), ("precision", "Precision"), ("f1_ign", "F1 (abstentions ignored)")]):
    L.grouped_bars(ax, sc, "model", MODELS, "approach", APP, col, ACOL, names=MN, err=False)
    ax.set_title(title); ax.set_ylim(0, 0.8)
    ax.set_xticklabels([MN[m] for m in MODELS], rotation=20, ha="right")
    ax.legend(handles=[Patch(color=ACOL[a], label=a) for a in APP], fontsize=7.5, loc="upper left", ncol=2)
save(fig, "accuracy_by_approach")

# ------------------------------------------------------------------------------------------------ F2 outcomes and rates
fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
for ax, a in zip(axes, APP[1:] + ["A mean"]):
    if a == "A mean":
        d = L.metrics(A, ["model"])
        ax.set_title("A, all 9 prompts")
    else:
        d = L.metrics(core[core.approach == a], ["model"])
        ax.set_title(a)
    L.stacked_outcomes(ax, d, "model", MODELS, names=MN)
axes[0].legend(handles=[Patch(color=L.OCOLOR[o], label=o) for o in ("correct", "wrong", "abstained")],
               ncol=3, loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=8)
save(fig, "outcomes_by_approach")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for ax, (col, title) in zip(axes, [("f1", "F1"), ("precision", "Precision"), ("abstention", "Abstention rate")]):
    L.grouped_bars(ax, sc, "model", MODELS, "approach", APP, col, ACOL, names=MN, err=False)
    ax.set_title(title)
    ax.set_ylim(0, 1 if col == "abstention" else 0.8)
    if col == "abstention":
        L.pct(ax)
    ax.set_xticklabels([MN[m] for m in MODELS], rotation=20, ha="right")
    ax.legend(handles=[Patch(color=ACOL[a], label=a) for a in APP], fontsize=7.5, loc="upper left", ncol=2)
save(fig, "f1_precision_abstention")

# ------------------------------------------------------------------------------------------------ F3 energy
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1, 1, 1]})
for ax, col, title in [(axes[0], "EC", "Energy per run (Wh, log scale)"),
                       (axes[1], "Wh_per_correct", "Energy per correct answer (Wh, log scale)")]:
    L.grouped_bars(ax, sc, "model", MODELS, "approach", APP, col, ACOL, names=MN, err=False, labels=False)
    ax.set_yscale("log"); ax.set_title(title)
    ax.set_xticklabels([MN[m] for m in MODELS], rotation=20, ha="right")
    ax.margins(y=0.25)
    ax.legend(handles=[Patch(color=ACOL[a], label=a) for a in APP], fontsize=7.5, loc="upper left", ncol=2)
ax = axes[2]
for m in MODELS:
    c = cfg[cfg.model == m]
    a_ = c[c.approach == "A"]
    ax.scatter(a_.EC, a_.f1, s=22, color=L.MCOLOR[m], alpha=0.55, edgecolor="none")
    for b, mk in [("B search", "^"), ("B search + memory", "s")]:
        r = c[c.config == b]
        ax.scatter(r.EC, r.f1, s=70, marker=mk, color=L.MCOLOR[m], edgecolor=L.SURFACE, linewidth=1)
ax.set_xscale("log"); ax.set_xlabel("energy per run (Wh, log scale)"); ax.set_ylabel("F1")
ax.set_title("F1 against energy, every configuration")
ax.legend(handles=L.model_handles() + [Line2D([], [], ls="", marker="o", color=L.INK2, ms=5, label="A (a prompt)"),
                                        Line2D([], [], ls="", marker="^", color=L.INK2, ms=7, label="B search"),
                                        Line2D([], [], ls="", marker="s", color=L.INK2, ms=6, label="B search + memory")],
          fontsize=7.5, loc="upper left", ncol=2)
save(fig, "energy")

# ------------------------------------------------------------------------------------------------ F4 tokens
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
x = np.arange(len(MODELS)); w = 0.2
for i, a in enumerate(APP):
    d = sc[sc.approach == a].set_index("model").reindex(MODELS)
    pos = x + (i - 1.5) * w
    prompt_in = d.tok_in - d.evidence.fillna(0)
    stacks = [(prompt_in, 1.0, "input (prompt)"), (d.evidence.fillna(0), 0.65, "input (search evidence)"),
              (d.tok_out - d.tok_reason.fillna(0), 0.4, "output"), (d.tok_reason.fillna(0), 0.2, "reasoning")]
    bottom = np.zeros(len(MODELS))
    for val, alpha, lab in stacks:
        ax.bar(pos, val, w - 0.02, bottom=bottom, color=ACOL[a], alpha=alpha, edgecolor=L.SURFACE, linewidth=0.5)
        bottom = bottom + val.to_numpy()
    for p_, t_ in zip(pos, bottom):
        ax.text(p_, t_ + 60, f"{t_:,.0f}", ha="center", va="bottom", fontsize=6.3, rotation=90, color=L.INK2)
ax.set_xticks(x, [MN[m] for m in MODELS], rotation=20, ha="right"); ax.grid(axis="x", visible=False)
ax.set_ylabel("tokens per run"); ax.set_title("Tokens per run (input, evidence, output, reasoning)"); ax.margins(y=0.15)
ax.legend(handles=[Patch(color=ACOL[a], label=a) for a in APP] +
          [Patch(color=L.INK2, alpha=al, label=lb) for al, lb in [(1, "input (prompt)"), (0.65, "input (search evidence)"),
                                                                 (0.4, "output"), (0.2, "reasoning")]],
          fontsize=7, loc="upper left", ncol=2)
ax = axes[1]
for m in MODELS:
    c = cfg[cfg.model == m]
    a_ = c[c.approach == "A"]
    ax.scatter(a_.tok_in + a_.tok_out, a_.f1, s=22, color=L.MCOLOR[m], alpha=0.55, edgecolor="none")
    for b, mk in [("B search", "^"), ("B search + memory", "s")]:
        r = c[c.config == b]
        ax.scatter(r.tok_in + r.tok_out, r.f1, s=70, marker=mk, color=L.MCOLOR[m], edgecolor=L.SURFACE, linewidth=1)
ax.set_xscale("log"); ax.set_xlabel("tokens per run (input + output, log scale)"); ax.set_ylabel("F1")
ax.set_title("F1 against tokens, every configuration")
ax.legend(handles=L.model_handles() + [Line2D([], [], ls="", marker="o", color=L.INK2, ms=5, label="A (a prompt)"),
                                        Line2D([], [], ls="", marker="^", color=L.INK2, ms=7, label="B search"),
                                        Line2D([], [], ls="", marker="s", color=L.INK2, ms=6, label="B search + memory")],
          fontsize=7.5, loc="upper left", ncol=2)
save(fig, "tokens")

# ------------------------------------------------------------------------------------------------ F4b token amplification
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axes[0]
L.grouped_bars(ax, sc, "model", MODELS, "approach", APP, "amp", ACOL, names=MN, err=False)
ax.axhline(1, color=L.INK2, lw=0.8, ls="--"); ax.set_ylim(0, 8.5)
ax.set_ylabel("|T_process| / |T_user|"); ax.set_title("Token amplification per model and strategy")
ax.set_xticklabels([MN[m] for m in MODELS], rotation=20, ha="right")
ax.legend(handles=[Patch(color=ACOL[a], label=a) for a in APP], fontsize=7.5, loc="upper left", ncol=2)
ax = axes[1]
ca = cfg[cfg.approach == "A"]
pa = ca.groupby("config").agg(amp=("amp", "mean"), t_user=("t_user", "mean")).reindex(L.PROMPTS)
pa = pa.sort_values("t_user")
ax.bar(range(len(pa)), pa.amp, color=ACOL["A mean"], width=0.7)
for i, (a_, u_) in enumerate(zip(pa.amp, pa.t_user)):
    ax.text(i, a_ + 0.04, f"{a_:.2f}", ha="center", fontsize=7.5, color=L.INK2)
ax.set_xticks(range(len(pa)), [f"{L.PNAME[p]}\n({u:.0f})" for p, u in zip(pa.index, pa.t_user)], rotation=40, ha="right", fontsize=7.5)
ax.axhline(1, color=L.INK2, lw=0.8, ls="--"); ax.grid(axis="x", visible=False)
ax.set_title("Strategy A per prompt, mean of the 5 models\n(tokens written per task in brackets)", fontsize=10)
ax = axes[2]
for m in MODELS:
    c = cfg[cfg.model == m]
    a_ = c[c.approach == "A"]
    ax.scatter(a_.amp, a_.f1, s=22, color=L.MCOLOR[m], alpha=0.55, edgecolor="none")
    for b, mk in [("B search", "^"), ("B search + memory", "s")]:
        r = c[c.config == b]
        ax.scatter(r.amp, r.f1, s=70, marker=mk, color=L.MCOLOR[m], edgecolor=L.SURFACE, linewidth=1)
ax.set_xlabel("token amplification |T_process| / |T_user|"); ax.set_ylabel("F1")
ax.set_title("F1 against token amplification, every configuration")
ax.legend(handles=L.model_handles() + [Line2D([], [], ls="", marker="o", color=L.INK2, ms=5, label="A (a prompt)"),
                                        Line2D([], [], ls="", marker="^", color=L.INK2, ms=7, label="B search"),
                                        Line2D([], [], ls="", marker="s", color=L.INK2, ms=6, label="B search + memory")],
          fontsize=7.5, loc="center", ncol=2)
save(fig, "token_amplification")
FACTS["amp"] = {f"{MN[r.model]} | {r.approach}": round(float(r.amp), 2) for r in sc.itertuples()}
FACTS["amp_per_prompt"] = {L.PNAME[p]: round(float(x), 2) for p, x in pa.amp.items()}
FACTS["amp_A_prompt_range_per_model"] = {MN[m]: [round(float(ca[ca.model == m].amp.min()), 2), round(float(ca[ca.model == m].amp.max()), 2)] for m in MODELS}
FACTS["spearman_amp_f1_A"] = round(float(ca[["amp", "f1"]].corr(method="spearman").iloc[0, 1]), 2)
FACTS["spearman_amp_f1_A_within"] = {MN[m]: round(float(ca[ca.model == m][["amp", "f1"]].corr(method="spearman").iloc[0, 1]), 2) for m in MODELS}

# ------------------------------------------------------------------------------------------------ F5 difficulty
q = DIFF.quantile([1 / 3, 2 / 3]).to_numpy()
BIN = ["easy", "medium", "hard"]
v["dbin"] = pd.cut(v["diff"], [-np.inf, q[0], q[1], np.inf], labels=BIN)
core["dbin"] = core.task_id.map(v.drop_duplicates("task_id").set_index("task_id").dbin)
FACTS["pooled_by_bin"] = {a: L.metrics(core[core.approach == a], ["dbin"]).set_index("dbin").reindex(BIN)[["f1", "precision", "abstention"]].round(3).to_dict()
                          for a in ["A best", "B search", "B search + memory"]}
FACTS["difficulty_cuts"] = [round(float(x), 2) for x in q]
FACTS["tasks_per_bin"] = v.drop_duplicates("task_id").dbin.value_counts().reindex(BIN).astype(int).to_dict()

fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, sharey="row")
for j, a in enumerate(["A best", "B search", "B search + memory"]):
    for i, col in enumerate(["f1", "precision"]):
        ax = axes[i, j]
        d = L.metrics(core[core.approach == a], ["model", "dbin"])
        for m in MODELS:
            dm = d[d.model == m].set_index("dbin").reindex(BIN)
            ax.plot(range(3), dm[col], marker="o", color=L.MCOLOR[m], lw=1.8, label=MN[m])
        pooled = L.metrics(core[core.approach == a], ["dbin"]).set_index("dbin").reindex(BIN)
        ax.plot(range(3), pooled[col], color=L.INK, lw=2.6, ls="--", label="all models")
        for k, val in enumerate(pooled[col]):
            ax.text(k, val + 0.02, f"{val:.2f}", ha="center", fontsize=7.5, color=L.INK)
        ax.set_title(f"{a}: {'F1' if col == 'f1' else 'precision'}")
        ax.set_xticks(range(3), [f"{b}\n(≤{q[0]:.1f})" if b == "easy" else (f"{b}\n({q[0]:.1f}–{q[1]:.1f})" if b == "medium" else f"{b}\n(>{q[1]:.1f})") for b in BIN])
        ax.set_ylim(0, 1.0 if col == "precision" else 0.85)
        ax.legend(fontsize=7, loc="upper right")
fig.supxlabel("global difficulty (newDiffMeasure), thirds of the 108 tasks", fontsize=9)
save(fig, "accuracy_vs_difficulty_bins")

# per task: share of the 15 core answers that are correct, against the difficulty
pt = core.groupby("task_id").agg(correct=("outcome", lambda s: (s == "correct").mean()),
                                 abstained=("outcome", lambda s: (s == "abstained").mean())).join(DIFF)
pt = pt.join(SEL.set_index("task_id").isUnique)
rho = pt[["newDiffMeasure", "correct"]].corr(method="spearman").iloc[0, 1]
rho_by = {a: core[core.approach == a].groupby("task_id").outcome.apply(lambda s: (s == "correct").mean()).to_frame("c").join(DIFF)
          .corr(method="spearman").iloc[0, 1] for a in ["A best", "B search", "B search + memory"]}
FACTS["spearman_difficulty_vs_share_correct"] = round(float(rho), 3)
FACTS["spearman_by_approach"] = {k: round(float(x), 3) for k, x in rho_by.items()}
fig, ax = plt.subplots(figsize=(8.5, 4.6))
jit = np.random.default_rng(0).uniform(-0.012, 0.012, len(pt))
ax.scatter(pt.newDiffMeasure, pt.correct + jit, s=28, color=L.BLUE, alpha=0.65, edgecolor="none", label="one task")
bins = np.linspace(pt.newDiffMeasure.min(), pt.newDiffMeasure.max(), 7)
mid = 0.5 * (bins[1:] + bins[:-1]); mean = pt.groupby(pd.cut(pt.newDiffMeasure, bins, include_lowest=True), observed=False).correct.mean()
ax.plot(mid, mean.to_numpy(), color=L.INK, lw=2.2, marker="o", label="mean per difficulty bin")
L.pct(ax); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("global difficulty of the task (newDiffMeasure)"); ax.set_ylabel("share of the 15 answers that are correct")
ax.set_title(f"Each task: how many of the 15 model × approach answers are correct (Spearman ρ = {rho:.2f})")
ax.legend(fontsize=8, loc="upper right")
save(fig, "accuracy_vs_global_difficulty")

# the components of the original design, for reference
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
for ax, (fac, lab) in zip(axes, L.FACTORS.items()):
    for a in ["A best", "B search", "B search + memory"]:
        d = L.metrics(core[core.approach == a], [fac]).set_index(fac).reindex(L.STRATA)
        ax.plot(range(3), d.f1, marker="o", color=ACOL[a], lw=2, label=a)
    ax.set_xticks(range(3), L.STRATA); ax.set_title(lab, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("F1 (5 models pooled)")
save(fig, "accuracy_vs_strata")

# ------------------------------------------------------------------------------------------------ F6 where models are right
order = DIFF.sort_values().index.tolist()
rowsdef = [(m, a) for a in ["A best", "B search", "B search + memory"] for m in MODELS]
code = {"correct": 2, "wrong": 1, "abstained": 0}
M = np.full((len(rowsdef), len(order)), np.nan)
for i, (m, a) in enumerate(rowsdef):
    s = core[(core.model == m) & (core.approach == a)].set_index("task_id").outcome
    M[i] = [code.get(s.get(t), np.nan) for t in order]
fig = plt.figure(figsize=(16, 7.4))
gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 6, 1.6], hspace=0.08)
axd = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axd); axs = fig.add_subplot(gs[2], sharex=axd)
axd.imshow(DIFF.loc[order].to_numpy()[None, :], aspect="auto", cmap=L.SEQ)
axd.set_yticks([0], ["difficulty"]); axd.tick_params(labelbottom=False); axd.grid(False)
cmap = ListedColormap([L.OCOLOR["abstained"], L.OCOLOR["wrong"], L.OCOLOR["correct"]])
ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
ax.set_yticks(range(len(rowsdef)), [f"{MN[m]} · {a}" for m, a in rowsdef], fontsize=7.5)
for k in (5, 10):
    ax.axhline(k - 0.5, color=L.SURFACE, lw=3)
ax.tick_params(labelbottom=False); ax.grid(False)
nc = (M == 2).sum(axis=0)
axs.bar(range(len(order)), nc, color=L.AQUA, width=0.9)
axs.set_ylabel("correct\n(of 15)", fontsize=8); axs.set_xlabel("the 108 tasks, ordered from easiest to hardest (newDiffMeasure)")
axs.set_xticks([]); axs.set_ylim(0, 15)
fig.legend(handles=[Patch(color=L.OCOLOR[o], label=o) for o in ("correct", "wrong", "abstained")], ncol=3,
           loc="upper right", bbox_to_anchor=(0.99, 0.995), fontsize=8.5)
fig.suptitle("Where each model is right: every task × model × approach", x=0.01, ha="left")
fig.savefig(OUT / "where_models_are_right.png", bbox_inches="tight", dpi=200)
fig.savefig(OUT / "where_models_are_right.pdf", bbox_inches="tight")
plt.close(fig)
FACTS["tasks_no_one_right"] = int((nc == 0).sum())
FACTS["tasks_all_15_right"] = int((nc == 15).sum())
FACTS["tasks_right_by_A_best_any"] = int((M[:5] == 2).any(axis=0).sum())
FACTS["tasks_right_by_B_search_any"] = int((M[5:10] == 2).any(axis=0).sum())
FACTS["tasks_right_by_any"] = int((M == 2).any(axis=0).sum())
FACTS["tasks_right_only_with_B"] = int(((M[5:] == 2).any(axis=0) & ~(M[:5] == 2).any(axis=0)).sum())
FACTS["tasks_right_only_with_A"] = int(((M[:5] == 2).any(axis=0) & ~(M[5:] == 2).any(axis=0)).sum())

# per property: who finds what
pp = core.assign(c=core.outcome.eq("correct")).pivot_table(index="property", columns=["approach"], values="c", aggfunc="mean")
pp = pp[["A best", "B search", "B search + memory"]]
pp.insert(0, "tasks", core[(core.model == MODELS[0]) & (core.approach == "A best")].property.value_counts())
pp = pp.sort_values("tasks", ascending=False)
table(pp.reset_index(), "recall_by_property")
fig, ax = plt.subplots(figsize=(9, 0.38 * len(pp) + 1.4))
L.heat(ax, pp[["A best", "B search", "B search + memory"]], fmt="{:.0%}", vmin=0, vmax=1,
       ylabels=[f"{p}  ({int(n)})" for p, n in zip(pp.index, pp.tasks)], cbar=False)
ax.set_xticklabels(["A best", "B search", "B search + memory"], rotation=0, ha="center")
ax.set_title("Share found per hidden property (5 models pooled; tasks per property in brackets)")
save(fig, "recall_by_property")

# per prompt heatmap (Strategy A)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
for ax, (tab, title, fmt) in zip(axes, [(t1, "F1", "{:.2f}"), (t1p, "Precision", "{:.2f}"), (t1a, "Abstention", "{:.0%}")]):
    L.heat(ax, tab.T, fmt=fmt, vmin=0, vmax=1 if title == "Abstention" else 0.6, cbar=False,
           xlabels=[L.PNAME[p] for p in tab.index], ylabels=[MN[m] for m in tab.columns])
    for i, m in enumerate(tab.columns):
        j = list(tab.index).index(best[m])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=L.INK, lw=2))
    ax.set_title(f"Strategy A: {title} (box = best prompt by F1)")
save(fig, "strategy_a_prompts")

# ------------------------------------------------------------------------------------------------ facts
FACTS["n_tasks"] = 108
FACTS["properties"] = core[(core.model == MODELS[0]) & (core.approach == "A best")].property.value_counts().to_dict()
FACTS["pooled"] = {a: {k: round(float(x), 3) for k, x in L.metrics(core[core.approach == a] if a != "A all" else A, ["approach"] if a != "A all" else ["strategy"])
                       .iloc[0][["f1", "precision", "f1_ign", "abstention", "error"]].items()} for a in ["A all", "A best", "B search", "B search + memory"]}
FACTS["energy_range_wh"] = {"A": [round(float(cfg[cfg.approach == "A"].EC.min()), 5), round(float(cfg[cfg.approach == "A"].EC.max()), 4)],
                            "B": [round(float(cfg[cfg.approach != "A"].EC.min()), 3), round(float(cfg[cfg.approach != "A"].EC.max()), 3)]}
FACTS["wh_per_correct"] = {f"{MN[r.model]} | {r.approach}": round(float(r.Wh_per_correct), 4) for r in sc.itertuples()}
FACTS["wh_per_correct_range"] = {"A": [round(float(cfg[cfg.approach == "A"].Wh_per_correct.min()), 4), round(float(cfg[cfg.approach == "A"].Wh_per_correct.max()), 3)],
                                 "B": [round(float(cfg[cfg.approach != "A"].Wh_per_correct.min()), 2), round(float(cfg[cfg.approach != "A"].Wh_per_correct.max()), 2)]}
FACTS["spearman_energy_f1_A"] = round(float(cfg[cfg.approach == "A"][["EC", "f1"]].corr(method="spearman").iloc[0, 1]), 2)
FACTS["energy_wh"] = {f"{MN[r.model]} | {r.approach}": round(float(r.EC), 5) for r in sc.itertuples()}
FACTS["tokens"] = {f"{MN[r.model]} | {r.approach}": round(float(r.tok_in + r.tok_out)) for r in sc.itertuples()}
json.dump(FACTS, open(OUT / "facts.json", "w"), indent=1, default=float)
print(json.dumps(FACTS, indent=1, default=float)[:6000])
print(sorted(p.name for p in OUT.iterdir()))

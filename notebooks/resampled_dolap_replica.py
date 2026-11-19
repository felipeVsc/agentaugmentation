"""The figures and tables of the DOLAP 2026 paper, redrawn with the new data: the 108 resampled airport tasks
(data/new-diff-airports_ressampled.csv), the 5 current models, the 9 prompts of Strategy A, our scoring.

    venv/bin/python notebooks/resampled_dolap_replica.py

DOLAP Fig. 1 (prompt tokens as Table R0; energy by model), Fig. 2 (energy by prompt and model), Table 2 (example tasks and difficulty),
Table 3 (best accuracy per model), Fig. 3 (accuracy vs difficulty, context vs no context), Fig. 4 (accuracy vs effort).
Context on = the 108 tasks (airports v5_ctx); context off = their counterpart in airports v5_noctx, which exists for 77
of them (36 identical tasks, 41 with the airport's own properties instead of context). Context comparisons use these 77.
Writes dolap_*.png/.pdf, tables/dolap_*.csv and dolap_facts.json to notebooks/figures/paper_reports/airports_resampled/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paper_report_lib as L  # noqa: E402

ROOT = L.ROOT
OUT = L.figdir("airports_resampled")
SEL = pd.read_csv(ROOT / "data/new-diff-airports_ressampled.csv").rename(columns={"taskID": "task_id"})
IDS = set(SEL.task_id)
DIFF = SEL.set_index("task_id").newDiffMeasure
MODELS, MN, PN = L.MODELS, L.MNAME, L.PNAME
PROMPTS = L.PROMPTS
FACTS = {}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ data
ctx_t = pd.read_csv(ROOT / "tasks/airports/v5_ctx/tasks.csv").set_index("task_id")
noc_t = pd.read_csv(ROOT / "tasks/airports/v5_noctx/tasks.csv")
# context-off counterpart of each selected task: the same id (identical task) or the noctx task paired to it
pair = {t: t for t in noc_t.task_id if t in IDS}
pair.update({r.task_id: r.paired_task_id for r in noc_t.itertuples() if r.paired_task_id in IDS})
off2on = {}
for off, on in pair.items():
    off2on[off] = on
FACTS["paired_tasks"] = len(set(off2on.values()))
FACTS["identical_pairs"] = sum(1 for k, v in off2on.items() if k == v)

von = pd.read_csv(ROOT / "results/airports/v5_ctx/scored/scored_voted.csv", low_memory=False)
von = von[von.strategy.eq("A") & von.task_id.isin(IDS) & von.model.isin(MODELS)].assign(context=True)
von["key"] = von.task_id
voff = pd.read_csv(ROOT / "results/airports/v5_noctx/scored/scored_voted.csv", low_memory=False)
voff = voff[voff.strategy.eq("A") & voff.task_id.isin(off2on) & voff.model.isin(MODELS)].assign(context=False)
voff["key"] = voff.task_id.map(off2on)
paired = set(voff.key)
both = pd.concat([von[von.key.isin(paired)], voff])          # the 77 paired tasks, both conditions
both["diff"] = both.key.map(DIFF)
q = DIFF.quantile([1 / 3, 2 / 3]).to_numpy()
BIN = ["easy", "medium", "hard"]
both["dbin"] = pd.cut(both["diff"], [-np.inf, q[0], q[1], np.inf], labels=BIN)

# effort and energy (Strategy A, context on, 108 tasks). Energy per run in Wh (EcoLogits; Strategy A makes no search), with
# no GWP / water and no normalisation, as in the new version of the paper (energy only).
e = L.load_effort()
e = e[(e.dataset == "airports") & (e.set == "ctx") & (e.strategy == "A") & e.task_id.isin(IDS) & e.model.isin(MODELS)].copy()
cfg = e.groupby(["model", "variant"]).agg(EC=("EC_run", "mean"), template_tokens=("template_tokens", "mean"),
                                          user_tokens=("tokens_user", "mean"), out=("tokens_out", "mean")).reset_index()
assert len(cfg) == 45, len(cfg)
PFE = cfg.groupby("variant").template_tokens.mean().reindex(PROMPTS)   # prompt formulation effort: template tokens
USER = cfg.groupby("variant").user_tokens.mean().reindex(PROMPTS)     # template + task input, per task
EN_MODEL = cfg.groupby("model").EC.mean()
by_eff = PFE.sort_values().index.tolist()

# ------------------------------------------------------------------------------------------------ Table R0 (DOLAP Fig. 1, left)
t0 = pd.DataFrame({"prompt": [PN[p] for p in by_eff], "template tokens": [round(float(PFE[p])) for p in by_eff],
                   "tokens written per task": [round(float(USER[p])) for p in by_eff],
                   "mean energy per run (Wh)": [cfg[cfg.variant == p].EC.mean() for p in by_eff]})
t0.to_csv(OUT / "tables/dolap_table_r0_prompts.csv", index=False)

# ------------------------------------------------------------------------------------------------ Fig. 1 (right): energy by model
fig, ax = plt.subplots(figsize=(8, 3.6))
order = EN_MODEL.sort_values(ascending=True).index
ax.barh([MN[m] for m in order], EN_MODEL[order], color=[L.MCOLOR[m] for m in order], height=0.6)
for i, m in enumerate(order):
    ax.text(EN_MODEL[m] * 1.12, i, f"{EN_MODEL[m]:.4f} Wh", va="center", fontsize=8, color=L.INK2)
ax.set_xscale("log"); ax.set_xlim(3e-4, 1); ax.set_xlabel("energy per run, mean over the 9 prompts (Wh, log scale)")
ax.set_title("Energy consumption by model (Strategy A)"); ax.grid(axis="y", visible=False)
save(fig, "dolap_fig1_energy_by_model")
FACTS["pfe"] = {PN[p]: round(float(PFE[p])) for p in by_eff}
FACTS["energy_by_model_wh"] = {MN[m]: round(float(EN_MODEL[m]), 5) for m in MODELS}

# ------------------------------------------------------------------------------------------------ Fig. 2: energy by prompt and model
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
for ax, m in zip(axes, MODELS):
    c = cfg[cfg.model == m].set_index("variant").reindex(by_eff)
    ax.barh([PN[p] for p in by_eff], c.EC, color=L.MCOLOR[m], height=0.65)
    for i, val in enumerate(c.EC):
        ax.text(val * 1.01, i, f" {val:.4f}", va="center", fontsize=7, color=L.INK2)
    ax.set_title(MN[m]); ax.set_xlim(0, c.EC.max() * 1.35); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
axes[0].set_ylabel("prompt (by formulation effort)")
fig.supxlabel("energy per run (Wh); each panel has its own scale", fontsize=9)
save(fig, "dolap_fig2_energy_by_prompt")
FACTS["energy_prompt_range_per_model_wh"] = {MN[m]: [round(float(cfg[cfg.model == m].EC.min()), 5), round(float(cfg[cfg.model == m].EC.max()), 5)] for m in MODELS}
FACTS["energy_prompt_ratio_per_model"] = {MN[m]: round(float(cfg[cfg.model == m].EC.max() / cfg[cfg.model == m].EC.min()), 1) for m in MODELS}
FACTS["most_expensive_prompt"] = {MN[m]: PN[cfg[cfg.model == m].sort_values("EC").variant.iloc[-1]] for m in MODELS}
FACTS["spearman_template_energy"] = {MN[m]: round(float(cfg[cfg.model == m][["template_tokens", "EC"]].corr(method="spearman").iloc[0, 1]), 2) for m in MODELS}
FACTS["spearman_output_energy"] = {MN[m]: round(float(cfg[cfg.model == m][["out", "EC"]].corr(method="spearman").iloc[0, 1]), 2) for m in MODELS}

# ------------------------------------------------------------------------------------------------ Table 2
t = ctx_t.loc[list(IDS)].assign(diff=lambda d: d.index.map(DIFF))
nodes = t.groupby("node_id")["diff"].agg(["min", "max", "size"])
nodes = nodes[nodes["size"] >= 2].assign(span=lambda d: d["max"] - d["min"]).sort_values("span", ascending=False)
ex = []
for nid in nodes.index[:2]:
    g = t[t.node_id == nid].sort_values("diff")
    for _, r in g.iloc[[0, -1]].iterrows():
        H = ", ".join(h["property_name"] for h in json.loads(r.H))
        G = ", ".join(f"{x['property_name']}" + ("" if x.get("origin") == "node" else f" ({x['path'].split('->')[-1].split('-')[0] or x['element_type']})")
                      for x in json.loads(r.G))
        ex.append({"airport": nid, "task": r.name, "H": H, "G": G, "difficulty (newDiffMeasure)": round(float(r["diff"]), 2)})
tab2 = pd.DataFrame(ex)
tab2.to_csv(OUT / "tables/dolap_table2_example_tasks.csv", index=False)

# ------------------------------------------------------------------------------------------------ Table 3
rows = []
for m in MODELS:
    d = L.metrics(both[both.model == m], ["variant", "context"])
    b = d.sort_values(["f1", "precision"], ascending=False).iloc[0]
    rows.append({"model": MN[m], "F1": b.f1, "precision": b.precision, "F1 ign": b.f1_ign, "recall": b.recall,
                 "abstention": b.abstention, "context": "Y" if b.context else "N", "prompt": PN[b.variant],
                 "effort (tokens)": round(float(PFE[b.variant])), "mean energy (Wh/run)": EN_MODEL[m]})
tab3 = pd.DataFrame(rows)
tab3.to_csv(OUT / "tables/dolap_table3_best_per_model.csv", index=False)

# ------------------------------------------------------------------------------------------------ Fig. 3
fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
for j, b in enumerate(BIN):
    d = L.metrics(both[both.dbin == b], ["model", "context"])
    for i, (col, lab) in enumerate([("f1", "F1"), ("precision", "precision")]):
        ax = axes[i, j]
        pv = d.pivot(index="model", columns="context", values=col)
        for m in MODELS:
            ax.scatter(pv.loc[m, False], pv.loc[m, True], s=70, color=L.MCOLOR[m], edgecolor=L.SURFACE, zorder=3)
            ax.annotate(MN[m], (pv.loc[m, False], pv.loc[m, True]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=L.INK2)
        ax.plot([0, 1], [0, 1], ls="--", color=L.ORANGE, lw=1)
        ax.set_xlim(0, 0.8 if col == "f1" else 1); ax.set_ylim(0, 0.8 if col == "f1" else 1); ax.set_aspect("equal")
        ax.set_xlabel(f"no context ({lab})"); ax.set_ylabel(f"context ({lab})")
        n = both[(both.dbin == b)].key.nunique()
        ax.set_title(f"Difficulty: {b} ({n} tasks) — {lab}")
save(fig, "dolap_fig3_accuracy_vs_difficulty")
d3 = L.metrics(both, ["dbin", "model", "context"])
FACTS["fig3"] = {f"{b} | {MN[m]}": {k: round(float(d3[(d3.dbin == b) & (d3.model == m) & (d3.context == c)][k].iloc[0]), 3)
                                    for k in ["f1"] for c in [True, False]} for b in BIN for m in MODELS}
fig3 = d3.pivot_table(index=["dbin", "model"], columns="context", values=["f1", "precision"], observed=True)
FACTS["fig3_above_diagonal_f1"] = {b: int(sum(fig3.loc[(b, m), ("f1", True)] > fig3.loc[(b, m), ("f1", False)] for m in MODELS)) for b in BIN}
FACTS["fig3_above_diagonal_precision"] = {b: int(sum(fig3.loc[(b, m), ("precision", True)] > fig3.loc[(b, m), ("precision", False)] for m in MODELS)) for b in BIN}
FACTS["fig3_pooled"] = L.metrics(both, ["dbin", "context"]).pivot(index="dbin", columns="context", values="f1").round(3).to_dict()

# ------------------------------------------------------------------------------------------------ Fig. 4
d4 = L.metrics(both, ["model", "variant", "context"])
d4["pfe"] = d4.variant.map(PFE)
fig, axes = plt.subplots(2, 5, figsize=(18, 6.6), sharex=True, sharey="row")
for j, m in enumerate(MODELS):
    for i, (col, lab) in enumerate([("f1", "F1"), ("precision", "precision")]):
        ax = axes[i, j]
        for c, mk, colr in [(True, "o", L.BLUE), (False, "x", L.ORANGE)]:
            dd = d4[(d4.model == m) & (d4.context == c)].sort_values("pfe")
            ax.scatter(dd.pfe, dd[col], marker=mk, s=40, color=colr, label="context" if c else "no context", zorder=3)
        ax.set_title(f"{MN[m]}: {lab}"); ax.set_ylim(0, 0.75 if col == "f1" else 0.9)
        ax.legend(fontsize=7.5, loc="upper left")
        if j == 0:
            ax.set_ylabel(lab)
fig.supxlabel("prompt formulation effort (template tokens)", fontsize=9)
save(fig, "dolap_fig4_accuracy_vs_effort")
rho = d4.groupby(["model", "context"]).apply(lambda g: g[["pfe", "f1"]].corr(method="spearman").iloc[0, 1], include_groups=False)
FACTS["fig4_spearman_effort_f1"] = {f"{MN[m]} | {'ctx' if c else 'noctx'}": round(float(r), 2) for (m, c), r in rho.items()}
cx = d4.pivot_table(index=["model", "variant"], columns="context", values="f1")
FACTS["fig4_ctx_better_configs"] = int((cx[True] > cx[False]).sum())
FACTS["fig4_configs"] = int(len(cx))
FACTS["pooled_context"] = L.metrics(both, ["context"]).set_index("context")[["f1", "precision", "abstention"]].round(3).to_dict()

json.dump(FACTS, open(OUT / "dolap_facts.json", "w"), indent=1, default=float)
print(json.dumps(FACTS, indent=1, default=float))
print(tab2.to_string()); print(tab3.round(3).to_string())

"""Figures for the co-author review page (notebooks/figures/paper_reports/coauthor/), redrawn from the latest scored rows.

    venv/bin/python notebooks/coauthor_figures.py

Sections: airports A, airports B, football A, football B, both datasets. Strategy B uses the context-on sets only (the
context-off B run of 10 Nov 2026 is incomplete and is not used). Context effects use Strategy A on the paired tasks whose
given properties differ (paper_report_lib.context_pairs). Writes one PNG and PDF per figure and facts.json with the
numbers quoted in the captions."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent))
from paper_report_lib import *  # noqa: E402,F403

F = "coauthor"
DS = ["airports", "football"]
DNAME = {"airports": "Airports", "football": "Football"}
facts = {}
APPROACH = ["A zero-shot", "A best", "B search", "B search + memory"]
ACOLOR = {"A zero-shot": "#c9c8c2", "A best": "#52514e", "B search": "#2a78d6", "B search + memory": "#eb6834"}
AMARK = {"A zero-shot": "o", "A best": "s", "B search": "D", "B search + memory": "^"}

vote = {d: load_voted((d, "ctx")) for d in DS}
vote_off = {d: load_voted((d, "noctx")).query("strategy == 'A'") for d in DS}   # context-off B is not used
runs = {d: load_runs((d, "ctx")) for d in DS}
eff = load_effort()
eff = eff[eff.model.isin(MODELS) & ((eff.set == "ctx") | (eff.strategy == "A"))]
det = {d: load_b_detail(d) for d in DS}


def best_prompt(v):
    m = metrics(v[v.strategy == "A"], ["model", "approach"])
    return dict(m.loc[m.groupby("model").f1.idxmax(), ["model", "approach"]].to_numpy())


BEST = {d: best_prompt(vote[d]) for d in DS}
facts["best_prompt"] = {d: {MNAME[m]: PNAME[p] for m, p in BEST[d].items()} for d in DS}


def approaches(v, d):
    """Voted rows labelled with the four approaches of the comparisons (A zero-shot, A best, B search, B + memory)."""
    a = v[v.strategy == "A"]
    out = [a[a.approach == "zero"].assign(ap="A zero-shot"),
           a[a.apply(lambda r: BEST[d][r.model] == r.approach, axis=1)].assign(ap="A best"),
           v[v.approach == "search_evidence"].assign(ap="B search"),
           v[v.approach == "search_memory"].assign(ap="B search + memory")]
    return pd.concat(out, ignore_index=True)


AP = {d: approaches(vote[d], d) for d in DS}

# main metrics: F1 and precision; F1 ign (= 2P / (1 + P)) kept for comparison with DOLAP
SC3 = [("f1", "F1"), ("precision", "precision"), ("f1_ign", "F1 (abstentions ignored)")]


def boot3(v, by, B=1000, seed=0):
    """boot() plus the 95% bootstrap interval of precision (same resampling of the tasks of each group)."""
    t = boot(v, by, B, seed)
    rng = np.random.default_rng(seed)
    lo, hi = [], []
    for _, g in v.groupby(by, sort=False, observed=True):
        oc = g.outcome.to_numpy()
        idx = rng.integers(0, len(oc), (B, len(oc)))
        C, W = (oc == "correct")[idx].sum(1), (oc == "wrong")[idx].sum(1)
        p = np.where(C + W > 0, C / np.maximum(C + W, 1), np.nan)
        lo.append(np.nanpercentile(p, 2.5)); hi.append(np.nanpercentile(p, 97.5))
    t["precision_lo"], t["precision_hi"] = lo, hi
    return t


def pdiff(x, y, score, B=1000, seed=0):
    """paired_diff() that also handles precision (second - first, paired 95% bootstrap over the shared tasks)."""
    if score != "precision":
        return paired_diff(x, y, score, B, seed)
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner")
    rng = np.random.default_rng(seed)

    def pr(o):
        c, w = (o == "correct").sum(-1), (o == "wrong").sum(-1)
        return np.where(c + w > 0, c / np.maximum(c + w, 1), np.nan)
    xa, ya = j.x.to_numpy(), j.y.to_numpy()
    idx = rng.integers(0, len(j), (B, len(j)))
    ds = pr(ya[idx]) - pr(xa[idx])
    return float(pr(ya) - pr(xa)), np.nanpercentile(ds, 2.5), np.nanpercentile(ds, 97.5)


def tertile(v):
    t = v.drop_duplicates("task_id").set_index("task_id").difficulty
    q = pd.qcut(t, 3, labels=["easy", "middle", "hard"])
    return v.task_id.map(q)


# ================================================================================================ Strategy A
for d in DS:
    v = vote[d][vote[d].strategy == "A"]
    m = metrics(v, ["model", "approach"])
    # A1: model x prompt heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(19, 3.6))
    for ax, (s, lab) in zip(axes, SC3):
        heat(ax, m.pivot(index="model", columns="approach", values=s).reindex(index=MODELS, columns=PROMPTS),
             xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN, vmin=0, vmax=0.8)
        ax.set_title(lab)
    fig.suptitle(f"{DNAME[d]}, Strategy A, context on: score per model and prompt (voted over 3 runs)")
    save(fig, F, f"{d}_a_heatmap"); plt.close(fig)
    span_m = m.groupby("approach")[["f1", "precision", "f1_ign"]].agg(lambda s: s.max() - s.min()).mean()
    span_p = m.groupby("model")[["f1", "precision", "f1_ign"]].agg(lambda s: s.max() - s.min()).mean()
    facts[f"{d}_a"] = {"f1_range": [m.f1.min(), m.f1.max()], "f1_ign_range": [m.f1_ign.min(), m.f1_ign.max()], "precision_range": [m.precision.min(), m.precision.max()],
                       "best": {MNAME[k]: m[(m.model == k) & (m.approach == p)][["f1", "precision", "f1_ign", "abstention"]].round(3).to_dict("records")[0]
                                for k, p in BEST[d].items()},
                       "span_over_models": span_m.round(3).to_dict(), "span_over_prompts": span_p.round(3).to_dict(),
                       "prompt_range_per_model": (m.groupby("model").f1.max() - m.groupby("model").f1.min()).rename(MNAME).round(3).to_dict(),
                       "abstention_range": [m.abstention.min(), m.abstention.max()]}

    # A2: user effort vs score, per model (one point per prompt)
    e = eff[(eff.dataset == d) & (eff.set == "ctx") & (eff.strategy == "A")]
    ut = e.groupby("variant").tokens_user.mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.8), sharex=True)
    for ax, (s, lab) in zip(axes, SC3):
        for k in MODELS:
            t = m[m.model == k].set_index("approach").reindex(PROMPTS)
            x = ut.reindex(PROMPTS).to_numpy()
            o = np.argsort(x)
            ax.plot(x[o], t[s].to_numpy()[o], "-o", color=MCOLOR[k], ms=4, lw=1.2, label=MNAME[k])
        ax.set_title(lab); ax.set_xlabel("user tokens per task (template + task input)"); ax.set_ylim(0, None)
    axes[0].legend(fontsize=8, ncol=1)
    fig.suptitle(f"{DNAME[d]}, Strategy A, context on: score against user effort, one point per prompt")
    save(fig, F, f"{d}_a_effort"); plt.close(fig)
    facts[f"{d}_a"]["user_tokens"] = ut.rename(PNAME).round(0).to_dict()
    facts[f"{d}_a"]["mean_over_models_per_prompt"] = m.groupby("approach")[["f1", "precision", "f1_ign"]].mean().rename(PNAME).round(3).to_dict()

    # A3: paired context effect (on - off), pooled over prompts, on the pairs whose G differs
    pairs = context_pairs(d)
    off = vote_off[d][vote_off[d].task_id.isin(pairs)].assign(task_on=lambda x: x.task_id.map(pairs))
    rows = []
    for k in MODELS:
        x = off[off.model == k].set_index(["approach", "task_on"]).outcome
        y = v[(v.model == k) & v.task_id.isin(set(pairs.values()))].set_index(["approach", "task_id"]).outcome
        for s, _ in SC3:
            dd, lo, hi = pdiff(x, y, s)
            rows.append({"model": k, "score": s, "diff": dd, "lo": lo, "hi": hi})
    cx = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.2), sharey=True)
    for ax, (s, lab) in zip(axes, SC3):
        t = cx[cx.score == s].set_index("model").reindex(MODELS)
        y = np.arange(len(MODELS))
        ax.scatter(t["diff"], y, c=[MCOLOR[k] for k in MODELS], s=60, zorder=3, edgecolor=SURFACE)
        for xv, yv in zip(t["diff"], y):
            ax.annotate(f"{xv:+.3f}", (xv, yv), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ax.margins(x=0.25)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(MN); ax.invert_yaxis(); ax.set_title(f"Δ {lab} (context on − off)")
    fig.suptitle(f"{DNAME[d]}, Strategy A: effect of context on the {len(pairs)} paired tasks whose given properties differ (9 prompts pooled)")
    save(fig, F, f"{d}_a_context"); plt.close(fig)
    facts[f"{d}_a"]["context"] = {"pairs": len(pairs), **{f"{MNAME[r.model]} {r.score}": [round(r["diff"], 3), round(r.lo, 3), round(r.hi, 3)] for _, r in cx.iterrows()}}

    # A4: difficulty tertile, per model, A best
    ab = AP[d][AP[d].ap == "A best"].copy()
    ab["tert"] = tertile(ab)
    t = boot3(ab, ["tert", "model"])
    fig, axes = plt.subplots(1, 3, figsize=(19, 3.8))
    for ax, (s, lab) in zip(axes, SC3):
        grouped_bars(ax, t, "tert", ["easy", "middle", "hard"], "model", MODELS, s, MCOLOR, names=MNAME, err=False)
        ax.set_title(lab); ax.set_xlabel("task difficulty tertile (difficulty score)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{DNAME[d]}, Strategy A best prompt, context on: score by difficulty tertile")
    save(fig, F, f"{d}_a_difficulty"); plt.close(fig)
    tp = metrics(ab, ["tert"]).set_index("tert")
    facts[f"{d}_a"]["difficulty_pooled"] = tp[["f1", "precision", "f1_ign", "tasks"]].round(3).to_dict()

    # A5: energy per run (dot plot) and F1 against energy
    en = e.groupby(["model", "variant"]).EC_run.mean().reset_index()
    fm = en.merge(m.rename(columns={"approach": "variant"}), on=["model", "variant"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    y = np.arange(len(MODELS))
    for i, k in enumerate(MODELS):
        t = en[en.model == k]
        axes[0].scatter(t.EC_run, np.full(len(t), i), color=MCOLOR[k], s=18, alpha=0.7)
        axes[0].scatter(t.EC_run.mean(), i, color=MCOLOR[k], marker="|", s=400, lw=2.5)
        axes[0].text(t.EC_run.max() * 1.25, i, f"{t.EC_run.mean():.4f} Wh", va="center", fontsize=7.5, color=INK2)
        tt = fm[fm.model == k]
        axes[1].scatter(tt.EC_run, tt.f1, color=MCOLOR[k], s=26, label=MNAME[k], edgecolor=SURFACE)
        c = tt[tt.variant == "compact"]
        axes[1].annotate("compact", (c.EC_run.iloc[0], c.f1.iloc[0]), fontsize=6.5, color=INK2, xytext=(4, 2), textcoords="offset points")
    axes[0].set_yticks(y); axes[0].set_yticklabels(MN); axes[0].invert_yaxis(); logaxis(axes[0], "x")
    axes[0].set_xlim(axes[0].get_xlim()[0], axes[0].get_xlim()[1] * 4)
    axes[0].set_xlabel("Wh per run (log)"); axes[0].set_title("energy per run: dots = prompts, bar = mean")
    logaxis(axes[1], "x"); axes[1].set_xlabel("Wh per run (log)"); axes[1].set_ylabel("F1"); axes[1].set_title("F1 against energy, one point per prompt")
    axes[1].legend(fontsize=7.5)
    fig.suptitle(f"{DNAME[d]}, Strategy A, context on: estimated energy (EcoLogits)")
    save(fig, F, f"{d}_a_energy"); plt.close(fig)
    facts[f"{d}_a"]["energy_mean"] = en.groupby("model").EC_run.mean().rename(MNAME).round(4).to_dict()
    facts[f"{d}_a"]["energy_prompt_ratio"] = (en.groupby("model").EC_run.max() / en.groupby("model").EC_run.min()).rename(MNAME).round(1).to_dict()

# ================================================================================================ Strategy B
for d in DS:
    a = AP[d]
    t = boot3(a, ["model", "ap"])
    # B1: four approaches per model with CIs
    fig, axes = plt.subplots(1, 4, figsize=(21, 3.9))
    for ax, s, lab in zip(axes, ["f1", "precision", "f1_ign", "abstention"], ["F1", "precision", "F1 (abstentions ignored)", "abstention"]):
        grouped_bars(ax, t, "model", MODELS, "ap", APPROACH, s, ACOLOR, names=MNAME, err=False)
        ax.set_title(lab)
        if s == "abstention":
            pct(ax)
        ax.tick_params(axis="x", labelsize=7, rotation=15)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"{DNAME[d]}, context on: Strategy A (zero-shot, best prompt) and Strategy B per model, voted over 3 runs")
    save(fig, F, f"{d}_b_scores"); plt.close(fig)
    facts[f"{d}_b"] = {"scores": {f"{MNAME[r.model]} | {r.ap}": [round(r.f1, 3), round(r.f1_ign, 3), round(r.abstention, 3), round(r.precision, 3)]
                                  for _, r in t.iterrows()}}
    pooled = metrics(a, ["ap"]).set_index("ap")
    facts[f"{d}_b"]["pooled"] = pooled[["f1", "f1_ign", "precision", "abstention"]].round(3).to_dict()
    # paired B - A best
    rows = []
    for k in MODELS:
        x = a[(a.model == k) & (a.ap == "A best")].set_index("task_id").outcome
        for b in ["B search", "B search + memory"]:
            y = a[(a.model == k) & (a.ap == b)].set_index("task_id").outcome
            for s, _ in SC3:
                dd, lo, hi = pdiff(x, y, s)
                rows.append({"model": MNAME[k], "b": b, "score": s, "diff": round(dd, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
        y = a[(a.model == k) & (a.ap == "B search")].set_index("task_id").outcome
        z = a[(a.model == k) & (a.ap == "B search + memory")].set_index("task_id").outcome
        for s, _ in SC3:
            dd, lo, hi = pdiff(y, z, s)
            rows.append({"model": MNAME[k], "b": "memory - search", "score": s, "diff": round(dd, 3), "lo": round(lo, 3), "hi": round(hi, 3)})
    facts[f"{d}_b"]["paired"] = rows

    # B2: outcomes
    t2 = metrics(a[a.ap.isin(["A best", "B search", "B search + memory"])], ["model", "ap"])
    t2["cfg"] = t2.model.map(MNAME) + " · " + t2.ap
    fig, ax = plt.subplots(figsize=(9, 6))
    stacked_outcomes(ax, t2, "cfg", [f"{MNAME[k]} · {x}" for k in MODELS for x in ["A best", "B search", "B search + memory"]])
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title(f"{DNAME[d]}, context on: correct / wrong / abstained answers", pad=26)
    save(fig, F, f"{d}_b_outcomes"); plt.close(fig)

    # B3: entity identification
    dt = det[d][det[d].variant == "search_evidence"]
    k_ = ["variant", "model", "task_id", "run_index"]
    rr = runs[d][runs[d].variant.isin(["search_evidence", "search_memory"])].merge(det[d][k_ + ["entity_match", "answer_cited"]], on=k_, how="left")
    ent = dt.groupby("model").agg(named=("entity_named", "mean"), match=("entity_match", "mean"), answered=("answered", "mean")).reindex(MODELS)
    em = metrics(rr[rr.variant == "search_evidence"].assign(m=lambda x: np.where(x.entity_match, "right entity", "other runs")), ["model", "m"])
    fig, axes = plt.subplots(1, 3, figsize=(19, 3.8))
    x = np.arange(len(MODELS)); w = 0.27
    for i, (c, lab, col) in enumerate([("named", "plan names an entity", "#86b6ef"), ("match", "plan names the right entity", "#1c5cab"), ("answered", "run answers", "#eb6834")]):
        axes[0].bar(x + (i - 1) * w, ent[c], w - 0.03, color=col, label=lab)
        for xx, vv in zip(x + (i - 1) * w, ent[c]):
            axes[0].text(xx, vv + 0.01, f"{vv:.0%}", ha="center", fontsize=6.5, color=INK2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(MN, fontsize=7.5); pct(axes[0]); axes[0].legend(fontsize=7.5); axes[0].set_title("share of B search runs")
    axes[0].grid(axis="x", visible=False)
    for ax, s, lab in [(axes[1], "precision", "precision"), (axes[2], "f1_ign", "F1 (abstentions ignored)")]:
        grouped_bars(ax, em, "model", MODELS, "m", ["right entity", "other runs"], s, {"right entity": "#1c5cab", "other runs": "#c9c8c2"},
                     names=MNAME, err=False)
        ax.tick_params(axis="x", labelsize=7.5); ax.set_title(f"{lab} per run, by whether the plan named the right entity", fontsize=9.5)
    axes[1].legend(fontsize=7.5)
    fig.suptitle(f"{DNAME[d]}, Strategy B search, context on: identifying the entity")
    save(fig, F, f"{d}_b_entity"); plt.close(fig)
    facts[f"{d}_b"]["entity"] = ent.rename(index=MNAME).round(3).to_dict()
    facts[f"{d}_b"]["entity_f1"] = {f"{MNAME[r.model]} | {r.m}": [round(r.f1, 3), round(r.precision, 3), round(r.f1_ign, 3)] for _, r in em.iterrows()}

    # B4: memory answers with and without citation
    mem = rr[(rr.variant == "search_memory") & (rr.outcome != "abstained")].copy()
    mem["source"] = np.where(mem.answer_cited, "cites the evidence", "no citation (memory)")
    tm = metrics(mem, ["model", "source"])
    tm["share"] = tm.tasks / tm.groupby("model").tasks.transform("sum")
    SC = {"cites the evidence": "#1c5cab", "no citation (memory)": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    grouped_bars(axes[0], tm, "model", MODELS, "source", list(SC), "share", SC, names=MNAME, err=False)
    grouped_bars(axes[1], tm, "model", MODELS, "source", list(SC), "precision", SC, names=MNAME, err=False)
    axes[0].set_title("share of the answers given"); pct(axes[0]); axes[1].set_title("precision (correct / answered)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"{DNAME[d]}, Strategy B search + memory (single runs): answers that cite the evidence vs answers from memory")
    save(fig, F, f"{d}_b_memory"); plt.close(fig)
    facts[f"{d}_b"]["memory_sources"] = {f"{MNAME[r.model]} | {r.source}": [round(r.share, 3), round(r.precision, 3), int(r.tasks)] for _, r in tm.iterrows()}

    # B5: H x G strata, F1 and F1 ign, three approaches pooled over models
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    lv = ["high", "mid", "low"]
    for j, apn in enumerate(["A best", "B search", "B search + memory"]):
        ms = metrics(a[a.ap == apn], ["H_stratum", "G_stratum"])
        for i, (s, lab) in enumerate(SC3):
            heat(axes[i, j], ms.pivot(index="H_stratum", columns="G_stratum", values=s).reindex(index=lv, columns=lv),
                 vmin=0, vmax=0.8, cbar=j == 2, xlabels=[f"G pop. {x}" for x in lv], ylabels=[f"H pop. {x}" for x in lv])
            axes[i, j].set_title(f"{apn}: {lab}", fontsize=9.5)
    fig.suptitle(f"{DNAME[d]}, context on: score by popularity stratum of the hidden property (rows) and given properties (columns), 5 models pooled")
    save(fig, F, f"{d}_b_strata"); plt.close(fig)
    facts[f"{d}_b"]["strata"] = {apn: metrics(a[a.ap == apn], ["H_stratum", "G_stratum"]).assign(k=lambda x: x.H_stratum + "/" + x.G_stratum)
                                 .set_index("k")[["f1", "precision", "f1_ign", "tasks"]].round(2).to_dict() for apn in ["A best", "B search", "B search + memory"]}

    # B6: energy split of B
    eb = eff[(eff.dataset == d) & (eff.set == "ctx") & eff.strategy.isin(["B", "B_memory"])]
    g = eb.groupby(["variant", "model"]).agg(llm=("EC_llm", "mean"), run=("EC_run", "mean"), searches=("searches", "mean")).reset_index()
    gs = g[g.variant == "search_evidence"].set_index("model").reindex(MODELS)
    gm = g[g.variant == "search_memory"].set_index("model").reindex(MODELS)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(MODELS))
    ax.barh(y, gs.llm, color="#2a78d6", label="LLM calls (EcoLogits)")
    ax.barh(y, gs.run - gs.llm, left=gs.llm, color="#c9c8c2", label="web searches (0.3 Wh each)")
    ax.scatter(gm.run, y, marker="^", color="#eb6834", zorder=3, label="total, search + memory")
    for yy, r in zip(y, gs.itertuples()):
        ax.text(max(r.run, gm.run.iloc[yy]) + 0.03, yy, f"{r.run:.2f} Wh, {r.searches:.1f} searches ({1 - r.llm / r.run:.0%} search)", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(MN); ax.invert_yaxis(); ax.set_xlim(0, g.run.max() * 1.7); ax.set_xlabel("Wh per run (mean)")
    ax.legend(fontsize=7.5, loc="lower right"); ax.grid(axis="y", visible=False)
    ax.set_title(f"{DNAME[d]}, Strategy B search, context on: energy per run")
    save(fig, F, f"{d}_b_energy"); plt.close(fig)
    ea = eff[(eff.dataset == d) & (eff.set == "ctx") & (eff.strategy == "A")].groupby("model").EC_run.mean()
    facts[f"{d}_b"]["energy"] = {MNAME[k]: {"search": round(gs.run[k], 3), "memory": round(gm.run[k], 3), "llm_search": round(gs.llm[k], 3),
                                            "searches": round(gs.searches[k], 2), "x_A_mean": round(gs.run[k] / ea[k], 0)} for k in MODELS}

    # B7: per property, F1 ign and abstention for A best and B
    pc = a[a.ap.isin(["A best", "B search", "B search + memory"])]
    top = pc[pc.ap == "A best"].property.value_counts()
    props = top[top >= 15].index[:14].tolist()
    tp = metrics(pc[pc.property.isin(props)], ["property", "ap"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.42 * len(props) + 1.6), sharey=True)
    PC = {k: ACOLOR[k] for k in ["A best", "B search", "B search + memory"]}
    for ax, s, lab in zip(axes, ["f1", "precision", "f1_ign"], ["F1", "precision", "F1 (abstentions ignored)"]):
        yy = np.arange(len(props)); h = 0.27
        for i, apn in enumerate(PC):
            dd = tp[tp.ap == apn].set_index("property").reindex(props)
            ax.barh(yy + (i - 1) * h, dd[s], h - 0.03, color=PC[apn], label=apn)
        ax.set_yticks(yy); ax.set_yticklabels([f"{p} (n={top[p]})" for p in props], fontsize=7.5); ax.set_title(lab); ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis(); axes[2].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"{DNAME[d]}, context on: score per hidden property (5 models pooled; n = voted answers per approach)")
    save(fig, F, f"{d}_b_properties"); plt.close(fig)
    facts[f"{d}_b"]["properties"] = {f"{r.property} | {r.ap}": [round(r.f1, 2), round(r.precision, 2), round(r.f1_ign, 2), round(r.abstention, 2)] for _, r in tp.iterrows()}

# ================================================================================================ both datasets
# C1: headline dot plot
fig, axes = plt.subplots(2, 3, figsize=(19, 6.5), sharex="col")
for i, d in enumerate(DS):
    t = boot3(AP[d], ["model", "ap"])
    for j, (s, lab) in enumerate(SC3):
        ax = axes[i, j]
        for q, apn in enumerate(APPROACH):
            dd = t[t.ap == apn].set_index("model").reindex(MODELS)
            yy = np.arange(len(MODELS)) + (q - 1.5) * 0.17
            ax.plot(dd[s], yy, AMARK[apn], color=ACOLOR[apn], ms=6, label=apn)
            for xv, yv in zip(dd[s], yy):
                ax.text(xv + 0.012, yv, f"{xv:.2f}", va="center", fontsize=6, color=INK2)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MN); ax.invert_yaxis(); ax.set_title(f"{DNAME[d]}: {lab}")
        ax.grid(axis="y", visible=False); ax.set_xlim(0, 0.85)
for (i, j), ax in np.ndenumerate(axes):   # the F1 ign column is empty on the left, the others on the right
    ax.legend(fontsize=6.5, loc="lower left" if j == 2 else "lower right", frameon=True, framealpha=0.9, edgecolor=GRID)
fig.suptitle("Context on: Strategy A and Strategy B per model on both datasets (voted over 3 runs)")
save(fig, F, "both_headline"); plt.close(fig)

# C2: paired differences B - A best
fig, axes = plt.subplots(1, 3, figsize=(19, 4.4), sharey=True)
for ax, (s, lab) in zip(axes, SC3):
    ylab = []
    yy = 0
    for d in DS:
        for k in MN:
            for b, c, mk in [("B search", "#2a78d6", "D"), ("B search + memory", "#eb6834", "^")]:
                r = [x for x in facts[f"{d}_b"]["paired"] if x["model"] == k and x["b"] == b and x["score"] == s][0]
                off = -0.15 if b == "B search" else 0.15
                ax.plot(r["diff"], yy + off, mk, color=c, ms=5, label=b if (yy == 0) else None)
                ax.text(r["diff"] + (0.008 if r["diff"] >= 0 else -0.008), yy + off, f"{r['diff']:+.2f}", va="center",
                        ha="left" if r["diff"] >= 0 else "right", fontsize=6, color=INK2)
            ylab.append(f"{DNAME[d]} · {k}")
            yy += 1
        yy += 0.6
    ax.axvline(0, color=INK2, lw=0.8); ax.margins(x=0.15)
    ax.set_title(f"Δ {lab}: B − A best prompt (paired by task)")
pos = [i + (0.6 if i >= 5 else 0) for i in range(10)]
axes[0].set_yticks(pos); axes[0].set_yticklabels(ylab, fontsize=8); axes[0].invert_yaxis(); axes[0].legend(fontsize=7.5)
fig.suptitle("Context on: Strategy B against Strategy A's best prompt, per model and dataset")
save(fig, F, "both_ab_paired"); plt.close(fig)

# C3: context effect, both datasets
fig, axes = plt.subplots(1, 3, figsize=(17, 3.6), sharey=True)
for ax, (s, lab) in zip(axes, SC3):
    for i, d in enumerate(DS):
        c = facts[f"{d}_a"]["context"]
        vals = np.array([c[f"{k} {s}"] for k in MN])
        yy = np.arange(len(MN)) + (i - 0.5) * 0.3
        ax.plot(vals[:, 0], yy, "o", color=DCOLOR[d], ms=6, label=f"{DNAME[d]} (n = {c['pairs']} pairs)")
        for xv, yv in zip(vals[:, 0], yy):
            ax.text(xv, yv - 0.12, f"{xv:+.3f}", ha="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8); ax.margins(x=0.15); ax.set_title(f"Δ {lab} (context on − off)")
axes[0].set_yticks(range(len(MN))); axes[0].set_yticklabels(MN); axes[0].invert_yaxis(); axes[0].legend(fontsize=7.5)
fig.suptitle("Strategy A: effect of graph context on the paired tasks whose given properties differ (9 prompts pooled)")
save(fig, F, "both_context"); plt.close(fig)

# C4: H stratum and difficulty tertile per approach, both datasets
for name, col, levels, xl in [("both_hstratum", "H_stratum", ["low", "mid", "high"], "popularity stratum of the hidden property"),
                              ("both_difficulty", "tert", ["easy", "middle", "hard"], "difficulty tertile (difficulty score)")]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    for i, d in enumerate(DS):
        a = AP[d].copy()
        a["tert"] = tertile(a)
        t = boot3(a, [col, "ap"])
        for j, (s, lab) in enumerate(SC3):
            ax = axes[i, j]
            for apn in APPROACH:
                dd = t[t.ap == apn].set_index(col).reindex(levels)
                ax.plot(range(3), dd[s], "-" + AMARK[apn], color=ACOLOR[apn], ms=5, label=apn)
                for xv, yv in zip(range(3), dd[s] if apn != "A zero-shot" else []):
                    ax.annotate(f"{yv:.2f}", (xv, yv), xytext=(7, 0), textcoords="offset points", va="center", fontsize=6, color=ACOLOR[apn])
            ax.set_xticks(range(3)); ax.set_xticklabels(levels); ax.set_title(f"{DNAME[d]}: {lab}", fontsize=9.5); ax.set_xlabel(xl, fontsize=8)
            facts.setdefault(name, {})[f"{d} {s}"] = {apn: t[t.ap == apn].set_index(col).reindex(levels)[s].round(3).tolist() for apn in APPROACH}
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Context on: score by {xl}, 5 models pooled")
    save(fig, F, name); plt.close(fig)

# C5: energy vs score, all configurations
fig, axes = plt.subplots(2, 3, figsize=(19, 7.5), sharex=True)
for i, d in enumerate(DS):
    e = eff[(eff.dataset == d) & (eff.set == "ctx")].groupby(["model", "variant"]).EC_run.mean().reset_index()
    sc = metrics(vote[d], ["model", "approach"]).rename(columns={"approach": "variant"})
    j_ = e.merge(sc, on=["model", "variant"])
    for j, (s, lab) in enumerate(SC3):
        ax = axes[i, j]
        for k in MODELS:
            t = j_[j_.model == k]
            aa = t[~t.variant.str.startswith("search")]
            ax.scatter(aa.EC_run, aa[s], color=MCOLOR[k], s=16, alpha=0.75)
            for v_, mk in [("search_evidence", "D"), ("search_memory", "^")]:
                bb = t[t.variant == v_]
                ax.scatter(bb.EC_run, bb[s], color=MCOLOR[k], marker=mk, s=70, edgecolor=INK, linewidth=0.6)
        logaxis(ax, "x"); ax.set_title(f"{DNAME[d]}: {lab}"); ax.set_ylim(0, None)
        if i == 1:
            ax.set_xlabel("Wh per run (log)")
h = model_handles() + [plt.Line2D([], [], ls="", marker="o", color=MUTED, ms=5, label="A: one prompt"),
                       plt.Line2D([], [], ls="", marker="D", color=MUTED, ms=7, label="B: search"),
                       plt.Line2D([], [], ls="", marker="^", color=MUTED, ms=7, label="B: search + memory")]
axes[0, 0].legend(handles=h, fontsize=7, ncol=2)
fig.suptitle("Context on: score against estimated energy per run (LLM calls + 0.3 Wh per web search), every configuration")
save(fig, F, "both_energy_f1"); plt.close(fig)

# C6: token amplification per approach
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
AMP = {"A (mean of 9 prompts)": "#52514e", "B search": "#2a78d6", "B search + memory": "#eb6834"}
for ax, d in zip(axes, DS):
    e = eff[(eff.dataset == d) & (eff.set == "ctx")].copy()
    e["amp"] = e.tokens_proc / e.tokens_user
    e["g"] = np.select([e.variant == "search_evidence", e.variant == "search_memory"], ["B search", "B search + memory"], "A (mean of 9 prompts)")
    t = e.groupby(["model", "g"]).amp.mean().reset_index()
    grouped_bars(ax, t, "model", MODELS, "g", list(AMP), "amp", AMP, names=MNAME, err=False)
    ax.axhline(1, color=INK2, lw=0.8, ls="--"); ax.set_title(DNAME[d]); ax.tick_params(axis="x", labelsize=7.5)
    facts.setdefault("amplification", {})[d] = {f"{MNAME[r.model]} | {r.g}": round(r.amp, 2) for _, r in t.iterrows()}
    facts["amplification"][f"{d} per prompt"] = e[e.strategy == "A"].groupby("variant").amp.mean().rename(PNAME).round(2).to_dict()
axes[0].set_ylabel("|T_process| / |T_user|"); axes[0].legend(fontsize=7.5)
fig.suptitle("Context on: token amplification (tokens processed per token written by the user)")
save(fig, F, "both_amplification"); plt.close(fig)

# C7: model vs prompt spread
rows = []
for d in DS:
    for s, lab in SC3:
        rows.append({"d": DNAME[d], "what": "over models (fixed prompt)", "s": lab, "v": facts[f"{d}_a"]["span_over_models"][s]})
        rows.append({"d": DNAME[d], "what": "over prompts (fixed model)", "s": lab, "v": facts[f"{d}_a"]["span_over_prompts"][s]})
sp = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.4), sharey=True)
for ax, (s, lab) in zip(axes, SC3):
    grouped_bars(ax, sp[sp.s == lab], "d", ["Airports", "Football"], "what", ["over models (fixed prompt)", "over prompts (fixed model)"], "v",
                 {"over models (fixed prompt)": "#1c5cab", "over prompts (fixed model)": "#86b6ef"}, err=False)
    ax.set_title(f"mean span (max − min) of {lab}")
axes[0].legend(fontsize=7.5)
fig.suptitle("Strategy A, context on: does the model or the prompt change the score more?")
save(fig, F, "both_model_vs_prompt"); plt.close(fig)

# C8: entity identification on both datasets
fig, ax = plt.subplots(figsize=(11, 3.6))
x = np.arange(len(MODELS)); w = 0.2
for i, (d, c, lab) in enumerate([("airports", "named", "Airports: names an entity"), ("airports", "match", "Airports: right entity"),
                                 ("football", "named", "Football: names an entity"), ("football", "match", "Football: right entity")]):
    vv = [facts[f"{d}_b"]["entity"][c][k] for k in MN]
    ax.bar(x + (i - 1.5) * w, vv, w - 0.02, color=["#86b6ef", "#1c5cab", "#f4a582", "#c2410c"][i], label=lab)
ax.set_xticks(x); ax.set_xticklabels(MN); pct(ax); ax.legend(fontsize=7.5, ncol=2); ax.grid(axis="x", visible=False)
ax.set_title("Strategy B search, context on: share of runs whose plan names an entity, and the right one")
save(fig, F, "both_entity"); plt.close(fig)

# ================================================================================================ precision, recall, abstention
PRA = [("precision", "precision = c / (c + w)"), ("recall", "recall = c / (c + w + a)"), ("abstention", "abstention = a / (c + w + a)")]


def boot_pra(v, by, B=1000, seed=0):
    """metrics() with 95% bootstrap intervals of precision, recall and abstention (resampling the tasks of each group)."""
    rng = np.random.default_rng(seed)
    t = metrics(v, by)
    ci = {f"{s}_{e}": [] for s, _ in PRA for e in ("lo", "hi")}
    for _, g in v.groupby(by, sort=False, observed=True):
        oc = g.outcome.to_numpy()
        idx = rng.integers(0, len(oc), (B, len(oc)))
        C, W = (oc == "correct")[idx].sum(1), (oc == "wrong")[idx].sum(1)
        n = len(oc)
        vals = {"precision": np.where(C + W > 0, C / np.maximum(C + W, 1), np.nan), "recall": C / n, "abstention": (n - C - W) / n}
        for s, _ in PRA:
            ci[f"{s}_lo"].append(np.nanpercentile(vals[s], 2.5)); ci[f"{s}_hi"].append(np.nanpercentile(vals[s], 97.5))
    for k, x in ci.items():
        t[k] = x
    return t


for d in DS:
    # Strategy A: precision / recall / abstention per model and prompt
    m = metrics(vote[d][vote[d].strategy == "A"], ["model", "approach"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.6))
    for ax, (s, lab) in zip(axes, PRA):
        heat(ax, m.pivot(index="model", columns="approach", values=s).reindex(index=MODELS, columns=PROMPTS),
             xlabels=[PNAME[p] for p in PROMPTS], ylabels=MN, vmin=0, vmax=1, fmt="{:.0%}" if s == "abstention" else "{:.2f}",
             cmap=SEQ)
        ax.set_title(lab)
    fig.suptitle(f"{DNAME[d]}, Strategy A, context on: precision, recall and abstention per model and prompt (voted over 3 runs)")
    save(fig, F, f"{d}_a_prf"); plt.close(fig)
    facts[f"{d}_a"]["prf"] = {s: [round(m[s].min(), 3), round(m[s].max(), 3)] for s, _ in PRA}
    facts[f"{d}_a"]["prf_per_model_mean"] = m.groupby("model")[["precision", "recall", "abstention"]].mean().rename(MNAME).round(3).to_dict()

# both datasets: the four approaches per model
fig, axes = plt.subplots(2, 3, figsize=(17, 7.2), sharey="col")
for i, d in enumerate(DS):
    t = boot_pra(AP[d], ["model", "ap"])
    facts.setdefault("prf", {})[d] = {f"{MNAME[r.model]} | {r.ap}": [round(r.precision, 3), round(r.recall, 3), round(r.abstention, 3)] for _, r in t.iterrows()}
    facts["prf"][f"{d} pooled"] = metrics(AP[d], ["ap"]).set_index("ap")[["precision", "recall", "abstention"]].round(3).to_dict()
    for j, (s, lab) in enumerate(PRA):
        ax = axes[i, j]
        grouped_bars(ax, t, "model", MODELS, "ap", APPROACH, s, ACOLOR, names=MNAME, err=False)
        ax.set_title(f"{DNAME[d]}: {lab}"); pct(ax) if s == "abstention" else None
        ax.tick_params(axis="x", labelsize=7.5); ax.set_ylim(0, 1)
axes[0, 0].legend(fontsize=7.5, ncol=2)
fig.suptitle("Context on: precision, recall and abstention per model for Strategy A (zero-shot, best prompt) and Strategy B")
save(fig, F, "both_prf"); plt.close(fig)

# both datasets: precision against recall, every configuration, with iso-F1 curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for ax, d in zip(axes, DS):
    m = metrics(vote[d], ["model", "approach"])
    r_ = np.linspace(0.01, 1, 200)
    for f1 in (0.1, 0.2, 0.3, 0.4, 0.5):
        p_ = f1 * r_ / (2 * r_ - f1)
        ok = (p_ > 0) & (p_ <= 1)
        ax.plot(r_[ok], p_[ok], color=GRID, lw=1, zorder=0)
        ax.text(0.445, f1 * 0.445 / (2 * 0.445 - f1), f"F1 = {f1}", fontsize=6.5, color=MUTED) if 2 * 0.445 > f1 and f1 * 0.445 / (2 * 0.445 - f1) <= 0.95 else None
    for k in MODELS:
        t = m[m.model == k]
        a = t[~t.approach.str.startswith("search")]
        ax.scatter(a.recall, a.precision, color=MCOLOR[k], s=18, alpha=0.75)
        for v_, mk in [("search_evidence", "D"), ("search_memory", "^")]:
            b = t[t.approach == v_]
            ax.scatter(b.recall, b.precision, color=MCOLOR[k], marker=mk, s=75, edgecolor=INK, linewidth=0.6)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1); ax.set_xlabel("recall"); ax.set_title(DNAME[d])
axes[0].set_ylabel("precision")
axes[1].legend(handles=h, fontsize=7, ncol=2, loc="upper right")
fig.suptitle("Context on: precision against recall for every configuration (A prompts: dots; B search: diamonds; B + memory: triangles)")
save(fig, F, "both_precision_recall"); plt.close(fig)

# F1 of the 9 Strategy A prompts per model (box) with Strategy B on top
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, d in zip(axes, DS):
    m = metrics(vote[d], ["model", "approach"])
    a = m[~m.approach.str.startswith("search")]
    x = np.arange(len(MODELS))
    bp = ax.boxplot([a[a.model == k].f1 for k in MODELS], positions=x, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, k in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MCOLOR[k]); patch.set_alpha(0.35); patch.set_edgecolor(MCOLOR[k])
    rng = np.random.default_rng(0)
    for i, k in enumerate(MODELS):
        f = a[a.model == k].f1.to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(f)), f, color=MCOLOR[k], s=14, zorder=3, edgecolor=SURFACE, linewidth=0.5)
        for v_, mk, c, dx in [("search_evidence", "D", "#2a78d6", 0.36), ("search_memory", "^", "#eb6834", 0.36)]:
            b = m[(m.model == k) & (m.approach == v_)].f1.iloc[0]
            ax.scatter(i + dx, b, marker=mk, color=c, s=55, zorder=4, edgecolor=INK, linewidth=0.6)
            ax.text(i + dx + 0.07, b, f"{b:.2f}", va="center", fontsize=6.5, color=INK2)
        ax.text(i, f.max() + 0.012, f"median {np.median(f):.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(MN, fontsize=8); ax.set_title(DNAME[d]); ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(MODELS) - 0.2)
    facts.setdefault("f1_box", {})[d] = {MNAME[k]: [round(v, 3) for v in np.percentile(a[a.model == k].f1, [0, 25, 50, 75, 100])] for k in MODELS}
axes[0].set_ylabel("F1 (voted over 3 runs)"); axes[0].set_ylim(0, None)
hb = [plt.Line2D([], [], ls="", marker="s", color=MUTED, alpha=0.5, ms=9, label="A: 9 prompts (box = quartiles, dots = prompts)"),
      plt.Line2D([], [], ls="", marker="D", color="#2a78d6", ms=6, label="B: search"),
      plt.Line2D([], [], ls="", marker="^", color="#eb6834", ms=7, label="B: search + memory")]
for ax in axes:
    ax.legend(handles=hb, fontsize=7, loc="upper left", frameon=True, framealpha=0.9, edgecolor=GRID)
fig.suptitle("Context on: F1 of the 9 Strategy A prompts per model, with Strategy B for comparison")
save(fig, F, "both_f1_box"); plt.close(fig)

# F1 of the 9 Strategy A prompts per model, Strategy A only
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, d in zip(axes, DS):
    m = metrics(vote[d][vote[d].strategy == "A"], ["model", "approach"])
    x = np.arange(len(MODELS))
    bp = ax.boxplot([m[m.model == k].f1 for k in MODELS], positions=x, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, k in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MCOLOR[k]); patch.set_alpha(0.35); patch.set_edgecolor(MCOLOR[k])
    rng = np.random.default_rng(0)
    for i, k in enumerate(MODELS):
        t = m[m.model == k]
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(t)), t.f1, color=MCOLOR[k], s=16, zorder=3, edgecolor=SURFACE, linewidth=0.5)
        top = t.loc[t.f1.idxmax()]
        ax.annotate(PNAME[top.approach], (i, top.f1), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        ax.text(i, t.f1.max() + 0.03, f"median {t.f1.median():.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(MN, fontsize=8); ax.set_title(DNAME[d]); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("F1 (voted over 3 runs)"); axes[0].set_ylim(0, None)
fig.suptitle("Strategy A, context on: F1 of the 9 prompts per model (box = quartiles and median, dots = prompts; best prompt named)")
save(fig, F, "both_f1_box_a"); plt.close(fig)

(FIGDIR / F / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
print("done", len(list((FIGDIR / F).glob("*.png"))), "figures")

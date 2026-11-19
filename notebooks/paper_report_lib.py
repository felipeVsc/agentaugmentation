"""Shared data loading, scoring and chart helpers of the paper reports (notebooks/build_paper_reports_notebook.py).

Everything is computed from the scored rows (results/<dataset>/<set>/scored/), the per-run effort of
results/energy_effort/runs.csv and the Strategy B search detail (scored/b_search_*.csv), so every number in the reports
can be recomputed by counting rows in those files. Scoring as in the other reports: voted answer per (model,
configuration, task); F1 = 2c / (2c + 2w + a), F1 with abstentions ignored = 2c / (2c + w)."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SETDIR = {("airports", "ctx"): "airports/v5_ctx", ("airports", "noctx"): "airports/v5_noctx",
          ("football", "ctx"): "football/v1_ctx_idmix", ("football", "noctx"): "football/v1_noctx_idmix"}
E_SEARCH_WH = 0.3   # energy of one web search (Wh), as in the energy and effort report

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd8", "#d9a514"
NEUTRAL = "#c9c8c2"
SEQ = LinearSegmentedColormap.from_list("blue", ["#eef5fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f4a582", "#f2f1ed", "#86b6ef", "#1c5cab"])
DCOLOR = {"airports": "#86b6ef", "football": "#1c5cab"}
SCOLOR = {"ctx": "#1c5cab", "noctx": "#86b6ef"}
OCOLOR = {"correct": AQUA, "wrong": ORANGE, "abstained": NEUTRAL}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.titlesize": 12, "figure.titleweight": "bold",
    "font.size": 9.5, "legend.frameon": False, "figure.dpi": 110,
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
SHOTS = {"zero": 0, "zerocot": 0, "few_1": 1, "fewcot_1": 1, "few_3": 3, "fewcot_3": 3, "few_5": 5, "fewcot_5": 5}
SCORES = [("f1", "F1"), ("f1_ign", "F1 (abstentions ignored)")]
STRATA = ["low", "mid", "high"]
FACTORS = {"H_stratum": "pop(p) of the hidden property (H stratum)", "G_stratum": "given properties (G stratum)",
           "node_stratum": "centrality of the node (node stratum)"}
FIGDIR = ROOT / "notebooks/figures/paper_reports"

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")


# ------------------------------------------------------------------------------------------------------- data
def load_voted(*keys):
    """Voted rows of the given (dataset, set) keys, one row per (approach, model, task); `approach` is the prompt for A,
    'search_evidence' / 'search_memory' for B (the memory variant is strategy B_memory in every set)."""
    out = []
    for d, s in keys:
        v = pd.read_csv(ROOT / "results" / SETDIR[(d, s)] / "scored/scored_voted.csv", low_memory=False)
        v["strategy"] = np.where(v.variant == "search_memory", "B_memory", v.strategy)
        out.append(v.assign(dataset=d, set=s, approach=v.variant))
    v = pd.concat(out, ignore_index=True)
    v = v[v.model.isin(MODELS)]
    v["mname"] = v.model.map(MNAME)
    return v


def load_runs(*keys):
    """Scored rows of the single runs (same keys as load_voted)."""
    out = []
    for d, s in keys:
        r = pd.read_csv(ROOT / "results" / SETDIR[(d, s)] / "scored/scored_runs.csv", low_memory=False)
        r["strategy"] = np.where(r.variant == "search_memory", "B_memory", r.strategy)
        out.append(r.assign(dataset=d, set=s, approach=r.variant))
    r = pd.concat(out, ignore_index=True)
    return r[r.model.isin(MODELS)]


def load_effort():
    """Per-run effort (results/energy_effort/runs.csv) with the energy of the searches and token amplification."""
    e = pd.read_csv(ROOT / "results/energy_effort/runs.csv")
    e["approach"] = e.variant
    e["EC_llm"] = e.EC_wh
    e["EC_run"] = e.EC_wh + E_SEARCH_WH * e.searches
    e["tokens_user"] = e[["template_tokens", "input_tokens", "context_tokens", "targets_tokens"]].sum(axis=1)
    e["tokens_proc"] = e.tokens_in + e.tokens_out
    return e


def load_b_detail(dataset):
    return pd.read_csv(ROOT / "results" / SETDIR[(dataset, "ctx")] / "scored/b_search_runs.csv")


def load_domains(dataset):
    return pd.read_csv(ROOT / "results" / SETDIR[(dataset, "ctx")] / "scored/b_evidence_domains.csv")


def context_pairs(dataset):
    """Context-off task id -> its context-on task, for the pairs whose context-on G has context properties (the other
    pairs are the same task in both sets, so they give the same answers from the response cache)."""
    c = pd.read_csv(ROOT / "tasks" / SETDIR[(dataset, "ctx")] / "tasks.csv", usecols=["task_id", "n_G_context"])
    n = pd.read_csv(ROOT / "tasks" / SETDIR[(dataset, "noctx")] / "tasks.csv", usecols=["task_id", "paired_task_id"])
    m = n.merge(c, left_on="paired_task_id", right_on="task_id", suffixes=("_off", "_on"))
    m = m[m.n_G_context > 0]
    return dict(zip(m.task_id_off, m.task_id_on))


def popularity(dataset):
    return pd.read_csv(ROOT / "tasks" / SETDIR[(dataset, "ctx")] / "property_popularity.csv")


# ------------------------------------------------------------------------------------------------------ scores
def _scores(c, w, a):
    n = c + w + a
    p = c / (c + w) if c + w else np.nan
    r = c / n if n else np.nan
    return {"tasks": n, "correct": c, "wrong": w, "abstained": a, "precision": p, "recall": r,
            "f1": 2 * c / (2 * c + 2 * w + a) if n else np.nan, "f1_ign": 2 * c / (2 * c + w) if c + w else np.nan,
            "abstention": a / n if n else np.nan, "error": w / n if n else np.nan}


def metrics(df, by):
    rows = []
    for k, g in df.groupby(by, sort=False, observed=True):
        o = g.outcome.value_counts()
        rows.append({**dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,))),
                     **_scores(o.get("correct", 0), o.get("wrong", 0), o.get("abstained", 0))})
    return pd.DataFrame(rows)


def boot(df, by, B=1000, seed=0):
    """metrics() with 95% bootstrap intervals of F1 and F1 ign (resampling the tasks of each group)."""
    rng = np.random.default_rng(seed)
    t = metrics(df, by)
    lo = {k: [] for k in ("f1_lo", "f1_hi", "f1_ign_lo", "f1_ign_hi")}
    for k, g in df.groupby(by, sort=False, observed=True):
        oc = g.outcome.to_numpy()
        c, w = (oc == "correct").astype(int), (oc == "wrong").astype(int)
        idx = rng.integers(0, len(oc), (B, len(oc)))
        C, W = c[idx].sum(1), w[idx].sum(1)
        A = len(oc) - C - W
        f1 = 2 * C / np.maximum(2 * C + 2 * W + A, 1)
        fi = np.where(C + W > 0, 2 * C / np.maximum(2 * C + W, 1), np.nan)
        lo["f1_lo"].append(np.percentile(f1, 2.5)); lo["f1_hi"].append(np.percentile(f1, 97.5))
        lo["f1_ign_lo"].append(np.nanpercentile(fi, 2.5)); lo["f1_ign_hi"].append(np.nanpercentile(fi, 97.5))
    for k, v in lo.items():
        t[k] = v
    return t


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(x, y, key="task_id"):
    """Task-by-task outcomes of two configurations (Series of outcomes indexed by task): counts and McNemar on correct."""
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner")
    xr, yr = j.x == "correct", j.y == "correct"
    r = {"tasks": len(j), "both right": int((xr & yr).sum()), "only first right": int((xr & ~yr).sum()),
         "only second right": int((~xr & yr).sum()), "neither right": int((~xr & ~yr).sum())}
    r["p (McNemar)"] = mcnemar(r["only first right"], r["only second right"])
    return r


def paired_diff(x, y, score="f1", B=1000, seed=0):
    """Difference second - first of a score over the shared tasks, with a paired 95% bootstrap interval."""
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner")
    rng = np.random.default_rng(seed)

    def sc(o):
        c, w = (o == "correct").sum(-1), (o == "wrong").sum(-1)
        a = o.shape[-1] - c - w
        return 2 * c / np.maximum(2 * c + 2 * w + a, 1) if score == "f1" else 2 * c / np.maximum(2 * c + w, 1)
    xa, ya = j.x.to_numpy(), j.y.to_numpy()
    d = sc(ya) - sc(xa)
    idx = rng.integers(0, len(j), (B, len(j)))
    ds = sc(ya[idx]) - sc(xa[idx])
    return d, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def agreement(runs, by):
    """Share of (task) cells whose runs all have the same outcome, and the sd of the F1 of the single runs."""
    g = runs.groupby(by + ["task_id"]).outcome.agg(lambda s: s.nunique() == 1).groupby(level=list(range(len(by)))).mean()
    per = metrics(runs, by + ["run_index"])
    sd = per.groupby(by).agg(f1_sd=("f1", "std"), f1_ign_sd=("f1_ign", "std"))
    return pd.concat([g.rename("same outcome in all runs"), sd], axis=1).reset_index()


def numeric_error(v):
    """Relative error |answer - truth| / |truth| of the wrong numeric answers (one per row; NaN otherwise)."""
    def one(a, t):
        try:
            a, t = json.loads(a), json.loads(t)
        except (TypeError, ValueError):
            return np.nan
        try:
            a, t = float(a), float(t)   # numbers may come as strings ("0")
        except (TypeError, ValueError):
            return np.nan
        return abs(a - t) / abs(t) if t else np.nan
    return [one(a, t) if o == "wrong" else np.nan for a, t, o in zip(v.answer, v.truth, v.outcome)]


# ------------------------------------------------------------------------------------------------------ tables
VNAME = {"search_evidence": "search", "search_memory": "search + memory"}
PCT_COLS = {"abstention", "error", "share", "named", "match", "answered", "truncated", "same outcome in all runs"}


def table(df, folder, name, caption=None, pct=(), digits=3):
    """Shows a results table (model and variant ids as names, rounded, shares as %) and saves it as
    notebooks/figures/paper_reports/<folder>/tables/<name>.csv (values rounded to 4 decimals, shares as fractions)."""
    from IPython.display import display
    d = df.copy() if isinstance(df.index, pd.RangeIndex) else df.reset_index()
    for c in d.columns:
        if c in ("model", "mname"):
            d[c] = d[c].map(lambda m: MNAME.get(m, m))
        elif c in ("approach", "variant"):
            d[c] = d[c].map(lambda v: VNAME.get(v, PNAME.get(v, v)))
    d = d.rename(columns={"approach": "variant", "mname": "model"})
    out = figdir(folder) / "tables"
    out.mkdir(exist_ok=True)
    d.round(4).to_csv(out / f"{name}.csv", index=False)
    pcts = set(pct) | (PCT_COLS & set(d.columns))
    fmt = {c: ("{:.0%}" if c in pcts else f"{{:.{digits}f}}") for c in d.columns if pd.api.types.is_float_dtype(d[c])}
    st = d.style.hide(axis="index").format(fmt, na_rep="–")
    if caption:
        st = st.set_caption(caption).set_table_styles([{"selector": "caption", "props": "caption-side: top; font-weight: bold; text-align: left"}])
    display(st)


# ------------------------------------------------------------------------------------------------------ charts
def figdir(name):
    d = FIGDIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, folder, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir(folder) / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.show()


def model_handles(models=MODELS, marker="o"):
    return [plt.Line2D([], [], ls="", marker=marker, color=MCOLOR[m], ms=7, label=MNAME[m]) for m in models]


def grouped_bars(ax, tab, cat, cats, hue, hues, val, colors, names=None, err=True, labels=True, fmt="{:.2f}", width=0.8):
    """Bars of `val` per category (x) and hue, with the bootstrap interval when `err` and the columns exist."""
    x = np.arange(len(cats))
    w = width / len(hues)
    if val in ("abstention", "error", "share"):
        fmt = "{:.0%}"
    for i, h in enumerate(hues):
        d = tab[tab[hue] == h].set_index(cat).reindex(cats)
        pos = x + (i - (len(hues) - 1) / 2) * w
        ax.bar(pos, d[val], w - 0.03, color=colors[h], label=(names or {}).get(h, h))
        if err and f"{val}_lo" in d:
            ax.errorbar(pos, d[val], yerr=[d[val] - d[f"{val}_lo"], d[f"{val}_hi"] - d[val]], fmt="none",
                        ecolor=INK2, elinewidth=0.8, capsize=2)
        if labels:
            top = d[f"{val}_hi"] if err and f"{val}_hi" in d else d[val]
            bot = d[f"{val}_lo"] if err and f"{val}_lo" in d else d[val]
            for p_, v_, t_, b_ in zip(pos, d[val], top, bot):
                if pd.notna(v_):   # above the interval, or below it for a negative value
                    neg = v_ < 0
                    ax.text(p_, (b_ - 0.008) if neg else (t_ + 0.008), fmt.format(v_), ha="center", va="top" if neg else "bottom",
                            fontsize=6.5, color=INK2, rotation=90 if len(hues) > 3 else 0)
    if labels:
        ax.margins(y=0.12)   # room for the value labels above the intervals
    ax.set_xticks(x)
    ax.set_xticklabels([(names or {}).get(c, c) for c in cats] if cat != hue else cats)
    ax.grid(axis="x", visible=False)


def heat(ax, pivot, fmt="{:.2f}", cmap=SEQ, vmin=None, vmax=None, xlabels=None, ylabels=None, cbar=True, center=None):
    m = pivot.to_numpy(dtype=float)
    if center is not None:
        r = np.nanmax(np.abs(m - center))
        vmin, vmax = center - r, center + r
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    lo, hi = im.get_clim()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isnan(m[i, j]):
                continue
            frac = (m[i, j] - lo) / (hi - lo) if hi > lo else 0
            dark = frac > 0.62 if center is None else abs(frac - 0.5) > 0.33
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(xlabels or [str(c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(ylabels or [str(c) for c in pivot.index])
    ax.grid(False)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return im


def stacked_outcomes(ax, tab, cat, cats, names=None, horizontal=True):
    """Shares of correct / wrong / abstained per category, as 100% stacked bars."""
    d = tab.set_index(cat).reindex(cats)
    left = np.zeros(len(cats))
    y = np.arange(len(cats))
    for o in ("correct", "wrong", "abstained"):
        share = d[o] / d.tasks
        kw = dict(color=OCOLOR[o], label=o, edgecolor=SURFACE, linewidth=1)
        if horizontal:
            ax.barh(y, share, 0.72, left=left, **kw)
        else:
            ax.bar(y, share, 0.72, bottom=left, **kw)
        for yy, l_, s in zip(y, left, share):
            if s >= 0.1:
                xx, yv = (l_ + s / 2, yy) if horizontal else (yy, l_ + s / 2)
                ax.text(xx, yv, f"{s:.0%}", ha="center", va="center", fontsize=7, color=INK if o != "correct" else "white")
        left = left + share.fillna(0).to_numpy()
    if horizontal:
        ax.set_yticks(y); ax.set_yticklabels([(names or {}).get(c, c) for c in cats]); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(y); ax.set_xticklabels([(names or {}).get(c, c) for c in cats], rotation=35, ha="right")
        ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="x", visible=False)


def pct(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def logaxis(ax, which="y"):
    for a in which:
        (ax.set_xscale if a == "x" else ax.set_yscale)("log")
        (ax.xaxis if a == "x" else ax.yaxis).set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

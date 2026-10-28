"""Task definition of the published DOLAP 2026 paper.

"On the Efficacy of Using LLMs for Context Driven Entity Augmentation in
Property Graphs" (Vasconcelos et al., DOLAP 2026), Algorithm 2, as implemented
in github.com/felipeVsc/llmaugmentation (``main/context.ipynb``).

Two entry points:

- ``build_published``: draw a new task set with the published design, from the
  snapshot (any dataset).
- ``import_published``: convert the exact published task file
  (``test_results.csv``) to ``tasks.csv`` / ``ground_truth.csv``, so the new
  strategies run on the very same 36 tasks.

The published code and the printed Algorithm 2 differ. ``published.profile``
chooses one of them (see ``PUBLISHED_PROFILES`` in config.py and ``DEVIATIONS``):

| | code (default) | paper (Algorithm 2) |
|---|---|---|
| node strata | terciles of PageRank (``pd.qcut(q=3)``) | quartiles: Q1, Q2∪Q3, Q4 |
| easy / hard properties | global: pop >= 80% / < 80% | per node: pop <= Q1 / >= Q3 over context(n) |
| H pool | the whole class, even properties n has no value for | class ∩ properties of n |
| G | 3 from class ∩ props(n), minus H; topped up to 2 from the class | 3 from (class ∩ props(n)) \\ H |
| H per task | redrawn for each of the 4 tasks | H1, H4 drawn once per node |

Both profiles: 3 nodes per stratum, |H| = |G| = 3, tasks (H,G) in {easy,hard}²,
so 36 tasks; difficulty = specificity(H) + specificity(G) with
specificity(S) = exp(−mean_{p∈S} ln(pop(p) × cent(n))) and cent(n) the
log-min-max-normalized PageRank, as in the published code.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import ConfigError, TaskDefConfig
from .graph import Graph, infer_type, is_null
from .output import write_csv, write_json

log = logging.getLogger(__name__)

DESIGN = "dolap2026"
STRATA = ("easy", "medium", "hard")  # published order: easy (top centrality) first
COMBOS = (("easy", "easy"), ("easy", "hard"), ("hard", "easy"), ("hard", "hard"))  # (H class, G class)

PUBLISHED_TASK_COLUMNS = [
    "task_id",
    "task_set_version",
    "dataset",
    "design",
    "node_type",
    "node_id",
    "node_centrality",
    "node_centrality_raw",
    "node_stratum",
    "H_stratum",
    "G_stratum",
    "difficulty_prop",
    "size_H",
    "size_G",
    "H",
    "G",
    "G_drawn",
    "context",
    "difficulty",
    "specificity_H",
    "specificity_G",
    "published_difficulty",
    "published_specificity_H",
    "published_specificity_G",
    "published_input",
    "published_input_context",
]

DEVIATIONS = [
    "Draws use random.Random(published.seed); the published run used unseeded random, so the same "
    "procedure gives other nodes and properties. Use `tasks import-published` for the exact published tasks.",
    "Nodes whose normalized centrality is 0 (the minimum PageRank) are never drawn: ln(0) makes their "
    "difficulty undefined (the published code would have crashed on them).",
    "code profile: when class ∩ props(n) has fewer than size_G properties, all of them are drawn "
    "(random.sample in the published code would have raised).",
    "code profile: the top-up of G to min_G draws from the whole class, like the published code, so a "
    "G property can have no value on n; it is kept in G_drawn and in the difficulty, and left out of the input.",
    "paper profile: 'before 1st / after 3rd quartile' is read as pop <= Q1 / pop >= Q3; with strict bounds "
    "the easy class is empty whenever Q3 = 1.0 (many properties are always present).",
    "paper profile: G is drawn among n's own properties (no-context condition of the paper); Algorithm 2 "
    "line 11 would also allow context properties, which the published code never used.",
    "paper profile: node strata are [Q0,Q1], (Q1,Q3], (Q3,Q4] so that no node is in two strata.",
    "Context entries are written in the published format [kind, path, type, property, value]; their order "
    "(1-hop nodes, 1-hop edges, longer n:1 paths) approximates the published context generator.",
]

_HOP = re.compile(r"(-\[[^\]]+\]->|<-\[[^\]]+\]-)")


# Scores -------------------------------------------------------------------


@dataclass
class Scores:
    """pop(p) for the node type's properties, and raw centrality of the nodes, keyed by node key."""

    pop: dict[str, float]
    raw_cent: dict[str, float]
    pop_source: str
    cent_source: str
    _bounds: tuple[float, float] | None = None

    def bounds(self) -> tuple[float, float]:
        if self._bounds is None:
            positive = [v for v in self.raw_cent.values() if v > 0]
            if not positive:
                raise ValueError("no positive centrality values")
            self._bounds = (math.log(min(positive)), math.log(max(positive)))
        return self._bounds

    def cent(self, key: str) -> float:
        """Log-min-max normalization, as ``computeCentralityNormalized`` in the published code."""
        raw = self.raw_cent[key]
        if raw <= 0:
            return 0.0
        lo, hi = self.bounds()
        return (math.log(raw) - lo) / (hi - lo) if hi > lo else 1.0


def specificity(names: list[str], pop: dict[str, float], cent: float) -> float:
    """exp(−(1/|S|) Σ ln(pop(p) × cent)); every property here belongs to n, so cent_p(e) = cent(n)."""
    if not names:
        return 0.0
    acc = sum(math.log(pop[p] * cent) for p in names)
    return math.exp(-acc / len(names))


def _read_popularity_file(path: Path, scale: float) -> dict[str, float]:
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["prop"]: float(row["value"]) / scale for row in csv.DictReader(fh)}


def _read_centrality_file(path: Path, id_col: str, score_col: str) -> dict[str, float]:
    with open(path, encoding="utf-8", newline="") as fh:
        return {row[id_col]: float(row[score_col]) for row in csv.DictReader(fh) if row.get(id_col)}


def _excluded(cfg: TaskDefConfig, node_type: str, name: str) -> bool:
    ex = set(cfg.properties.exclude_properties)
    return name in ex or f"{node_type}.{name}" in ex


def node_properties(cfg: TaskDefConfig, graph: Graph | None, pop_file: dict[str, float] | None) -> list[str]:
    """Properties of the node type that can be drawn."""
    t = cfg.dataset.node_type
    if cfg.published.properties:
        names = list(cfg.published.properties)
    elif pop_file is not None:
        names = list(pop_file)
    else:
        names = sorted(
            {
                name
                for nid in graph.nodes_of_type(t)
                for name, value in graph.nodes[nid].props.items()
                if infer_type(value) in ("string", "integer", "float", "boolean", "date")
            }
        )
    return sorted(n for n in names if not _excluded(cfg, t, n))


def load_scores(cfg: TaskDefConfig, graph: Graph | None) -> Scores:
    pc = cfg.published
    t = cfg.dataset.node_type
    pop_file = _read_popularity_file(Path(pc.popularity_file), pc.popularity_scale) if pc.popularity_file else None
    names = node_properties(cfg, graph, pop_file)
    if pop_file is not None:
        missing = [n for n in names if n not in pop_file]
        if missing:
            raise ConfigError(f"published.popularity_file has no value for {missing}")
        pop, pop_source = {n: pop_file[n] for n in names}, f"file:{pc.popularity_file}"
    else:
        ids = graph.nodes_of_type(t)
        empty_null = cfg.properties.empty_string_is_null
        pop = {
            n: sum(not is_null(graph.nodes[i].props.get(n), empty_null) for i in ids) / len(ids) for n in names
        }
        pop_source = "computed: share of nodes of the type with a non-null value"
    pop = {n: v for n, v in pop.items() if v > 0}

    if pc.centrality_file:
        raw = _read_centrality_file(Path(pc.centrality_file), pc.centrality_id_column, pc.centrality_score_column)
        cent_source = f"file:{pc.centrality_file}"
    else:
        from .centrality import compute_centrality

        c = compute_centrality(graph, cfg.centrality)
        raw = {node_key(graph, nid, pc.node_key): c.raw[nid] for nid in graph.nodes_of_type(t)}
        cent_source = f"computed:{cfg.centrality.measure} (raw score; log-min-max normalized over the node type)"
    return Scores(pop, raw, pop_source, cent_source)


def node_key(graph: Graph, node_id: str, key_property: str) -> str:
    """The id used in the published files (e.g. ICAO), or the snapshot id when it already is that id."""
    value = graph.nodes[node_id].props.get(key_property)
    return str(value) if value not in (None, "") else node_id


# Node data ----------------------------------------------------------------


@dataclass
class NodeData:
    key: str  # e.g. ICAO
    node_id: str  # id in the snapshot (equal to key when dataset.node_id_property is the key)
    props: dict[str, Any]  # non-null properties of n
    value_types: dict[str, str]
    context: list[list]  # published-format entries, without n's own properties
    context_props: list[tuple[str, str]] = field(default_factory=list)  # (element type, property) in context(n)


def _published_path(path: str) -> tuple[str, list[str], str, int]:
    """``Airport-[OF]->City-[IN]->Country`` -> ("node", ["OF", "City", "IN"], "Country", 2 hops);
    ``Airport-[ROUTE_TO]->`` -> ("edge", ["ROUTE_TO"], "ROUTE_TO", 1 hop)."""
    tokens = [x for x in _HOP.split(path) if x]
    clean = [x.strip("-<>[]") if _HOP.fullmatch(x) else x for x in tokens]
    hops = sum(1 for x in tokens if _HOP.fullmatch(x))
    if _HOP.fullmatch(tokens[-1]):
        return "edge", clean[1:], clean[-1], hops
    return "node", clean[1:-1], clean[-1], hops


def graph_node_data(graph: Graph, cfg: TaskDefConfig, node_ids: list[str]) -> dict[str, NodeData]:
    from .context import context_elements, detect_n1, n1_keys

    n1 = n1_keys(detect_n1(graph))
    empty_null = cfg.properties.empty_string_is_null
    out = {}
    for nid in node_ids:
        node = graph.nodes[nid]
        props = {k: v for k, v in node.props.items() if not is_null(v, empty_null)}
        groups: dict[int, list[list]] = {0: [], 1: [], 2: []}
        ctx_props = set()
        for ref in context_elements(graph, nid, n1, cfg.dataset.context_max_hops):
            if ref.element_id == nid and ref.kind == "node":
                continue
            element = graph.element(ref.element_id)
            kind, rels, etype, hops = _published_path(ref.path)
            group = 2 if hops > 1 else (0 if kind == "node" else 1)
            for name in sorted(element.props):
                value = element.props[name]
                if is_null(value, empty_null):
                    continue
                groups[group].append([kind, rels, etype, name, value])
                ctx_props.add((element.type, name))
        out[nid] = NodeData(
            key=node_key(graph, nid, cfg.published.node_key),
            node_id=nid,
            props=props,
            value_types={k: node.value_type(k) for k in props},
            context=groups[0] + groups[1] + groups[2],
            context_props=sorted(ctx_props),
        )
    return out


# Sampling -----------------------------------------------------------------


@dataclass
class PublishedTask:
    node: NodeData
    node_stratum: str
    h_class: str
    g_class: str
    H: list[str]
    G_drawn: list[str]
    context: list[list]
    cent: float
    cent_raw: float
    published: dict[str, Any] = field(default_factory=dict)  # import only

    @property
    def difficulty_prop(self) -> str:
        return f"{self.h_class}{self.g_class}"


def node_strata(scores: Scores, keys: list[str], method: str) -> tuple[dict[str, list[str]], list[float]]:
    """Node keys per stratum (easy = most central). ``keys`` are the nodes that exist in the dataset."""
    import pandas as pd

    keys = sorted(k for k in keys if k in scores.raw_cent)
    values = pd.Series([scores.raw_cent[k] for k in keys], index=keys)
    if method == "terciles":
        labels = pd.qcut(values, q=3, labels=[1, 2, 3])  # same call as the published code
        strata = {
            "easy": [k for k in keys if labels[k] == 3],
            "medium": [k for k in keys if labels[k] == 2],
            "hard": [k for k in keys if labels[k] == 1],
        }
        bounds = [float(x) for x in values.quantile([0, 1 / 3, 2 / 3, 1])]
    else:
        q = [float(x) for x in np.quantile(values.to_numpy(), [0, 0.25, 0.5, 0.75, 1.0])]
        strata = {
            "hard": [k for k in keys if values[k] <= q[1]],
            "medium": [k for k in keys if q[1] < values[k] <= q[3]],
            "easy": [k for k in keys if values[k] > q[3]],
        }
        bounds = q
    return strata, bounds


def property_classes(
    scores: Scores, node: NodeData, method: str, threshold: float
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if method == "threshold":
        pools = {
            "easy": sorted(p for p, v in scores.pop.items() if v >= threshold),
            "hard": sorted(p for p, v in scores.pop.items() if v < threshold),
        }
        return pools, {"threshold": threshold}
    # Algorithm 2 line 7: quartiles of pop() over the properties of context(n) (n included).
    own = [p for p in scores.pop if p in node.props]
    values = [scores.pop[p] for p in own] + [scores.pop.get(name, np.nan) for _t, name in node.context_props]
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    q1, q3 = (float(x) for x in np.quantile(values, [0.25, 0.75]))
    pools = {
        "easy": sorted(p for p, v in scores.pop.items() if v >= q3),
        "hard": sorted(p for p, v in scores.pop.items() if v <= q1),
    }
    return pools, {"Q1": q1, "Q3": q3}


class PublishedSampler:
    def __init__(self, cfg: TaskDefConfig, scores: Scores, leak_partners=None):
        self.cfg = cfg
        self.pc = cfg.published
        self.opts = self.pc.resolved()
        self.scores = scores
        self.rng = random.Random(self.pc.seed)
        self.leak_partners = leak_partners or (lambda name: set())
        self.report: Counter = Counter()

    def draw_nodes(self, strata: dict[str, list[str]]) -> dict[str, list[str]]:
        chosen = {}
        for s in STRATA:
            pool = [k for k in strata[s] if self.scores.cent(k) > 0]
            self.report[f"excluded_zero_centrality_{s}"] += len(strata[s]) - len(pool)
            n = self.pc.nodes_per_stratum
            if len(pool) < n:
                raise ValueError(f"stratum {s!r} has {len(pool)} usable nodes, {n} needed")
            chosen[s] = self.rng.sample(pool, n)
        return chosen

    def _sample(self, pool: list[str], k: int, reason: str) -> list[str]:
        if len(pool) < k:
            self.report[reason] += 1
            k = len(pool)
        return self.rng.sample(pool, k)

    def draw_H(self, pools, cls: str, node: NodeData) -> list[str] | None:
        pool = pools[cls]
        if self.opts["hidden_requires_value"]:
            pool = [p for p in pool if p in node.props]
        if len(pool) < self.pc.size_H:
            self.report[f"skipped_short_H_pool_{cls}"] += 1
            return None
        return self.rng.sample(pool, self.pc.size_H)

    def draw_G(self, pools, cls: str, node: NodeData, H: list[str]) -> list[str] | None:
        blocked = set(H)
        if self.pc.apply_leakage_pairs:
            for h in H:
                blocked |= self.leak_partners(h)
        available = [p for p in pools[cls] if p in node.props]
        if self.pc.profile == "code" and self.opts["min_G"] < self.pc.size_G:
            # Published code: sample from class ∩ props(n), drop overlaps with H, top up from the whole class.
            drawn = self._sample(available, self.pc.size_G, f"short_G_pool_{cls}")
            G = [p for p in drawn if p not in blocked]
            topup = [p for p in pools[cls] if p not in blocked and p not in G]
            while len(G) < self.opts["min_G"] and topup:
                pick = self.rng.choice(topup)
                topup.remove(pick)
                G.append(pick)
                self.report["G_topped_up"] += 1
            if len(G) < self.opts["min_G"]:
                self.report[f"skipped_short_G_pool_{cls}"] += 1
                return None
            return G
        pool = [p for p in available if p not in blocked]
        if len(pool) < self.opts["min_G"]:
            self.report[f"skipped_short_G_pool_{cls}"] += 1
            return None
        return self.rng.sample(pool, min(self.pc.size_G, len(pool)))

    def context_slice(self, node: NodeData) -> list[list]:
        start, lo, hi = self.pc.context_slice
        return node.context[start : self.rng.randint(lo, hi)]

    def tasks_for(self, node: NodeData, stratum: str) -> list[PublishedTask]:
        pools, _info = property_classes(
            self.scores, node, self.opts["property_classes"], self.pc.easy_threshold
        )
        fixed_H = {}
        fixed_G = {}
        if not self.opts["redraw_H_per_task"]:
            for cls in ("easy", "hard"):
                fixed_H[cls] = self.draw_H(pools, cls, node)
                fixed_G[cls] = self.draw_G(pools, cls, node, fixed_H[cls] or [])
        out = []
        cent = self.scores.cent(node.key)
        for h_cls, g_cls in COMBOS:
            if self.opts["redraw_H_per_task"]:
                H = self.draw_H(pools, h_cls, node)
                G = self.draw_G(pools, g_cls, node, H or []) if H else None
            else:
                H, G = fixed_H[h_cls], fixed_G[g_cls]
                if H and G and set(H) & set(G):  # can only happen when the classes overlap
                    G = [p for p in G if p not in H] or None
            if not H or not G:
                self.report["tasks_skipped"] += 1
                continue
            out.append(
                PublishedTask(node, stratum, h_cls, g_cls, H, G, self.context_slice(node), cent, self.scores.raw_cent[node.key])
            )
        return out


# Rows ---------------------------------------------------------------------


def _task_id(dataset: str, task: PublishedTask) -> str:
    payload = {
        "dataset": dataset,
        "design": DESIGN,
        "node": task.node.key,
        "difficulty_prop": task.difficulty_prop,
        "H": task.H,
        "G": task.G_drawn,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def task_rows(cfg: TaskDefConfig, tasks: list[PublishedTask], scores: Scores) -> tuple[list[dict], list[dict]]:
    t = cfg.dataset.node_type
    rows, gt = [], []
    seen = set()
    for task in tasks:
        n = task.node
        tid = _task_id(cfg.dataset.name, task)
        if tid in seen:
            continue
        seen.add(tid)
        H = [{"prop_key": f"{t}.{p}", "property_name": p, "element_type": t, "path": t} for p in task.H]
        G = [
            {
                "prop_key": f"{t}.{p}",
                "property_name": p,
                "element_type": t,
                "element_id": n.node_id,
                "path": t,
                "value": n.props[p],
                "popularity": scores.pop.get(p),
                "centrality": task.cent,
            }
            for p in task.G_drawn
            if p in n.props  # the published input only lists G properties that have a value
        ]
        spec_h = specificity(task.H, scores.pop, task.cent)
        spec_g = specificity(task.G_drawn, scores.pop, task.cent)
        pub = task.published
        rows.append(
            {
                "task_id": tid,
                "task_set_version": cfg.version,
                "dataset": cfg.dataset.name,
                "design": f"{DESIGN}-{pub.get('origin', cfg.published.profile)}",
                "node_type": t,
                "node_id": n.key,
                "node_centrality": repr(task.cent),
                "node_centrality_raw": repr(task.cent_raw),
                "node_stratum": task.node_stratum,
                "H_stratum": task.h_class,
                "G_stratum": task.g_class,
                "difficulty_prop": task.difficulty_prop,
                "size_H": len(H),
                "size_G": len(G),
                "H": json.dumps(H, ensure_ascii=False),
                "G": json.dumps(G, ensure_ascii=False),
                "G_drawn": json.dumps(task.G_drawn, ensure_ascii=False),
                "context": json.dumps(task.context, ensure_ascii=False),
                "difficulty": repr(spec_h + spec_g),
                "specificity_H": repr(spec_h),
                "specificity_G": repr(spec_g),
                "published_difficulty": _r(pub.get("diffScore")),
                "published_specificity_H": _r(pub.get("specificityH")),
                "published_specificity_G": _r(pub.get("specificityG")),
                "published_input": pub.get("input") or published_input_text(n.props, task.G_drawn),
                "published_input_context": pub.get("input_context")
                or published_input_text(n.props, task.G_drawn, task.context),
            }
        )
        for p in task.H:
            value = n.props.get(p)
            gt.append(
                {
                    "task_id": tid,
                    "prop_key": f"{t}.{p}",
                    "value": json.dumps(value, ensure_ascii=False),
                    "value_type": n.value_types.get(p, infer_type(value)),
                }
            )
    return rows, gt


def _r(x) -> str:
    return "" if x is None else repr(x)


# Build --------------------------------------------------------------------


def _check_output(out_dir: Path, overwrite: bool) -> None:
    if (out_dir / "tasks.csv").exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds a task set. Task sets are immutable: use a new output_dir, or --overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _leak_partners(cfg: TaskDefConfig):
    from .leakage import LeakagePairs

    pairs = LeakagePairs(cfg.properties.leakage_pairs)
    t = cfg.dataset.node_type

    def partners(name: str) -> set[str]:
        related = pairs.related(f"{t}.{name}")
        return {k.split(".", 1)[1] for k in related if k.startswith(f"{t}.")}

    return partners


def build_published(cfg: TaskDefConfig, refresh_snapshot: bool = False, overwrite: bool = False) -> dict:
    from .pipeline import code_version, load_snapshot

    out_dir = Path(cfg.output_dir)
    _check_output(out_dir, overwrite)
    graph, source_info = load_snapshot(cfg, refresh=refresh_snapshot)
    scores = load_scores(cfg, graph)
    t = cfg.dataset.node_type
    ids = graph.nodes_of_type(t)
    by_key = {node_key(graph, nid, cfg.published.node_key): nid for nid in ids}
    strata, bounds = node_strata(scores, list(by_key), cfg.published.resolved()["node_strata"])
    sampler = PublishedSampler(cfg, scores, _leak_partners(cfg))
    chosen = sampler.draw_nodes(strata)
    data = graph_node_data(graph, cfg, [by_key[k] for keys in chosen.values() for k in keys])
    by_node_key = {d.key: d for d in data.values()}
    tasks = [task for s in STRATA for k in chosen[s] for task in sampler.tasks_for(by_node_key[k], s)]
    rows, gt = task_rows(cfg, tasks, scores)
    write_csv(out_dir / "tasks.csv", PUBLISHED_TASK_COLUMNS, rows)
    write_csv(out_dir / "ground_truth.csv", ["task_id", "prop_key", "value", "value_type"], gt)
    manifest = _manifest(cfg, scores, rows, gt, sampler.report, code_version())
    manifest.update(
        {
            "mode": "build-published",
            "snapshot": {"hash": graph.content_hash(), **source_info, "stats": graph.stats()},
            "node_strata": {"method": cfg.published.resolved()["node_strata"], "bounds_raw_centrality": bounds, "sizes": {s: len(v) for s, v in strata.items()}},
            "nodes_drawn": chosen,
        }
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _manifest(cfg, scores, rows, gt, report, code_version) -> dict:
    opts = cfg.published.resolved()
    pools_threshold = {
        "easy": sorted(p for p, v in scores.pop.items() if v >= cfg.published.easy_threshold),
        "hard": sorted(p for p, v in scores.pop.items() if v < cfg.published.easy_threshold),
    }
    return {
        "design": DESIGN,
        "task_set_version": cfg.version,
        "dataset": cfg.dataset.name,
        "node_type": cfg.dataset.node_type,
        "profile": cfg.published.profile,
        "options": opts,
        "config": cfg.as_dict(),
        "code_version": code_version,
        "package_version": __version__,
        "popularity": {"source": scores.pop_source, "values": scores.pop},
        "property_classes_threshold": pools_threshold if opts["property_classes"] == "threshold" else None,
        "centrality": {"source": scores.cent_source, "log_min": scores.bounds()[0], "log_max": scores.bounds()[1]},
        "summary": {
            "tasks": len(rows),
            "per_node_stratum": dict(Counter(r["node_stratum"] for r in rows)),
            "per_difficulty_prop": dict(Counter(r["difficulty_prop"] for r in rows)),
            "size_G_given": dict(Counter(int(r["size_G"]) for r in rows)),
            "size_G_drawn": dict(Counter(len(json.loads(r["G_drawn"])) for r in rows)),
            "H_properties_without_ground_truth": sum(1 for g in gt if g["value"] == "null"),
        },
        "report": dict(report),
        "deviations": DEVIATIONS,
    }


# Published prompt input ---------------------------------------------------

_INPUT_TEMPLATE = """
    Here is the entity extracted from the database that you should seek information about:

        input: [INPUT]

        context: [NO_CONTEXT_ADDED].
    """
_CONTEXT_LEAD = "Here are some relationships and relationships information about this property in the graph:"


def with_context(no_context_input: str, context: list) -> str:
    """The published with-context input (``transformResultsToGetContext`` in context_main.ipynb)."""
    return no_context_input.replace("[NO_CONTEXT_ADDED]", str(context)).replace("context:", _CONTEXT_LEAD)


def published_input_text(props: dict, given: list[str], context: list | None = None) -> str:
    """The user message of the published run: G as ``prop:value`` pairs in the node's property order."""
    data = ",".join(f"{k}:{v}" for k, v in props.items() if k in given)
    text = _INPUT_TEMPLATE.replace("[INPUT]", data)
    return with_context(text, context) if context is not None else text


def read_node_values(path: Path, key_column: str) -> dict[str, tuple[dict, dict]]:
    """A CSV with one row per node (the published ``ground_truth.csv``): key column + one column per property.

    Values are read back as numbers when they parse as numbers; empty cells are missing values.
    """
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = row.get(key_column)
            if not key:
                continue
            props, types = {}, {}
            for name, raw in row.items():
                if raw is None or raw.strip() == "":
                    continue
                value = _parse_scalar(raw)
                props[name] = value
                types[name] = infer_type(value)
            out[key] = (props, types)
    return out


def _parse_scalar(raw: str):
    text = raw.strip()
    try:
        return int(text) if re.fullmatch(r"-?\d+", text) else float(text)
    except ValueError:
        return text


# Import -------------------------------------------------------------------


def _num(x) -> float | None:
    """Published difficulty fields are strings with decimal commas ('6,57…')."""
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x).replace(",", "."))


def parse_published_input(text: str) -> dict[str, str]:
    """``input: altitude:154.0,city:Koyuk`` -> {"altitude": "154.0", "city": "Koyuk"} (fallback only)."""
    m = re.search(r"input:\s*(.*)", text)
    if not m:
        return {}
    out = {}
    for part in m.group(1).strip().split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def read_published_tasks(path: Path) -> list[tuple[str, list[str], list[str], dict]]:
    csv.field_size_limit(1 << 30)
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"{path} row {i + 1}: expected 4 columns (node, H, G, metadata), got {len(row)}")
            node, H, G, meta = row
            rows.append((node, ast.literal_eval(H), ast.literal_eval(G), ast.literal_eval(meta)))
    return rows


def read_entities_context(path: Path) -> dict[str, tuple[dict, list]]:
    """The published ``context_entities.csv`` (columns key, value): own properties have an empty path."""
    import pandas as pd

    df = pd.read_csv(path)
    out = {}
    for key, value in zip(df["key"], df["value"]):
        entries = (_loads(value) if isinstance(value, str) else value) or []
        own = {e[3]: e[4] for e in entries if e[0] == "node" and list(e[1]) == []}
        out[str(key)] = (own, entries[len(own) :])
    return out


def _loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def import_published(
    cfg: TaskDefConfig,
    results_path: Path,
    entities_context: Path | None = None,
    node_values: Path | None = None,
    use_graph: bool = True,
    overwrite: bool = False,
    refresh_snapshot: bool = False,
) -> dict:
    """Convert the published task file into tasks.csv and ground_truth.csv."""
    from .pipeline import code_version, load_snapshot

    out_dir = Path(cfg.output_dir)
    _check_output(out_dir, overwrite)
    published = read_published_tasks(results_path)
    graph, source_info = (load_snapshot(cfg, refresh=refresh_snapshot) if use_graph else (None, {"source": "none"}))
    scores = load_scores(cfg, graph) if graph is not None or (cfg.published.popularity_file and cfg.published.centrality_file) else None
    if scores is None:
        raise ConfigError(
            "without the graph, set published.popularity_file and published.centrality_file "
            "(datasets/ranks/rank_properties.csv and ranks_pagerank.csv of the published repo)"
        )
    node_source = "published input (values parsed from text; no ground truth)"
    own_values: dict[str, tuple[dict, dict]] = {}
    if node_values:
        own_values = read_node_values(node_values, cfg.published.node_key)
        node_source = f"node values file: {node_values}"
    elif entities_context:
        for key, (own, _ctx) in read_entities_context(entities_context).items():
            own_values[key] = (own, {k: infer_type(v) for k, v in own.items()})
        node_source = f"entities context file: {entities_context}"
    elif graph is not None:
        t = cfg.dataset.node_type
        empty_null = cfg.properties.empty_string_is_null
        for nid in graph.nodes_of_type(t):
            node = graph.nodes[nid]
            props = {k: v for k, v in node.props.items() if not is_null(v, empty_null)}
            own_values[node_key(graph, nid, cfg.published.node_key)] = (props, {k: node.value_type(k) for k in props})
        node_source = f"graph snapshot ({source_info.get('source')})"

    rng = random.Random(cfg.published.seed)
    start, lo, hi = cfg.published.context_slice
    report: Counter = Counter()
    tasks = []
    for key, H, G, meta in published:
        if key in own_values:
            props, types = own_values[key]
        else:
            props = parse_published_input(meta.get("input", ""))
            types = {k: "string" for k in props}
            report["nodes_without_values"] += 1
        full_context = meta.get("context") or []
        node = NodeData(key=key, node_id=key, props=props, value_types=types, context=full_context)
        given_in_input = set(parse_published_input(meta.get("input", "")))
        expected = {p for p in G if p in props}
        if key in own_values and given_in_input != expected:
            report["G_input_mismatch"] += 1
            log.warning("%s %s: input gives %s, node values give %s", key, meta.get("difficultyProp"), sorted(given_in_input), sorted(expected))
        ctx = full_context[start : rng.randint(lo, hi)]  # published: metadata['context'][2:random.randint(5,10)]
        diff_prop = meta.get("difficultyProp", "")
        h_cls, g_cls = diff_prop[:4], diff_prop[4:]
        stratum = meta.get("entityDiff:") or meta.get("entityDiff")
        if key not in scores.raw_cent:
            raise ValueError(f"node {key} has no centrality in {scores.cent_source}")
        tasks.append(
            PublishedTask(
                node=node,
                node_stratum=stratum,
                h_class=h_cls,
                g_class=g_cls,
                H=list(H),
                G_drawn=list(G),
                context=ctx,
                cent=scores.cent(key),
                cent_raw=scores.raw_cent[key],
                published={
                    "origin": "import",
                    "diffScore": _num(meta.get("diffScore")),
                    "specificityH": _num(meta.get("specificityH")),
                    "specificityG": _num(meta.get("specificityG")),
                    "input": meta.get("input", ""),
                    "input_context": with_context(meta.get("input", ""), ctx),
                },
            )
        )
    rows, gt = task_rows(cfg, tasks, scores)
    if len(rows) != len(tasks):
        report["duplicate_tasks_dropped"] += len(tasks) - len(rows)
    diffs = [
        abs(float(r["difficulty"]) - float(r["published_difficulty"])) for r in rows if r["published_difficulty"]
    ]
    write_csv(out_dir / "tasks.csv", PUBLISHED_TASK_COLUMNS, rows)
    has_values = bool(own_values)
    if has_values:
        write_csv(out_dir / "ground_truth.csv", ["task_id", "prop_key", "value", "value_type"], gt)
    manifest = _manifest(cfg, scores, rows, gt, report, code_version())
    manifest.update(
        {
            "mode": "import-published",
            "source_file": {"path": str(results_path), "sha256": _sha256(results_path)},
            "node_values_source": node_source,
            "ground_truth_written": has_values,
            "difficulty_check": {
                "max_abs_difference_vs_published": max(diffs) if diffs else None,
                "tasks_compared": len(diffs),
            },
            "context_note": "The published file stores each node's full context; the with-context slice "
            f"entries[{start}:randint({lo},{hi})] is drawn here with published.seed, because the slices used "
            "in the published with-context run are not in this file.",
        }
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

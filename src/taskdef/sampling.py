"""Task sampling: Algorithm 2 of the draft (TASKDEFINITION_SPEC.md §4.6, §5).

Three independent stratifications:
- nodes by ``cent(n)``;
- H and G properties by one partition of ``pop(p)`` over N ∪ context(N)
  (``sampling.G_strata_by: popularity``, the default), or H by ``pop(p)`` and
  G by ``pop(p) * cent_p(e)`` (``score``).

Random draws are made by ordering candidates by a hash of
``(seed, salt, candidate id)``. This is a uniform random order that does
not depend on the other candidates, so adding or removing a candidate
never changes which of the remaining ones are drawn ahead of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .centrality import Centrality
from .config import PropertiesConfig, TaskDefConfig
from .context import ContextRef, N1Key, context_elements, prop_key
from .difficulty import Difficulty, compute_difficulty
from .graph import SCALAR_TYPES, Element, Graph, is_null
from .leakage import LeakagePairs, value_leaks
from .popularity import Popularity

log = logging.getLogger(__name__)

STRATA = ("low", "mid", "high")


# -- strata ---------------------------------------------------------------------


@dataclass(frozen=True)
class Quartiles:
    q: tuple[float, float, float, float, float]

    @classmethod
    def of(cls, values: Iterable[float]) -> "Quartiles":
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            raise ValueError("Cannot compute quartiles of an empty set")
        return cls(tuple(float(x) for x in np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])))

    def stratum(self, value: float) -> str:
        """Half-open quartiles, as in the draft (§3.3): low = [Q0, Q1),
        mid = Q2 ∪ Q3 = [Q1, Q3), high = [Q3, Q4].

        Ties at the bottom (Q0 == Q1, e.g. many elements share the minimum
        PageRank) would leave ``low`` empty; values equal to Q0 are then put
        in ``low``.
        """
        q0, q1, _q2, q3, _q4 = self.q
        if value < q1 or (q0 == q1 and value <= q0):
            return "low"
        if value < q3:
            return "mid"
        return "high"


@dataclass(frozen=True)
class Partitions:
    """Quartile boundaries computed before sampling (Algorithm 2 takes them as input)."""

    node: Quartiles
    h: Quartiles
    g: Quartiles


# -- candidates -----------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    prop_key: str
    property_name: str
    element_id: str
    element_kind: str
    element_type: str
    path: str
    value: Any
    value_type: str
    popularity: float
    centrality: float

    @property
    def score(self) -> float:
        return self.popularity * self.centrality

    @property
    def uid(self) -> str:
        return f"{self.prop_key}@{self.element_id}"


class Eligibility:
    """§4.6: exclusions, never_hide, scalar values only, non-null values only."""

    def __init__(self, cfg: PropertiesConfig):
        self.exclude = set(cfg.exclude_properties)
        self.never_hide = set(cfg.never_hide)
        self.empty_string_is_null = cfg.empty_string_is_null

    @staticmethod
    def _matches(entries: set[str], element_type: str, name: str, key: str) -> bool:
        return name in entries or f"{element_type}.{name}" in entries or key in entries

    def excluded(self, element_type: str, name: str, key: str) -> bool:
        return self._matches(self.exclude, element_type, name, key)

    def hideable(self, element_type: str, name: str, key: str) -> bool:
        return not self._matches(self.never_hide, element_type, name, key)

    def usable(self, element: Element, name: str, key: str) -> bool:
        if self.excluded(element.type, name, key):
            return False
        if element.value_type(name) not in SCALAR_TYPES:
            return False
        return not is_null(element.props.get(name), self.empty_string_is_null)


# -- tasks ----------------------------------------------------------------------


@dataclass
class Task:
    task_id: str
    dataset: str
    node_type: str
    node_id: str
    node_centrality: float
    node_stratum: str
    H_stratum: str
    G_stratum: str
    size_H: int
    size_G: int
    H: list[Candidate]
    G: list[Candidate]
    difficulty: Difficulty
    node_centrality_raw: float | None = None
    G_candidates: int = 0  # eligible G properties of n ∪ context(n) (all strata), METRICS.md §3.1


def make_task_id(dataset: str, node_id: str, H: list[Candidate], G: list[Candidate]) -> str:
    payload = {
        "dataset": dataset,
        "node_id": node_id,
        "H": sorted(c.prop_key for c in H),
        # A prop_key can come from several elements (non-n:1 neighbors), so G
        # is identified by (prop_key, element_id).
        "G": sorted([c.prop_key, c.element_id] for c in G),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _rank(seed: int, salt: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{salt}|{key}".encode()).hexdigest()


def stable_order(items: Iterable, seed: int, salt: str, key: Callable[[Any], str]) -> list:
    return sorted(items, key=lambda item: _rank(seed, salt, key(item)))


@dataclass
class BuildResult:
    tasks: list[Task]
    node_quartiles: Quartiles
    h_quartiles: Quartiles
    g_quartiles: Quartiles
    report: dict
    strata_summary: dict
    context_rows: list[dict] = field(default_factory=list)


class TaskBuilder:
    def __init__(
        self,
        graph: Graph,
        cfg: TaskDefConfig,
        centrality: Centrality,
        popularity: Popularity,
        n1: set[N1Key],
    ):
        self.graph = graph
        self.cfg = cfg
        self.cent = centrality
        self.pop = popularity
        self.n1 = n1
        self.node_type = cfg.dataset.node_type
        self.max_hops = cfg.dataset.context_max_hops
        self.seed = cfg.sampling.seed
        self.elig = Eligibility(cfg.properties)
        self.leak_pairs = LeakagePairs(cfg.properties.leakage_pairs)
        self.min_substring = cfg.properties.leak_min_substring_len
        self.N = graph.nodes_of_type(self.node_type)
        if not self.N:
            raise ValueError(f"No nodes of type {self.node_type!r} in the snapshot")
        self.node_set = set(self.N)

    # -- context ------------------------------------------------------------

    def _refs(self, node_id: str) -> list[ContextRef]:
        return context_elements(self.graph, node_id, self.n1, self.max_hops)

    def candidates(self, node_id: str, refs: list[ContextRef] | None = None) -> list[Candidate]:
        """Every usable property occurrence in ``n ∪ context(n)``."""
        out = []
        for ref in refs if refs is not None else self._refs(node_id):
            element = self.graph.element(ref.element_id)
            for name in sorted(element.props):
                key = prop_key(ref.path, name)
                if not self.elig.usable(element, name, key):
                    continue
                out.append(self._candidate(element, ref, name, key))
        return out

    def _candidate(self, element: Element, ref: ContextRef, name: str, key: str) -> Candidate:
        return Candidate(
            prop_key=key,
            property_name=name,
            element_id=element.id,
            element_kind=element.kind,
            element_type=element.type,
            path=ref.path,
            value=element.props[name],
            value_type=element.value_type(name),
            popularity=self.pop(element.kind, element.type, name),
            centrality=self.cent[element.id],
        )

    def _context_rows(self, node_id: str, refs: list[ContextRef]) -> list[dict]:
        rows = []
        for ref in refs:
            element = self.graph.element(ref.element_id)
            for name in sorted(element.props):
                key = prop_key(ref.path, name)
                rows.append(
                    {
                        "node_id": node_id,
                        "prop_key": key,
                        "property_name": name,
                        "element_id": element.id,
                        "element_kind": element.kind,
                        "element_type": element.type,
                        "path": ref.path,
                        "value": json.dumps(element.props[name], ensure_ascii=False),
                        "value_type": element.value_type(name),
                        "popularity": self.pop(element.kind, element.type, name),
                        "centrality": self.cent[element.id],
                        "eligible": self.elig.usable(element, name, key),
                    }
                )
        return rows

    def _is_own(self, c: Candidate, node_id: str) -> bool:
        return c.element_id == node_id and c.element_kind == "node" and c.path == self.node_type

    # -- quartiles ----------------------------------------------------------

    def h_property_scores(self) -> dict[str, float]:
        """Distinct H-eligible properties of the node type, each counted once: name -> pop(p)."""
        scalar_props: set[str] = set()
        for nid in self.N:
            node = self.graph.nodes[nid]
            for name in node.props:
                if node.value_type(name) in SCALAR_TYPES and not is_null(
                    node.props[name], self.elig.empty_string_is_null
                ):
                    scalar_props.add(name)
        scores = {}
        for name in sorted(scalar_props):
            key = prop_key(self.node_type, name)
            if self.elig.excluded(self.node_type, name, key) or not self.elig.hideable(self.node_type, name, key):
                continue
            scores[name] = self.pop("node", self.node_type, name)
        return scores

    def g_occurrence_scores(self) -> list[float]:
        """``pop(p) * cent_p(e)`` over the set of elements in N ∪ context(N), each element once.

        An element can be reached through several paths; exclusions given as
        full prop_keys are checked against the first path found.
        """
        first_ref: dict[str, ContextRef] = {}
        for nid in self.N:
            for ref in self._refs(nid):
                first_ref.setdefault(ref.element_id, ref)
        if not self.cfg.sampling.use_context:  # E = N: the nodes' own properties only
            first_ref = {eid: ref for eid, ref in first_ref.items() if eid in self.node_set and ref.path == self.node_type}
        scores = []
        distinct: dict[tuple[str, str, str], float] = {}
        for eid in sorted(first_ref):
            ref = first_ref[eid]
            element = self.graph.element(eid)
            for name in sorted(element.props):
                if self.elig.usable(element, name, prop_key(ref.path, name)):
                    pop = self.pop(element.kind, element.type, name)
                    scores.append(pop * self.cent[eid])
                    distinct[(element.kind, element.type, name)] = pop
        if self.cfg.sampling.G_strata_by == "popularity":
            return [distinct[k] for k in sorted(distinct)]
        return scores

    def g_value(self, c: Candidate) -> float:
        """The value G strata are drawn on (``sampling.G_strata_by``)."""
        return c.popularity if self.cfg.sampling.G_strata_by == "popularity" else c.score

    def partitions(self) -> Partitions:
        h_scores = self.h_property_scores()
        if not h_scores:
            raise ValueError("No H-eligible properties on the node type; check properties.exclude_properties/never_hide")
        node = Quartiles.of(self.cent[n] for n in self.N)
        if self.cfg.sampling.G_strata_by == "popularity":
            # One partition for H and G, as the draft's single set of property-score pairs.
            shared = Quartiles.of(self.g_occurrence_scores())
            return Partitions(node=node, h=shared, g=shared)
        return Partitions(node=node, h=Quartiles.of(h_scores.values()), g=Quartiles.of(self.g_occurrence_scores()))

    # -- build --------------------------------------------------------------

    def build(self, partitions: Partitions | None = None) -> BuildResult:
        """Algorithm 2. ``partitions`` defaults to the quartiles of this graph;
        passing fixed ones keeps the strata unchanged when nodes are added or removed."""
        s = self.cfg.sampling
        dataset = self.cfg.dataset.name

        parts = partitions or self.partitions()
        node_q, h_q, g_q = parts.node, parts.h, parts.g
        h_scores = self.h_property_scores()
        g_scores = self.g_occurrence_scores()

        node_pools = {st: [n for n in self.N if node_q.stratum(self.cent[n]) == st] for st in STRATA}
        strata_summary = {
            "node": {st: len(node_pools[st]) for st in STRATA},
            "H_property": {
                st: sorted(name for name, v in h_scores.items() if h_q.stratum(v) == st) for st in STRATA
            },
            "H_property_popularity": dict(sorted(h_scores.items())),
            "G_occurrence": {st: sum(1 for v in g_scores if g_q.stratum(v) == st) for st in STRATA},
        }

        report: dict = {
            "expected_before_skips": 27 * s.max_n * len(s.H_sizes) * len(s.G_sizes),
            "sizes_H": s.H_sizes,
            "sizes_G": s.G_sizes,
            "use_context": s.use_context,
            "G_strata_by": s.G_strata_by,
            "nodes_available": {st: len(node_pools[st]) for st in STRATA},
            "nodes_selected": {},
            "skipped": [],
            "duplicates_removed": 0,
        }

        tasks: list[Task] = []
        seen_ids: set[str] = set()
        context_rows: list[dict] = []

        for node_stratum in STRATA:
            pool = node_pools[node_stratum]
            selected = stable_order(pool, self.seed, f"node|{node_stratum}", key=str)[: s.max_n]
            if len(selected) < s.max_n:
                log.warning("Node stratum %s has %d nodes, fewer than max_n=%d", node_stratum, len(pool), s.max_n)
            report["nodes_selected"][node_stratum] = sorted(selected)

            for node_id in sorted(selected):
                refs = self._refs(node_id)
                context_rows.extend(self._context_rows(node_id, refs))
                cands = self.candidates(node_id, refs)
                if not s.use_context:
                    cands = [c for c in cands if self._is_own(c, node_id)]
                own = [
                    c
                    for c in cands
                    if self._is_own(c, node_id) and self.elig.hideable(c.element_type, c.property_name, c.prop_key)
                ]
                h_pools = {st: [c for c in own if h_q.stratum(c.popularity) == st] for st in STRATA}
                node_cent = self.cent[node_id]

                for size_h in s.H_sizes:
                    for size_g in s.G_sizes:
                        for h_stratum in STRATA:
                            combo = {
                                "node_id": node_id,
                                "node_stratum": node_stratum,
                                "size_H": size_h,
                                "size_G": size_g,
                                "H_stratum": h_stratum,
                            }
                            h_pool = h_pools[h_stratum]
                            if len(h_pool) < size_h:
                                for g_stratum in STRATA:
                                    report["skipped"].append(
                                        {**combo, "G_stratum": g_stratum, "reason": "H_insufficient", "available": len(h_pool)}
                                    )
                                continue
                            H = stable_order(
                                h_pool, self.seed, f"H|{node_id}|{size_h}|{size_g}|{h_stratum}", key=lambda c: c.uid
                            )[:size_h]
                            H = sorted(H, key=lambda c: c.prop_key)
                            blocked = self.leak_pairs.blocked_by(c.prop_key for c in H)
                            g_base = [
                                c
                                for c in cands
                                if c.prop_key not in blocked
                                and not any(value_leaks(h.value, c.value, self.min_substring) for h in H)
                            ]

                            for g_stratum in STRATA:
                                g_pool = [c for c in g_base if g_q.stratum(self.g_value(c)) == g_stratum]
                                ordered = stable_order(
                                    g_pool,
                                    self.seed,
                                    f"G|{node_id}|{size_h}|{size_g}|{h_stratum}|{g_stratum}",
                                    key=lambda c: c.uid,
                                )
                                G: list[Candidate] = []
                                used_keys: set[str] = set()
                                for c in ordered:
                                    if c.prop_key in used_keys:
                                        continue
                                    G.append(c)
                                    used_keys.add(c.prop_key)
                                    if len(G) == size_g:
                                        break
                                if len(G) < size_g:
                                    report["skipped"].append(
                                        {**combo, "G_stratum": g_stratum, "reason": "G_insufficient", "available": len(used_keys)}
                                    )
                                    continue
                                G = sorted(G, key=lambda c: (c.prop_key, c.element_id))
                                task_id = make_task_id(dataset, node_id, H, G)
                                if task_id in seen_ids:
                                    report["duplicates_removed"] += 1
                                    continue
                                seen_ids.add(task_id)
                                tasks.append(
                                    Task(
                                        task_id=task_id,
                                        dataset=dataset,
                                        node_type=self.node_type,
                                        node_id=node_id,
                                        node_centrality=node_cent,
                                        node_stratum=node_stratum,
                                        H_stratum=h_stratum,
                                        G_stratum=g_stratum,
                                        size_H=size_h,
                                        size_G=size_g,
                                        H=H,
                                        G=G,
                                        difficulty=compute_difficulty(
                                            node_cent, [c.score for c in H], [c.score for c in G]
                                        ),
                                        node_centrality_raw=self.cent.raw.get(node_id),
                                        G_candidates=len(cands),
                                    )
                                )

        report.update(_summarize(tasks, report["skipped"]))
        return BuildResult(tasks, node_q, h_q, g_q, report, strata_summary, context_rows)


def _summarize(tasks: list[Task], skipped: list[dict]) -> dict:
    per_combo = Counter((t.node_stratum, t.H_stratum, t.G_stratum, t.size_H, t.size_G) for t in tasks)
    skipped_by_reason = Counter(s["reason"] for s in skipped)
    skipped_by_node_h = defaultdict(int)
    for item in skipped:
        skipped_by_node_h[f"{item['node_stratum']}|{item['H_stratum']}"] += 1
    return {
        "tasks_total": len(tasks),
        "tasks_per_combination": [
            {"node_stratum": k[0], "H_stratum": k[1], "G_stratum": k[2], "size_H": k[3], "size_G": k[4], "tasks": v}
            for k, v in sorted(per_combo.items())
        ],
        "tasks_per_node_stratum": dict(sorted(Counter(t.node_stratum for t in tasks).items())),
        "tasks_per_H_stratum": dict(sorted(Counter(t.H_stratum for t in tasks).items())),
        "tasks_per_G_stratum": dict(sorted(Counter(t.G_stratum for t in tasks).items())),
        "skipped_total": len(skipped),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "skipped_by_node_and_H_stratum": dict(sorted(skipped_by_node_h.items())),
        "zero_difficulty": sum(1 for t in tasks if t.difficulty.value == 0.0),
    }

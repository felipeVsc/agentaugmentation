"""Element centrality, normalized to (0, 1] (TASKDEFINITION_SPEC.md §4.2)."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import CentralityConfig
from .graph import Graph

log = logging.getLogger(__name__)


def pagerank(graph: Graph, directed: bool, damping: float, max_iter: int, tol: float) -> dict[str, float]:
    node_ids = sorted(graph.nodes)
    n = len(node_ids)
    if n == 0:
        return {}
    index = {nid: i for i, nid in enumerate(node_ids)}
    src = np.fromiter((index[e.src] for e in graph.edges.values()), dtype=np.int64, count=len(graph.edges))
    dst = np.fromiter((index[e.dst] for e in graph.edges.values()), dtype=np.int64, count=len(graph.edges))
    if not directed:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
    out_degree = np.bincount(src, minlength=n).astype(float)
    dangling = out_degree == 0
    rank = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        contrib = np.bincount(dst, weights=rank[src] / out_degree[src], minlength=n) if len(src) else np.zeros(n)
        new_rank = (1.0 - damping) / n + damping * (contrib + rank[dangling].sum() / n)
        converged = np.abs(new_rank - rank).sum() < tol * n
        rank = new_rank
        if converged:
            break
    else:
        log.warning("PageRank did not converge in %d iterations", max_iter)
    return {nid: float(rank[i]) for nid, i in index.items()}


def degree(graph: Graph) -> dict[str, float]:
    return {
        nid: float(len(graph.out_edges.get(nid, ())) + len(graph.in_edges.get(nid, ())))
        for nid in graph.nodes
    }


class Centrality:
    """Raw and normalized centrality of every node and edge."""

    def __init__(self, raw: dict[str, float], normalized: dict[str, float]):
        self.raw = raw
        self.normalized = normalized

    def __getitem__(self, element_id: str) -> float:
        return self.normalized[element_id]

    def to_frame(self, graph: Graph) -> pd.DataFrame:
        rows = []
        for eid in sorted(self.normalized):
            element = graph.element(eid)
            rows.append(
                {
                    "element_id": eid,
                    "element_kind": element.kind,
                    "element_type": element.type,
                    "raw_centrality": self.raw.get(eid),
                    "centrality": self.normalized[eid],
                }
            )
        return pd.DataFrame(rows, columns=["element_id", "element_kind", "element_type", "raw_centrality", "centrality"])


def compute_centrality(graph: Graph, cfg: CentralityConfig) -> Centrality:
    if cfg.measure == "pagerank":
        raw = pagerank(graph, cfg.directed, cfg.damping, cfg.max_iter, cfg.tol)
    elif cfg.measure == "degree":
        raw = degree(graph)
    else:  # validated in config
        raise ValueError(cfg.measure)

    # Normalize per element type: divide by the max of the type; 0 -> epsilon.
    by_type: dict[str, list[str]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        by_type[node.type].append(nid)
    normalized: dict[str, float] = {}
    for ids in by_type.values():
        top = max(raw[i] for i in ids)
        for i in ids:
            value = raw[i] / top if top > 0 else 0.0
            normalized[i] = min(1.0, max(cfg.epsilon, value))

    # Edge centrality: mean of the normalized centralities of its endpoints.
    edge_raw: dict[str, float] = {}
    for eid, edge in graph.edges.items():
        value = (normalized[edge.src] + normalized[edge.dst]) / 2.0
        edge_raw[eid] = value
        normalized[eid] = min(1.0, max(cfg.epsilon, value))

    return Centrality({**raw, **edge_raw}, normalized)

"""Node context and property identity (TASKDEFINITION_SPEC.md §4.3, §4.4).

``context(n)`` contains the properties of:
1. ``n`` itself;
2. its direct neighbor nodes (both directions);
3. the edges between ``n`` and those neighbors;
4. elements on outgoing paths from ``n`` where every relationship is n:1,
   up to ``context_max_hops`` relationships.

Paths are written ``Airport-[IN]->City-[IN]->Country`` (outgoing) or
``Airport<-[HAS]-Country`` (incoming); an edge path ends at the arrow
(``Airport-[HAS_ROUTE]->``). ``prop_key = <path>.<property_name>``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .graph import Graph

N1Key = tuple[str, str, str]  # (source type, relationship type, target type)


@dataclass(frozen=True)
class ContextRef:
    element_id: str
    kind: str  # "node" | "edge"
    path: str


def detect_n1(graph: Graph) -> dict[N1Key, dict]:
    """For each (A, R, B): n:1 if every A node has at most one outgoing R to a B node."""
    per_source: Counter = Counter()
    edge_count: Counter = Counter()
    sources: dict[N1Key, set] = defaultdict(set)
    for edge in graph.edges.values():
        key = (graph.nodes[edge.src].type, edge.type, graph.nodes[edge.dst].type)
        per_source[(key, edge.src)] += 1
        edge_count[key] += 1
        sources[key].add(edge.src)
    max_out: dict[N1Key, int] = defaultdict(int)
    for (key, _src), count in per_source.items():
        max_out[key] = max(max_out[key], count)
    return {
        key: {"n1": max_out[key] <= 1, "max_out_degree": max_out[key], "edges": edge_count[key], "sources": len(sources[key])}
        for key in sorted(edge_count)
    }


def n1_keys(detected: dict[N1Key, dict]) -> set[N1Key]:
    return {key for key, info in detected.items() if info["n1"]}


def context_elements(graph: Graph, node_id: str, n1: set[N1Key], max_hops: int) -> list[ContextRef]:
    node = graph.nodes[node_id]
    t = node.type
    refs = [ContextRef(node_id, "node", t)]

    for eid in graph.out_edges.get(node_id, ()):
        edge = graph.edges[eid]
        neighbor = graph.nodes[edge.dst]
        edge_path = f"{t}-[{edge.type}]->"
        refs.append(ContextRef(eid, "edge", edge_path))
        if edge.dst != node_id:
            refs.append(ContextRef(neighbor.id, "node", edge_path + neighbor.type))

    for eid in graph.in_edges.get(node_id, ()):
        edge = graph.edges[eid]
        if edge.src == node_id:  # self-loop, already listed as outgoing
            continue
        neighbor = graph.nodes[edge.src]
        edge_path = f"{t}<-[{edge.type}]-"
        refs.append(ContextRef(eid, "edge", edge_path))
        refs.append(ContextRef(neighbor.id, "node", edge_path + neighbor.type))

    def extend(current_id: str, path: str, visited: frozenset, hops: int) -> None:
        if hops >= max_hops:
            return
        current = graph.nodes[current_id]
        for eid in graph.out_edges.get(current_id, ()):
            edge = graph.edges[eid]
            nxt = graph.nodes[edge.dst]
            if (current.type, edge.type, nxt.type) not in n1 or nxt.id in visited:
                continue
            edge_path = f"{path}-[{edge.type}]->"
            refs.append(ContextRef(eid, "edge", edge_path))
            refs.append(ContextRef(nxt.id, "node", edge_path + nxt.type))
            extend(nxt.id, edge_path + nxt.type, visited | {nxt.id}, hops + 1)

    # Hop 1 is already covered by the direct neighbors; continue n:1 paths from there.
    for eid in graph.out_edges.get(node_id, ()):
        edge = graph.edges[eid]
        neighbor = graph.nodes[edge.dst]
        if (t, edge.type, neighbor.type) in n1 and edge.dst != node_id:
            extend(neighbor.id, f"{t}-[{edge.type}]->{neighbor.type}", frozenset({node_id, neighbor.id}), 1)

    return refs


def prop_key(path: str, property_name: str) -> str:
    return f"{path}.{property_name}"

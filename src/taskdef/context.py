"""Node context and property identity (TASKDEFINITION_SPEC.md §4.3, §4.4).

Follows ``get_node_context`` in ``context_gen/Comparing-Nodes/findContextOfNode.py``
(the context definition of [1]); ``context(n)`` contains the properties of:
1. ``n`` itself;
2. its direct neighbor nodes (both directions);
3. the edges between ``n`` and those neighbors;
4. elements on outgoing paths from ``n`` whose relationship types are many-to-one,
   expanded breadth-first with each node visited once (APOC ``NODE_GLOBAL``),
   up to ``context_max_hops`` relationships (no limit when None, as in the script).

Each element is listed once, with its shortest path from ``n`` (the script keeps
the shortest provenance per element and property; on a tie, the first found).

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


def cardinality(min_out: int, max_out: int, min_in: int, max_in: int) -> str:
    """The classification of ``detect_relationship_cardinalities`` in
    ``context_gen/Comparing-Nodes/collect_indicators.py``, in the same order."""
    if min_out >= 1 and max_out <= 1 and min_in >= 1 and max_in <= 1:
        return "one-to-one"
    if max_in <= 1:
        return "one-to-many"
    if max_out <= 1:
        return "many-to-one"
    return "many-to-many"


def detect_n1(graph: Graph) -> dict[N1Key, dict]:
    """Cardinality of each (A, R, B). Out-degrees are counted over every A node and
    in-degrees over every B node, including those without an R (degree 0)."""
    out_deg: Counter = Counter()
    in_deg: Counter = Counter()
    edge_count: Counter = Counter()
    sources: dict[N1Key, set] = defaultdict(set)
    for edge in graph.edges.values():
        key = (graph.nodes[edge.src].type, edge.type, graph.nodes[edge.dst].type)
        out_deg[(key, edge.src)] += 1
        in_deg[(key, edge.dst)] += 1
        edge_count[key] += 1
        sources[key].add(edge.src)
    population = Counter(n.type for n in graph.nodes.values())
    stats: dict[N1Key, dict] = {}
    for key in edge_count:
        a, _r, b = key
        outs = [c for (k, _n), c in out_deg.items() if k == key]
        ins = [c for (k, _n), c in in_deg.items() if k == key]
        min_out = min(outs) if len(outs) == population[a] else 0
        min_in = min(ins) if len(ins) == population[b] else 0
        kind = cardinality(min_out, max(outs), min_in, max(ins))
        stats[key] = {
            "cardinality": kind,
            "max_out_degree": max(outs),
            "max_in_degree": max(ins),
            "edges": edge_count[key],
            "sources": len(sources[key]),
        }
    # As in the script, a relationship type is followed on every (A, R, B) once it is
    # many-to-one for one of them.
    many_to_one = {r for (_a, r, _b), s in stats.items() if s["cardinality"] == "many-to-one"}
    for key, s in stats.items():
        s["n1"] = key[1] in many_to_one
    return dict(sorted(stats.items()))


def n1_keys(detected: dict[N1Key, dict]) -> set[N1Key]:
    """The (type, relationship) pairs detected as many-to-one."""
    return {key for key, info in detected.items() if info["n1"]}


def hops(path: str) -> int:
    """Relationships between n and the element: 0 for n itself."""
    return path.count("]-")


def context_elements(graph: Graph, node_id: str, n1: set[N1Key], max_hops: int | None) -> list[ContextRef]:
    """Elements of context(n): the node, its edges and neighbours (following n:1 paths up to max_hops)."""
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

    # n:1 paths: breadth-first from n, each node visited once; an edge to a node
    # already visited is not traversed.
    visited = {node_id}
    frontier = [(node_id, t)]
    depth = 0
    while frontier and (max_hops is None or depth < max_hops):
        depth += 1
        nxt_frontier = []
        for current_id, path in frontier:
            current = graph.nodes[current_id]
            for eid in graph.out_edges.get(current_id, ()):
                edge = graph.edges[eid]
                nxt = graph.nodes[edge.dst]
                if (current.type, edge.type, nxt.type) not in n1 or nxt.id in visited:
                    continue
                visited.add(nxt.id)
                edge_path = f"{path}-[{edge.type}]->"
                refs.append(ContextRef(eid, "edge", edge_path))
                refs.append(ContextRef(nxt.id, "node", edge_path + nxt.type))
                nxt_frontier.append((nxt.id, edge_path + nxt.type))
        frontier = nxt_frontier

    # One entry per element: the shortest path, the first found on a tie.
    best: dict[str, ContextRef] = {}
    for ref in refs:
        if ref.element_id not in best or hops(ref.path) < hops(best[ref.element_id].path):
            best[ref.element_id] = ref
    return [ref for ref in refs if best[ref.element_id] is ref]


def prop_key(path: str, property_name: str) -> str:
    """Key of a property: '<path>.<property_name>'."""
    return f"{path}.{property_name}"

"""Snapshot, centrality, context and popularity on hand-built graphs."""

import pytest

from taskdef.centrality import compute_centrality
from taskdef.config import CentralityConfig
from taskdef.context import context_elements, detect_n1, n1_keys, prop_key
from taskdef.graph import Graph
from taskdef.popularity import compute_popularity

from conftest import edge, node


def test_snapshot_round_trip(tmp_path, graph):
    graph.save(tmp_path / "snap")
    loaded = Graph.load(tmp_path / "snap")
    assert loaded.content_hash() == graph.content_hash()
    assert loaded.stats() == graph.stats()


def test_n1_detection(graph):
    detected = detect_n1(graph)
    assert detected[("Airport", "IN", "City")]["n1"]
    assert detected[("City", "IN", "Country")]["n1"]
    assert not detected[("Airport", "HAS_ROUTE", "Airport")]["n1"]


def test_context_follows_n1_paths_only(graph):
    refs = context_elements(graph, "A1", n1_keys(detect_n1(graph)), max_hops=3)
    paths = {(r.element_id, r.path) for r in refs}
    assert ("A1", "Airport") in paths
    assert ("C2", "Airport-[IN]->City") in paths  # A1 is in cities[1 % 4] = C2
    assert ("K1", "Airport-[IN]->City-[IN]->Country") in paths
    assert ("in-C2", "Airport-[IN]->City-[IN]->") in paths
    # Direct route neighbors and their edges are in the context...
    assert ("A2", "Airport-[HAS_ROUTE]->Airport") in paths
    assert any(p == "Airport-[HAS_ROUTE]->" for _e, p in paths)
    assert any(p == "Airport<-[HAS_ROUTE]-Airport" for _e, p in paths)
    # ...but nothing is reached through two non-n:1 hops.
    assert not any(p.count("HAS_ROUTE") > 1 for _e, p in paths)
    assert not any("HAS_ROUTE" in p and "City" in p for _e, p in paths)


def test_context_respects_max_hops(graph):
    refs = context_elements(graph, "A1", n1_keys(detect_n1(graph)), max_hops=1)
    assert not any("Country" in r.path for r in refs)


def test_prop_key_format():
    assert prop_key("Airport", "elevation") == "Airport.elevation"
    assert prop_key("Airport-[IN]->City", "name") == "Airport-[IN]->City.name"
    assert prop_key("Airport-[HAS_ROUTE]->", "distance") == "Airport-[HAS_ROUTE]->.distance"


def test_popularity(graph):
    pop = compute_popularity(graph)
    assert pop("node", "Airport", "name") == 1.0
    assert pop("node", "Airport", "runways") == pytest.approx(6 / 12)
    assert pop("node", "Airport", "icao") == pytest.approx(4 / 12)
    assert pop("node", "Airport", "website") == pytest.approx(3 / 12)
    assert pop("node", "City", "name") == 1.0
    assert pop("node", "City", "website") == 0.0


def test_popularity_empty_string_is_null():
    g = Graph([node("a", "T", x="v"), node("b", "T", x="  ")], [])
    assert compute_popularity(g, empty_string_is_null=True)("node", "T", "x") == 0.5
    assert compute_popularity(g, empty_string_is_null=False)("node", "T", "x") == 1.0


def test_pagerank_cycle_is_uniform():
    g = Graph(
        [node("a", "T"), node("b", "T"), node("c", "T")],
        [edge("1", "R", "a", "b"), edge("2", "R", "b", "c"), edge("3", "R", "c", "a")],
    )
    cent = compute_centrality(g, CentralityConfig())
    assert sum(cent.raw[n] for n in "abc") == pytest.approx(1.0)
    for n in "abc":
        assert cent[n] == pytest.approx(1.0)
    assert cent["1"] == pytest.approx(1.0)


def test_normalization_per_type_and_edges(graph):
    cent = compute_centrality(graph, CentralityConfig())
    for ntype in ("Airport", "City", "Country"):
        values = [cent[n] for n in graph.nodes_of_type(ntype)]
        assert max(values) == pytest.approx(1.0)
        assert all(0 < v <= 1 for v in values)
    e = graph.edges["in-A1"]
    assert cent["in-A1"] == pytest.approx((cent[e.src] + cent[e.dst]) / 2)


def test_zero_centrality_becomes_epsilon():
    g = Graph([node("a", "T"), node("b", "T"), node("lonely", "T")], [edge("1", "R", "a", "b")])
    cent = compute_centrality(g, CentralityConfig(measure="degree", epsilon=1e-6))
    assert cent["lonely"] == 1e-6
    assert cent["a"] == 1.0

import pytest
import yaml

from taskdef.graph import Edge, Graph, Node


def node(nid, ntype, **props):
    return Node(id=nid, kind="node", type=ntype, props=props, prop_types={}, labels=[ntype])


def edge(eid, etype, src, dst, **props):
    return Edge(id=eid, kind="edge", type=etype, props=props, prop_types={}, src=src, dst=dst)


def toy_graph() -> Graph:
    """12 airports in 4 cities in 2 countries, with routes between airports.

    Airport -[IN]-> City -[IN]-> Country are n:1; HAS_ROUTE is not.
    Property presence varies so that popularities differ.
    """
    nodes = [node("K1", "Country", name="Brazil", code="BR"), node("K2", "Country", name="France", code="FR")]
    cities = [("C1", "Sao Paulo", "K1"), ("C2", "Rio de Janeiro", "K1"), ("C3", "Paris", "K2"), ("C4", "Lyon", "K2")]
    edges = []
    for cid, name, kid in cities:
        nodes.append(node(cid, "City", name=name, population=1000 * len(name)))
        edges.append(edge(f"in-{cid}", "IN", cid, kid))
    for i in range(1, 13):
        props = {
            "id": i,
            "name": f"Airport {i} International",
            "iata": f"A{i:02d}",
            "elevation": 100 + 37 * i,
            "latitude": -20.0 + i,
        }
        if i % 2 == 0:
            props["runways"] = 1 + i % 3
        if i % 3 == 0:
            props["icao"] = f"SB{i:02d}"
        if i % 4 == 0:
            props["website"] = f"https://airport{i}.example"
        if i % 6 == 0:
            props["opened"] = 1950 + i
        nodes.append(node(f"A{i}", "Airport", **props))
        edges.append(edge(f"in-A{i}", "IN", f"A{i}", cities[i % 4][0]))
    # Hubs: low ids get many routes, so centrality varies.
    rid = 0
    for i in range(1, 13):
        for j in range(1, 13):
            if i != j and (j <= 3 or (i + j) % 5 == 0):
                rid += 1
                edges.append(edge(f"r{rid}", "HAS_ROUTE", f"A{i}", f"A{j}", distance=10 * abs(i - j) + rid))
    return Graph(nodes, edges)


@pytest.fixture
def graph():
    return toy_graph()


@pytest.fixture
def config_file(tmp_path, graph):
    snap = tmp_path / "snapshot"
    graph.save(snap)

    def make(output_name="v1", **overrides):
        cfg = {
            "dataset": {
                "name": "toy",
                "node_type": "Airport",
                "context_max_hops": 3,
                "source": "snapshot",
                "snapshot_dir": str(snap),
            },
            "sampling": {"max_n": 2, "max_H": 2, "max_G": 2, "seed": 7},
            "properties": {"exclude_properties": ["id"], "leakage_pairs": [["Airport.iata", "Airport.icao"]]},
            "task_set_version": "toy-v1",
            "output_dir": str(tmp_path / output_name),
        }
        cfg.update(overrides)
        path = tmp_path / f"{output_name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    return make

"""Export a graph snapshot from Neo4j (TASKDEFINITION_SPEC.md §4.1)."""

from __future__ import annotations

import datetime as dt
import logging
import math

from .config import DatasetConfig, Neo4jConfig
from .graph import Edge, Graph, Node, choose_node_type, infer_type

log = logging.getLogger(__name__)


def convert_value(value):
    """Convert a Neo4j property value to ``(json_value, value_type)``."""
    try:
        from neo4j.time import Date, DateTime
    except ImportError:  # pragma: no cover - neo4j is a dependency
        Date = DateTime = ()
    if value is None:
        return None, "null"
    if isinstance(value, float) and math.isnan(value):
        return None, "null"
    if isinstance(value, (bool, int, float, str)):
        return value, infer_type(value)
    if isinstance(value, (Date, DateTime)):
        return value.iso_format(), "date"
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat(), "date"
    if isinstance(value, (list, tuple)):
        return [convert_value(v)[0] for v in value], "list"
    if isinstance(value, dict):
        return {str(k): convert_value(v)[0] for k, v in value.items()}, "map"
    # Time, Duration, spatial points, bytes...: kept as text, never eligible.
    return str(value), "other"


def convert_properties(props: dict) -> tuple[dict, dict]:
    values, types = {}, {}
    for name, raw in props.items():
        value, vtype = convert_value(raw)
        if vtype == "null":
            continue
        values[name] = value
        types[name] = vtype
    return values, types


def _driver(neo4j_cfg: Neo4jConfig):
    from neo4j import GraphDatabase

    uri, database, user, password = neo4j_cfg.connection()
    return GraphDatabase.driver(uri, auth=(user, password)), database


def check_connection(neo4j_cfg: Neo4jConfig) -> dict:
    """Connect and return node counts per label and edge counts per type."""
    driver, database = _driver(neo4j_cfg)
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            labels = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label"
            ).data()
            rels = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS c ORDER BY type"
            ).data()
    finally:
        driver.close()
    return {
        "labels": {r["label"]: r["c"] for r in labels},
        "relationship_types": {r["type"]: r["c"] for r in rels},
    }


def export_graph(neo4j_cfg: Neo4jConfig, dataset_cfg: DatasetConfig) -> Graph:
    idf = neo4j_cfg.id_function
    params: dict = {}
    node_where = ""
    edge_where = []
    if neo4j_cfg.labels:
        params["labels"] = neo4j_cfg.labels
        node_where = "WHERE any(l IN labels(n) WHERE l IN $labels)"
        edge_where.append("any(l IN labels(a) WHERE l IN $labels)")
        edge_where.append("any(l IN labels(b) WHERE l IN $labels)")
    if neo4j_cfg.relationship_types:
        params["rel_types"] = neo4j_cfg.relationship_types
        edge_where.append("type(r) IN $rel_types")
    node_query = f"MATCH (n) {node_where} RETURN toString({idf}(n)) AS id, labels(n) AS labels, properties(n) AS props"
    edge_query = (
        "MATCH (a)-[r]->(b) "
        + (("WHERE " + " AND ".join(edge_where)) if edge_where else "")
        + f" RETURN toString({idf}(r)) AS id, type(r) AS type, toString({idf}(a)) AS src,"
        f" toString({idf}(b)) AS dst, properties(r) AS props"
    )

    driver, database = _driver(neo4j_cfg)
    nodes: list[Node] = []
    edges: list[Edge] = []
    try:
        with driver.session(database=database, fetch_size=neo4j_cfg.fetch_size) as session:
            log.info("Exporting nodes from Neo4j")
            for record in session.run(node_query, params):
                values, types = convert_properties(dict(record["props"]))
                labels = list(record["labels"])
                nodes.append(
                    Node(
                        id=record["id"],
                        kind="node",
                        type=choose_node_type(labels, dataset_cfg.node_type, dataset_cfg.label_priority),
                        props=values,
                        prop_types=types,
                        labels=labels,
                    )
                )
            log.info("Exported %d nodes; exporting relationships", len(nodes))
            for record in session.run(edge_query, params):
                values, types = convert_properties(dict(record["props"]))
                edges.append(
                    Edge(
                        id=record["id"],
                        kind="edge",
                        type=record["type"],
                        props=values,
                        prop_types=types,
                        src=record["src"],
                        dst=record["dst"],
                    )
                )
            log.info("Exported %d relationships", len(edges))
    finally:
        driver.close()

    if dataset_cfg.node_id_property:
        nodes, edges = _remap_node_ids(nodes, edges, dataset_cfg)
    return Graph(nodes, edges)


def _remap_node_ids(nodes: list[Node], edges: list[Edge], dataset_cfg: DatasetConfig):
    """Use a domain property as node_id for nodes of the target type."""
    prop = dataset_cfg.node_id_property
    mapping: dict[str, str] = {}
    seen: set[str] = set()
    for node in nodes:
        if node.type != dataset_cfg.node_type:
            continue
        if prop not in node.props:
            raise ValueError(f"Node {node.id} of type {node.type} has no '{prop}' (dataset.node_id_property)")
        new_id = str(node.props[prop])
        if new_id in seen:
            raise ValueError(f"dataset.node_id_property '{prop}' is not unique: {new_id!r}")
        seen.add(new_id)
        mapping[node.id] = new_id
    other_ids = {n.id for n in nodes if n.id not in mapping}
    clash = other_ids & seen
    if clash:
        raise ValueError(f"node_id_property values clash with database ids: {sorted(clash)[:5]}")
    for node in nodes:
        node.id = mapping.get(node.id, node.id)
    for edge in edges:
        edge.src = mapping.get(edge.src, edge.src)
        edge.dst = mapping.get(edge.dst, edge.dst)
    return nodes, edges

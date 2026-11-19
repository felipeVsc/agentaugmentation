"""Live check of the snapshot context against ``findContextOfNode.py`` (TASKDEFINITION_SPEC.md §4.3).

``get_node_context`` and ``CARDINALITY_QUERY`` are copied unchanged from
``context_gen/Comparing-Nodes/findContextOfNode.py`` and
``collect_indicators.py`` (``detect_relationship_cardinalities``). They need APOC.
That module is not imported, because it pulls in the plotting and clustering
dependencies of the whole project.

``check_context`` runs the script on the live database for some nodes and compares
its context with ``context.context_elements`` on the snapshot, as multisets of
``(kind, label, property, hops, value)``. Element ids differ between the two
(``id(n)`` vs snapshot ids), and on a tie of path length the two may keep different
paths to the same element, so ids and paths are not compared.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Tuple

from .config import TaskDefConfig
from .context import context_elements, detect_n1, hops, n1_keys
from .graph import Graph
from .neo4j_source import _driver, convert_value

ContextTuple = Tuple[str, List[str], str, str, Any]
#                (type,   source,   label, property, value)

CARDINALITY_QUERY = """
    MATCH (s)-[r]->(t)
    WITH type(r) AS relType,
         apoc.text.join(labels(s),'|')  AS startLabels,
         apoc.text.join(labels(t),'|')  AS endLabels
    WITH DISTINCT relType, startLabels, endLabels

    CALL (relType, startLabels, endLabels) {
      MATCH (sx)
      WHERE apoc.text.join(labels(sx),'|')  = startLabels
      OPTIONAL MATCH (sx)-[rx]->(tx)
      WHERE type(rx) = relType
        AND apoc.text.join(labels(tx),'|')  = endLabels
      WITH sx, count(rx) AS outCnt
      RETURN
        min(outCnt) AS minOut,
        max(outCnt) AS maxOut,
        count(sx)   AS startPopulation
    }

    CALL (relType, startLabels, endLabels) {
      MATCH (ty)
      WHERE apoc.text.join(labels(ty),'|')  = endLabels
      OPTIONAL MATCH (sy)-[ry]->(ty)
      WHERE type(ry) = relType
        AND apoc.text.join(labels(sy),'|')  = startLabels
      WITH ty, count(ry) AS inCnt
      RETURN
        min(inCnt) AS minIn,
        max(inCnt) AS maxIn,
        count(ty)  AS endPopulation
    }

    RETURN
      relType,
      startLabels,
      endLabels,
      startPopulation,
      endPopulation,
      minOut, maxOut, minIn, maxIn,
      CASE
        WHEN minOut >= 1 AND maxOut <= 1 AND minIn >= 1 AND maxIn <= 1 THEN 'one-to-one'
        WHEN minOut >= 0 AND maxIn <= 1 THEN 'one-to-many'
        WHEN maxOut <= 1 AND minIn >= 0 THEN 'many-to-one'
        ELSE 'many-to-many'
      END AS cardinality
    ORDER BY relType, startLabels, endLabels
    """


def get_node_context(node_id: int, allowed_rels: List[str], db) -> List[ContextTuple]:
    """
    Retrieve the context of a node in a Neo4j property graph.

    Parameters
    ----------
    node_id : int
        Neo4j internal id of the start node (id(n)).
    allowed_rels : list of str
        Relationship types allowed when expanding directed outgoing paths
        (e.g., ["IN", "OF"]).
    db : any
        An object exposing a method `execute_query(cypher, parameters)`.

    Returns
    -------
    List[ContextTuple]
        A list of (type, source, label, property, value) tuples where:
          - type   ∈ {"node", "edge"}
          - source is a list indicating the provenance from the initial
            node, e.g. ['IN'] or ['IN','City','OF']
          - label  is the node label(s) (as "A:B") or the relationship type
          - property is the property name
          - value   is the property value
    """

    # ---------------------------------------------
    # 1) Base context: start node, direct neighbours, incident relationships
    # ---------------------------------------------
    base_cypher = """
    MATCH (n)
    WHERE id(n) = $id

    OPTIONAL MATCH (n)-[r0]-(nb)
    WITH n,
         collect(DISTINCT {
             kind:   'node',
             id:     id(nb),
             labels: labels(nb),
             props:  properties(nb),
             source: [type(r0)]
         }) AS directNodes,
         collect(DISTINCT {
             kind:   'rel',
             id:     id(r0),
             label:  type(r0),
             props:  properties(r0),
             source: [type(r0)]
         }) AS incidentRels

    RETURN
      { kind:   'node',
        id:     id(n),
        labels: labels(n),
        props:  properties(n),
        source: []
      } AS startNode,
      directNodes,
      incidentRels
    """

    base_rows = db.execute_query(base_cypher, {"id": node_id})
    if not base_rows:
        return []

    base_rec = base_rows[0]
    start_node = base_rec["startNode"]                 # map
    direct_nodes = base_rec.get("directNodes", [])     # list of maps
    incident_rels = base_rec.get("incidentRels", [])   # list of maps

    # ---------------------------------------------
    # Aggregator: keep shortest provenance per (element, property)
    # ---------------------------------------------
    best: Dict[Tuple[str, int, str], Tuple[List[str], str, Any]] = {}

    def consider(
        kind: str,
        element_id: int,
        source: List[str],
        label: str,
        props: Dict[str, Any],
    ):
        """Keep the shortest provenance list per (kind, id, property)."""
        if props is None:
            return
        for prop, value in props.items():
            key = (kind, element_id, prop)
            if key not in best or len(source) < len(best[key][0]):
                best[key] = (source, label, value)

    def label_str_from_labels(labels) -> str:
        if isinstance(labels, (list, tuple)):
            return ":".join(labels)
        return str(labels) if labels is not None else ""

    # Start node
    consider(
        kind=start_node["kind"],
        element_id=start_node["id"],
        source=start_node.get("source", []),
        label=label_str_from_labels(start_node.get("labels")),
        props=start_node.get("props"),
    )

    # Direct neighbour nodes
    for nd in direct_nodes:
        consider(
            kind=nd.get("kind", "node"),
            element_id=nd.get("id"),
            source=nd.get("source", []),
            label=label_str_from_labels(nd.get("labels")),
            props=nd.get("props"),
        )

    # Incident relationships
    for rd in incident_rels:
        consider(
            kind=rd.get("kind", "rel"),
            element_id=rd.get("id"),
            source=rd.get("source", []),
            label=str(rd.get("label", "")),
            props=rd.get("props"),
        )

    # ---------------------------------------------
    # 2) Path expansion with APOC for allowed_rels
    # ---------------------------------------------
    if allowed_rels:
        rel_filter = "|".join(r + ">" for r in allowed_rels)

        path_cypher = """
        MATCH (n)
        WHERE id(n) = $id

        CALL apoc.path.expandConfig(n, {
            relationshipFilter: $allowedFilter,
            minLevel: 1,
            uniqueness: 'NODE_GLOBAL'
        }) YIELD path

        WITH nodes(path) AS nds, relationships(path) AS rels
        RETURN
          [x IN nds |
             { id:     id(x),
               labels: labels(x),
               props:  properties(x)
             }
          ] AS pathNodes,
          [r IN rels |
             { id:    id(r),
               type:  type(r),
               props: properties(r)
             }
          ] AS pathRels
        """

        path_rows = db.execute_query(path_cypher, {
            "id": node_id,
            "allowedFilter": rel_filter,
        })

        # Helper: provenance functions (in Python, simpler & safer)
        def node_provenance(rel_types: List[str],
                            node_labels: List[str],
                            j: int) -> List[str]:
            """
            Provenance for node at index j (j >= 1) in the path:
            rels = [r0, r1, ...], labels = [L0, L1, L2, ...]
            Node 1 (City) via Airport-[:IN]->City:
              ['IN']
            Node 2 (Country) via Airport-[:IN]->City-[:OF]->Country:
              ['IN', 'City', 'OF']
            Formula:
              [r0] + flatten( [ [Lk, rk] for k in 1..j-1 ] )
            """
            if not rel_types:
                return []
            seq: List[str] = [rel_types[0]]
            for k in range(1, j):
                seq.append(node_labels[k])
                seq.append(rel_types[k])
            return seq

        def rel_provenance(rel_types: List[str],
                           node_labels: List[str],
                           i: int) -> List[str]:
            """
            Provenance for relationship at index i (0-based) in the path:
            Relationship 0 (IN) on Airport-[:IN]->City:
              ['IN']
            Relationship 1 (OF) on City-[:OF]->Country:
              ['IN', 'City', 'OF']
            Formula:
              [r0] + flatten( [ [Lk, rk] for k in 1..i ] )
            """
            if not rel_types:
                return []
            seq: List[str] = [rel_types[0]]
            for k in range(1, i + 1):
                seq.append(node_labels[k])
                seq.append(rel_types[k])
            return seq

        # Process each path
        for row in path_rows:
            path_nodes = row["pathNodes"]  # list of maps
            path_rels = row["pathRels"]    # list of maps

            if not path_rels:
                continue  # no edges => nothing beyond the start node

            rel_types = [r["type"] for r in path_rels]
            node_labels = [
                label_str_from_labels(n.get("labels")) for n in path_nodes
            ]

            # Nodes: indices 1..len(path_nodes)-1
            for j in range(1, len(path_nodes)):
                nm = path_nodes[j]
                prov = node_provenance(rel_types, node_labels, j)
                consider(
                    kind="node",
                    element_id=nm["id"],
                    source=prov,
                    label=node_labels[j],
                    props=nm.get("props"),
                )

            # Relationships: indices 0..len(path_rels)-1
            for i, rm in enumerate(path_rels):
                prov = rel_provenance(rel_types, node_labels, i)
                consider(
                    kind="rel",
                    element_id=rm["id"],
                    source=prov,
                    label=rm["type"],
                    props=rm.get("props"),
                )

    # ---------------------------------------------
    # 3) Build final list of tuples
    # ---------------------------------------------
    context: List[ContextTuple] = []
    for (kind, _elem_id, prop), (source, label, value) in best.items():
        type_str = "node" if kind == "node" else "edge"
        context.append((type_str, source, label, prop, value))

    return context


class _Db:
    """The ``execute_query(cypher, parameters)`` interface the script expects."""

    def __init__(self, driver, database):
        self.driver, self.database = driver, database

    def execute_query(self, query: str, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(query, parameters or {})]


def _value(value) -> str:
    return json.dumps(convert_value(value)[0], sort_keys=True, ensure_ascii=False, default=str)


def script_items(context: List[ContextTuple]) -> Counter:
    """Count the non-null (kind, label, property, hops, value) items of a context computed by the script."""
    return Counter(
        (kind, label, prop, (len(source) + 1) // 2, _value(value))
        for kind, source, label, prop, value in context
        if convert_value(value)[1] != "null"
    )


def snapshot_items(graph: Graph, node_id: str, n1, max_hops) -> Counter:
    """Count the same items for context(n) computed from the snapshot."""
    items: Counter = Counter()
    for ref in context_elements(graph, node_id, n1, max_hops):
        element = graph.element(ref.element_id)
        label = ":".join(element.labels) if ref.kind == "node" else element.type
        for name, value in element.props.items():
            items[(ref.kind, label, name, hops(ref.path), _value(value))] += 1
    return items


def _internal_id(db: _Db, cfg: TaskDefConfig, graph: Graph, node_id: str) -> int:
    prop = cfg.dataset.node_id_property
    if prop:
        value = graph.nodes[node_id].props[prop]
        rows = db.execute_query(f"MATCH (n:`{cfg.dataset.node_type}`) WHERE n[$p] = $v RETURN id(n) AS id", {"p": prop, "v": value})
    else:
        idf = cfg.neo4j.id_function
        rows = db.execute_query(f"MATCH (n) WHERE {idf}(n) = $v RETURN id(n) AS id", {"v": node_id})
    if len(rows) != 1:
        raise ValueError(f"Node {node_id!r} matches {len(rows)} nodes in Neo4j")
    return rows[0]["id"]


def check_context(cfg: TaskDefConfig, graph: Graph, node_ids: list[str]) -> dict:
    """Compare context(n) from the snapshot and from the live database for some nodes (read only)."""
    detected = detect_n1(graph)
    n1 = n1_keys(detected)
    driver, database = _driver(cfg.neo4j)
    try:
        db = _Db(driver, database)
        rows = db.execute_query(CARDINALITY_QUERY)
        live_m2o = sorted({r["relType"] for r in rows if r["cardinality"] == "many-to-one"})
        snap_m2o = sorted({k[1] for k, s in detected.items() if s["cardinality"] == "many-to-one"})
        nodes = {}
        for node_id in node_ids:
            live = script_items(get_node_context(_internal_id(db, cfg, graph, node_id), live_m2o, db))
            snap = snapshot_items(graph, node_id, n1, cfg.dataset.context_max_hops)
            nodes[node_id] = {
                "script_properties": sum(live.values()),
                "snapshot_properties": sum(snap.values()),
                "only_script": [list(k) for k in sorted((live - snap).elements())][:20],
                "only_snapshot": [list(k) for k in sorted((snap - live).elements())][:20],
            }
    finally:
        driver.close()
    return {
        "many_to_one_script": live_m2o,
        "many_to_one_snapshot": snap_m2o,
        "cardinalities_script": rows,
        "nodes": nodes,
        "match": live_m2o == snap_m2o and all(not v["only_script"] and not v["only_snapshot"] for v in nodes.values()),
    }

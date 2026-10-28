"""In-memory property graph snapshot and its on-disk format (TASKDEFINITION_SPEC.md §4.1).

A snapshot is two Parquet files, ``nodes.parquet`` and ``edges.parquet``.
Properties are stored as JSON, with a parallel map of value types, so that
types that JSON cannot express (dates) survive the round trip.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCALAR_TYPES = ("string", "integer", "float", "boolean", "date")


def infer_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "map"
    return "other"


def is_null(value, empty_string_is_null: bool = True) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if empty_string_is_null and isinstance(value, str) and value.strip() == "":
        return True
    return False


@dataclass
class Element:
    id: str
    kind: str  # "node" | "edge"
    type: str  # node label or relationship type
    props: dict
    prop_types: dict  # property name -> value type

    def value_type(self, name: str) -> str:
        return self.prop_types.get(name) or infer_type(self.props.get(name))


@dataclass
class Node(Element):
    labels: list[str] = field(default_factory=list)


@dataclass
class Edge(Element):
    src: str = ""
    dst: str = ""


def choose_node_type(labels: list[str], node_type: str, label_priority: list[str]) -> str:
    for label in label_priority:
        if label in labels:
            return label
    if node_type in labels:
        return node_type
    return sorted(labels)[0] if labels else "_unlabeled"


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self.nodes: dict[str, Node] = {}
        for node in nodes:
            if node.id in self.nodes:
                raise ValueError(f"Duplicate node id {node.id!r}")
            self.nodes[node.id] = node
        self.edges: dict[str, Edge] = {}
        self.out_edges: dict[str, list[str]] = defaultdict(list)
        self.in_edges: dict[str, list[str]] = defaultdict(list)
        for edge in sorted(edges, key=lambda e: e.id):
            if edge.id in self.edges:
                raise ValueError(f"Duplicate edge id {edge.id!r}")
            if edge.src not in self.nodes or edge.dst not in self.nodes:
                raise ValueError(f"Edge {edge.id!r} references a node missing from the snapshot")
            self.edges[edge.id] = edge
            self.out_edges[edge.src].append(edge.id)
            self.in_edges[edge.dst].append(edge.id)

    def element(self, element_id: str) -> Element:
        return self.nodes.get(element_id) or self.edges[element_id]

    def nodes_of_type(self, node_type: str) -> list[str]:
        return sorted(n.id for n in self.nodes.values() if n.type == node_type)

    def stats(self) -> dict:
        node_counts: dict[str, int] = defaultdict(int)
        edge_counts: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            node_counts[n.type] += 1
        for e in self.edges.values():
            edge_counts[e.type] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "nodes_per_type": dict(sorted(node_counts.items())),
            "edges_per_type": dict(sorted(edge_counts.items())),
        }

    # -- persistence ---------------------------------------------------------

    def _node_rows(self) -> list[dict]:
        return [
            {
                "element_id": n.id,
                "element_type": n.type,
                "labels": json.dumps(sorted(n.labels), ensure_ascii=False),
                "properties": json.dumps(n.props, ensure_ascii=False, sort_keys=True),
                "property_types": json.dumps(n.prop_types, sort_keys=True),
            }
            for n in sorted(self.nodes.values(), key=lambda n: n.id)
        ]

    def _edge_rows(self) -> list[dict]:
        return [
            {
                "element_id": e.id,
                "element_type": e.type,
                "src_id": e.src,
                "dst_id": e.dst,
                "properties": json.dumps(e.props, ensure_ascii=False, sort_keys=True),
                "property_types": json.dumps(e.prop_types, sort_keys=True),
            }
            for e in self.edges.values()
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        node_cols = ["element_id", "element_type", "labels", "properties", "property_types"]
        edge_cols = ["element_id", "element_type", "src_id", "dst_id", "properties", "property_types"]
        pd.DataFrame(self._node_rows(), columns=node_cols).to_parquet(directory / "nodes.parquet", index=False)
        pd.DataFrame(self._edge_rows(), columns=edge_cols).to_parquet(directory / "edges.parquet", index=False)

    @classmethod
    def load(cls, directory: str | Path) -> "Graph":
        directory = Path(directory)
        nodes_df = pd.read_parquet(directory / "nodes.parquet")
        edges_df = pd.read_parquet(directory / "edges.parquet")
        nodes = [
            Node(
                id=r.element_id,
                kind="node",
                type=r.element_type,
                props=json.loads(r.properties),
                prop_types=json.loads(r.property_types),
                labels=json.loads(r.labels),
            )
            for r in nodes_df.itertuples(index=False)
        ]
        edges = [
            Edge(
                id=r.element_id,
                kind="edge",
                type=r.element_type,
                props=json.loads(r.properties),
                prop_types=json.loads(r.property_types),
                src=r.src_id,
                dst=r.dst_id,
            )
            for r in edges_df.itertuples(index=False)
        ]
        return cls(nodes, edges)

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "nodes.parquet").exists() and (directory / "edges.parquet").exists()

    def content_hash(self) -> str:
        """Hash of the graph content, independent of the Parquet encoding."""
        h = hashlib.sha256()
        for row in self._node_rows():
            h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            h.update(b"\n")
        for row in self._edge_rows():
            h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            h.update(b"\n")
        return h.hexdigest()

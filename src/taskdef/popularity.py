"""Property popularity (TASKDEFINITION_SPEC.md §4.5).

``pop(p)`` = elements of the origin type with a non-null value for ``p``
divided by the number of elements of that type, over the whole snapshot.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .graph import Graph, is_null

PopKey = tuple[str, str, str]  # (element kind, element type, property name)


class Popularity:
    def __init__(self, non_null: dict[PopKey, int], totals: dict[tuple[str, str], int]):
        self.non_null = non_null
        self.totals = totals

    def __call__(self, kind: str, element_type: str, property_name: str) -> float:
        total = self.totals.get((kind, element_type), 0)
        if total == 0:
            return 0.0
        return self.non_null.get((kind, element_type, property_name), 0) / total

    def properties_of(self, kind: str, element_type: str) -> list[str]:
        return sorted(p for (k, t, p) in self.non_null if k == kind and t == element_type)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "element_kind": kind,
                "element_type": etype,
                "property_name": name,
                "non_null": count,
                "total": self.totals[(kind, etype)],
                "popularity": count / self.totals[(kind, etype)],
            }
            for (kind, etype, name), count in sorted(self.non_null.items())
        ]
        return pd.DataFrame(
            rows, columns=["element_kind", "element_type", "property_name", "non_null", "total", "popularity"]
        )


def compute_popularity(graph: Graph, empty_string_is_null: bool = True) -> Popularity:
    totals: Counter = Counter()
    non_null: Counter = Counter()
    for element in list(graph.nodes.values()) + list(graph.edges.values()):
        totals[(element.kind, element.type)] += 1
        for name, value in element.props.items():
            if not is_null(value, empty_string_is_null):
                non_null[(element.kind, element.type, name)] += 1
    return Popularity(dict(non_null), dict(totals))

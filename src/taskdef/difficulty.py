"""Task difficulty (TASKDEFINITION_SPEC.md §6).

    Difficulty(T) = cbrt( |H|/(|H|+|G|) * -log2(cent(n)) * sum_{p in H u G} -log2(pop(p) * cent_p(e)) )
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Difficulty:
    ratio: float          # |H| / (|H| + |G|)
    node_surprise: float  # -log2(cent(n))
    prop_surprise: float  # sum term
    value: float


def surprise(probability: float) -> float:
    # max() turns -0.0 into 0.0 when probability == 1.
    return max(0.0, -math.log2(probability))


def compute_difficulty(node_centrality: float, h_scores: list[float], g_scores: list[float]) -> Difficulty:
    """``h_scores`` / ``g_scores`` are ``pop(p) * cent_p(e)`` for each property of H / G."""
    size_h, size_g = len(h_scores), len(g_scores)
    if size_h + size_g == 0:
        raise ValueError("Difficulty is undefined when H and G are both empty")
    ratio = size_h / (size_h + size_g)
    node = surprise(node_centrality)
    prop = sum(surprise(s) for s in list(h_scores) + list(g_scores))
    return Difficulty(ratio, node, prop, (ratio * node * prop) ** (1.0 / 3.0))

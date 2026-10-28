import pytest

from taskdef.difficulty import compute_difficulty

# Table 1 of the draft: element centralities and property popularities.
CENT = {"e1": 0.7, "e2": 0.4, "e3": 0.2}
PROPS = {  # property -> (element, popularity)
    "a": ("e1", 0.6),
    "b": ("e1", 0.4),
    "c": ("e2", 0.1),
    "d": ("e2", 0.3),
    "e": ("e2", 0.6),
    "f": ("e3", 0.2),
    "g": ("e3", 0.1),
    "h": ("e3", 0.2),
}


def score(p):
    element, pop = PROPS[p]
    return pop * CENT[element]


# (node, H, G, ratio, -log(cent(n)), sum, difficulty) as printed in Table 1.
# The table computes the difficulty from rounded factors, so values are
# compared to two decimals with a tolerance of 0.01.
TABLE_1 = [
    ("e1", ["b"], ["a", "d"], 0.33, 0.51, 6.14, 1.01),
    ("e2", ["e"], ["f", "h"], 0.33, 1.32, 11.34, 1.71),
    ("e2", ["c", "d"], ["b"], 0.66, 1.32, 9.54, 2.03),
    ("e3", ["f", "h"], ["g"], 0.66, 2.32, 14.93, 2.84),
]


@pytest.mark.parametrize("node,H,G,ratio,node_s,prop_s,expected", TABLE_1)
def test_table_1(node, H, G, ratio, node_s, prop_s, expected):
    d = compute_difficulty(CENT[node], [score(p) for p in H], [score(p) for p in G])
    assert d.ratio == pytest.approx(ratio, abs=0.01)
    assert d.node_surprise == pytest.approx(node_s, abs=0.01)
    assert d.prop_surprise == pytest.approx(prop_s, abs=0.01)
    assert d.value == pytest.approx(expected, abs=0.01)


def test_central_node_has_zero_difficulty():
    d = compute_difficulty(1.0, [0.5], [0.5])
    assert d.node_surprise == 0.0
    assert d.value == 0.0


def test_empty_task_is_undefined():
    with pytest.raises(ValueError):
        compute_difficulty(0.5, [], [])

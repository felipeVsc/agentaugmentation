"""Sampling rules and the end-to-end build on the toy graph."""

import csv
import json

import pytest

from taskdef.config import ConfigError, Neo4jConfig, load_config
from taskdef.leakage import LeakagePairs, value_leaks
from taskdef.pipeline import build
from taskdef.sampling import Quartiles, stable_order


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# -- units ----------------------------------------------------------------------


def test_strata_boundaries():
    q = Quartiles.of([0.1, 0.2, 0.3, 0.4, 0.5])  # Q1 = 0.2, Q3 = 0.4; half-open as in the draft
    assert [q.stratum(v) for v in (0.1, 0.2, 0.3, 0.4, 0.5)] == ["low", "mid", "mid", "high", "high"]


def test_strata_tie_at_top_goes_high():
    q = Quartiles.of([0.2, 0.5, 1.0, 1.0, 1.0, 1.0])  # Q3 == Q4 == 1.0
    assert q.stratum(1.0) == "high"
    assert q.stratum(0.2) == "low"


def test_strata_tie_at_bottom_goes_low():
    q = Quartiles.of([0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.7, 0.9])  # Q0 == Q1 == 0.1, Q3 = 0.55
    assert q.stratum(0.1) == "low"
    assert q.stratum(0.3) == "mid" and q.stratum(0.9) == "high"


def test_stable_order_is_stable_when_items_are_removed():
    items = [f"n{i}" for i in range(50)]
    picked = stable_order(items, 1, "salt", key=str)[:5]
    not_picked = next(i for i in items if i not in picked)
    again = stable_order([i for i in items if i != not_picked], 1, "salt", key=str)[:5]
    assert again == picked


def test_value_leaks():
    assert value_leaks(2, 2.0, 4)
    assert value_leaks("São Paulo", " são  paulo ", 4)
    assert value_leaks("Guarulhos", "Guarulhos International", 4)
    assert not value_leaks("2", "0.82", 4)  # short values are only compared for equality
    assert not value_leaks(2459, 245, 4)


def test_leakage_pairs_are_transitive():
    pairs = LeakagePairs([["a", "b"], ["b", "c"]])
    assert pairs.related("a") == {"a", "b", "c"}
    assert pairs.related("z") == {"z"}


# -- config ---------------------------------------------------------------------


def test_unknown_config_key_fails(config_file):
    path = config_file(sampling={"max_n": 2, "typo": 1})
    with pytest.raises(ConfigError, match="typo"):
        load_config(path)


def test_missing_credentials_fail(monkeypatch):
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(ConfigError, match="NEO4J_USER"):
        Neo4jConfig().connection()


def test_credentials_from_environment(monkeypatch):
    monkeypatch.setenv("NEO4J_USER", "u")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")
    monkeypatch.setenv("NEO4J_URI", "neo4j://example:7687")
    assert Neo4jConfig().connection() == ("neo4j://example:7687", None, "u", "p")


# -- end to end -----------------------------------------------------------------


@pytest.fixture
def built(config_file):
    cfg = load_config(config_file())
    return cfg, build(cfg)


def test_build_writes_valid_task_set(built):
    cfg, out = built
    tasks = read_csv(out.output_dir / "tasks.csv")
    gt = read_csv(out.output_dir / "ground_truth.csv")
    assert tasks, "the toy graph must produce tasks"
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    assert len(tasks) == out.manifest["report"]["tasks_total"]
    assert len(tasks) + out.manifest["report"]["skipped_total"] + out.manifest["report"]["duplicates_removed"] == (
        out.manifest["report"]["expected_before_skips"]
    )

    gt_by_task = {}
    for row in gt:
        gt_by_task.setdefault(row["task_id"], {})[row["prop_key"]] = json.loads(row["value"])

    for t in tasks:
        H = json.loads(t["H"])
        G = json.loads(t["G"])
        assert len(H) == int(t["size_H"]) and len(G) == int(t["size_G"])
        assert all("value" not in h for h in H), "tasks.csv must not hold hidden values"
        h_keys = {h["prop_key"] for h in H}
        assert h_keys == set(gt_by_task[t["task_id"]])
        assert all(h["path"] == "Airport" for h in H), "H holds properties of n itself"
        assert not h_keys & {g["prop_key"] for g in G}
        assert "Airport.id" not in h_keys | {g["prop_key"] for g in G}, "excluded property used"
        if "Airport.iata" in h_keys:
            assert "Airport.icao" not in {g["prop_key"] for g in G}, "leakage pair split"
        for value in gt_by_task[t["task_id"]].values():
            assert not any(value_leaks(value, g["value"], 4) for g in G)
        assert float(t["difficulty"]) >= 0


def test_build_is_deterministic(config_file):
    a = build(load_config(config_file("a")))
    b = build(load_config(config_file("b")))
    for name in ("tasks.csv", "ground_truth.csv"):
        assert (a.output_dir / name).read_bytes() == (b.output_dir / name).read_bytes()
    for m in (a.manifest, b.manifest):
        m["config"].pop("output_dir")
    assert json.dumps(a.manifest, sort_keys=True) == json.dumps(b.manifest, sort_keys=True)


def test_task_set_is_immutable(config_file):
    cfg = load_config(config_file())
    build(cfg)
    with pytest.raises(FileExistsError):
        build(cfg)
    build(cfg, overwrite=True)


def test_manifest_records_strata_and_quartiles(built):
    _cfg, out = built
    m = out.manifest
    assert set(m["quartiles"]) == {"node_centrality", "H_property_popularity", "G_property_popularity", "intervals"}
    assert all(len(q) == 5 for k, q in m["quartiles"].items() if k != "intervals")
    assert "id" not in m["strata"]["H_property_popularity"]
    assert any(r["relationship"] == "IN" and r["n1"] for r in m["n1_relationships"])
    for name in ("centrality.parquet", "popularity.parquet", "context.parquet"):
        assert (out.output_dir / "intermediate" / name).exists()


def test_fixed_partitions_keep_other_nodes_tasks(config_file, graph):
    from taskdef.centrality import compute_centrality
    from taskdef.context import detect_n1, n1_keys
    from taskdef.popularity import compute_popularity
    from taskdef.sampling import TaskBuilder

    cfg = load_config(config_file())
    builder = TaskBuilder(graph, cfg, compute_centrality(graph, cfg.centrality), compute_popularity(graph), n1_keys(detect_n1(graph)))
    parts = builder.partitions()
    before = builder.build(parts).tasks
    assert [t.task_id for t in builder.build().tasks] == [t.task_id for t in before]

    # Drop one sampled node from N; with the partitions held fixed, the other nodes keep their tasks.
    gone = before[0].node_id
    builder.N = [n for n in builder.N if n != gone]
    after = builder.build(parts).tasks
    kept = {t.node_id for t in after}
    assert gone not in kept
    assert {t.task_id for t in after if t.node_id in kept} >= {t.task_id for t in before if t.node_id in kept}


def test_explicit_sizes_without_context(config_file):
    out = build(load_config(config_file("noctx", sampling={"max_n": 2, "max_H": 1, "sizes_G": [1, 3], "seed": 7,
                                                          "use_context": False})))
    tasks = read_csv(out.output_dir / "tasks.csv")
    assert tasks and {int(t["size_H"]) for t in tasks} == {1} and {int(t["size_G"]) for t in tasks} <= {1, 3}
    assert all(g["path"] == "Airport" for t in tasks for g in json.loads(t["G"]))
    assert out.manifest["report"]["expected_before_skips"] == 27 * 2 * 1 * 2


def test_invalid_sizes_fail(config_file):
    with pytest.raises(ConfigError, match="sizes_G"):
        load_config(config_file(sampling={"sizes_G": [0, 3]}))


def test_h_and_g_share_one_popularity_partition(config_file, graph):
    from taskdef.centrality import compute_centrality
    from taskdef.context import detect_n1, n1_keys
    from taskdef.popularity import compute_popularity
    from taskdef.sampling import TaskBuilder

    cfg = load_config(config_file())
    builder = TaskBuilder(graph, cfg, compute_centrality(graph, cfg.centrality), compute_popularity(graph), n1_keys(detect_n1(graph)))
    parts = builder.partitions()
    assert parts.h is parts.g
    for t in builder.build(parts).tasks:
        assert all(parts.g.stratum(c.popularity) == t.G_stratum for c in t.G)

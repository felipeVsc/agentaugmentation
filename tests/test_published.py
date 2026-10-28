"""DOLAP 2026 task design (taskdef.published) and its evaluation (evaluation.dolap)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from evaluation import dolap as D
from taskdef.config import load_config
from taskdef.published import build_published, import_published, specificity, _published_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dolap2026"
needs_data = pytest.mark.skipif(not (DATA / "test_results.csv").exists(), reason="published data not present")


def published_config(tmp_path, **published):
    cfg = {
        "dataset": {"name": "airports", "node_type": "Airport", "source": "neo4j", "node_id_property": "icao"},
        "properties": {"exclude_properties": ["airport_id"]},
        "published": {"popularity_file": str(DATA / "rank_properties.csv"),
                      "centrality_file": str(DATA / "ranks_pagerank.csv"), **published},
        "task_set_version": "dolap-test",
        "output_dir": str(tmp_path / "out"),
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return load_config(path)


def test_specificity_formula():
    # exp(-mean ln(pop * cent)): with pop = 1 it is 1 / cent
    assert specificity(["a", "b"], {"a": 1.0, "b": 1.0}, 0.25) == pytest.approx(4.0)
    assert specificity([], {}, 0.5) == 0.0


def test_published_path_format():
    assert _published_path("Airport-[OF]->City-[IN]->Country")[:3] == ("node", ["OF", "City", "IN"], "Country")
    assert _published_path("Airport-[ROUTE_TO]->")[:3] == ("edge", ["ROUTE_TO"], "ROUTE_TO")


@needs_data
def test_import_reproduces_published_tasks(tmp_path):
    cfg = published_config(tmp_path)
    m = import_published(cfg, DATA / "test_results.csv", node_values=DATA / "ground_truth.csv", use_graph=False)
    assert m["summary"]["tasks"] == 36
    assert m["summary"]["per_difficulty_prop"] == {q: 9 for q in D.QUADRANTS}
    assert m["difficulty_check"]["max_abs_difference_vs_published"] < 1e-9
    assert "G_input_mismatch" not in m["report"]
    tasks = pd.read_csv(tmp_path / "out" / "tasks.csv")
    row = tasks[(tasks.node_id == "PAKK") & (tasks.difficulty_prop == "easyeasy")].iloc[0]
    assert [h["property_name"] for h in json.loads(row.H)] == ["location", "source", "dst"]
    assert "input: altitude:154.0,city:Koyuk" in row.published_input
    assert "Here are some relationships" in row.published_input_context
    gt = pd.read_csv(tmp_path / "out" / "ground_truth.csv")
    assert len(gt) == 108


def toy_published_graph(tmp_path):
    from conftest import toy_graph

    snap = tmp_path / "snap"
    toy_graph().save(snap)
    return snap


@pytest.mark.parametrize("profile", ["code", "paper"])
def test_build_published_on_toy_graph(tmp_path, profile):
    snap = toy_published_graph(tmp_path)
    cfg = {
        "dataset": {"name": "toy", "node_type": "Airport", "source": "snapshot", "snapshot_dir": str(snap)},
        "properties": {"exclude_properties": ["id"]},
        "published": {"profile": profile, "size_H": 2, "size_G": 2, "nodes_per_stratum": 2,
                      "easy_threshold": 0.9, "node_key": "iata"},
        "task_set_version": "toy-dolap",
        "output_dir": str(tmp_path / f"out-{profile}"),
    }
    path = tmp_path / f"{profile}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    m = build_published(load_config(path))
    tasks = pd.read_csv(tmp_path / f"out-{profile}" / "tasks.csv")
    assert len(tasks) == m["summary"]["tasks"] > 0
    assert set(tasks.node_stratum) <= {"easy", "medium", "hard"}
    for t in tasks.itertuples():
        H = [h["property_name"] for h in json.loads(t.H)]
        assert len(H) == 2 and not set(H) & set(json.loads(t.G_drawn))
    if profile == "paper":  # H1/H4 are drawn once per node
        for _, g in tasks.groupby("node_id"):
            assert g[g.H_stratum == "easy"].H.nunique() <= 1 and g[g.H_stratum == "hard"].H.nunique() <= 1
    # Reproducible: the same seed gives the same tasks.
    cfg["output_dir"] = str(tmp_path / f"again-{profile}")
    path.write_text(yaml.safe_dump(cfg))
    build_published(load_config(path))
    again = pd.read_csv(tmp_path / f"again-{profile}" / "tasks.csv")
    assert list(again.task_id) == list(tasks.task_id)


# Evaluation ---------------------------------------------------------------------


def test_tolerance_rules():
    assert D.matches({"name": "Guarulhos–São Paulo Airport"}, {"name": "Guarulhos/Sao Paulo Airport"}, "name")
    assert not D.matches({"city": "Sandane"}, {"city": "Magadan"}, "city")
    assert D.matches({"altitude": 1026.0}, {"altitude": 1000.0}, "altitude")
    assert not D.matches({"altitude": 2.0}, {"altitude": 3.0}, "altitude")
    geo_truth = {"latitude": 33.6367, "longitude": -84.4281}
    assert D.matches(geo_truth, {"latitude": 33.64, "longitude": -84.43}, "latitude")
    assert D.matches(geo_truth, {"latitude": 33.64}, "latitude") is None  # paper: needs both coordinates
    assert D.matches_standard(geo_truth, {"latitude": 33.64}, "latitude")  # standard: lone coordinate, 0.1°
    assert not D.matches_standard(geo_truth, {"latitude": 34.0}, "latitude")
    loc = {"location": "POINT(-84.428101,33.6367)"}
    assert D.matches_standard(loc, {"location": "33.6367, -84.4281"}, "location")
    assert D.matches_standard(loc, {"location": "-84.43,33.64"}, "location")
    assert not D.matches_standard(loc, {"location": "Atlanta"}, "location")
    # Numbers given as text with a unit (Strategy B): metres -> feet for lengths, hemispheres for lone coordinates.
    truth = {"altitude": 162.0, "runwaysLength": 9000.0, "runways": 1.0, "timezone": 5.5, "longitude": -161.154}
    assert D.matches_standard(truth, {"altitude": "49 m"}, "altitude")
    assert not D.matches_standard(truth, {"altitude": "500 ft"}, "altitude")
    assert D.matches_standard(truth, {"runwaysLength": "2743 m"}, "runwaysLength")
    assert D.matches_standard(truth, {"runways": "1 runway"}, "runways")
    assert D.matches_standard(truth, {"timezone": "UTC+5:30"}, "timezone")
    assert D.matches_standard(truth, {"longitude": "161.15 W"}, "longitude")
    assert not D.matches_standard(truth, {"longitude": "161.15 E"}, "longitude")


def test_vote():
    assert D.vote(["10", "20", None]) == 15.0
    assert D.vote(["Paris", "Paris", "Orly"]) == "Paris"
    assert D.vote(["unknown", None]) is None


def test_metric_definitions():
    pub = pd.DataFrame({"model": "m", "correct": [True, False, False, True]})
    m = D.published_metrics(pub, ["model"]).iloc[0]
    assert (m.precision, m.recall) == (1.0, 0.5) and m.f1 == pytest.approx(2 * 0.5 / 1.5)
    std = pd.DataFrame({"model": "m", "outcome": ["correct", "wrong", "abstained", "abstained"]})
    s = D.standard_metrics(std, ["model"]).iloc[0]
    assert (s.precision, s.recall, s.abstention_rate, s.error_rate) == (0.5, 0.25, 0.5, 0.25)


@needs_data
def test_published_scoring_reproduces_table3(tmp_path):
    cfg = published_config(tmp_path)
    import_published(cfg, DATA / "test_results.csv", node_values=DATA / "ground_truth.csv", use_graph=False)
    tasks = D.load_tasks(tmp_path / "out" / "tasks.csv")
    votes = D.load_published_votes(DATA / "published_results", tasks)
    pm = D.published_metrics(D.score_published(votes, D.load_node_values(DATA / "ground_truth.csv")),
                             ["model", "context", "prompt"]).set_index(["model", "context", "prompt"])
    for model, ctx, prompt, f1, recall in [("GPT-5", False, "3cot", .67, .50), ("GPT-5 Mini", False, "few3", .57, .40),
                                           ("Gemini 2.5 Pro", True, "5cot", .65, .48), ("Gemini 2.5 Flash", True, "5cot", .61, .44),
                                           ("Claude Haiku 4.5", True, "compact", .54, .37), ("Claude Sonnet 4.5", True, "1cot", .80, .66)]:
        row = pm.loc[(model, ctx, prompt)]
        assert round(row.f1, 2) == pytest.approx(f1, abs=0.011) and round(row.recall, 2) == pytest.approx(recall, abs=0.011)

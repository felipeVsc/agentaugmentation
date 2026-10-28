"""Strategy A, offline (LLM_SPEC.md §10): rendering, parsing, runner, resume, repair, batch, leakage."""

import builtins
import csv
import json
from pathlib import Path

import pytest
import yaml

from llmcore.config import ConfigError, ModelConfig
from llmcore.providers.mock import MockProvider
from strategy_a import batch
from strategy_a.config import load_config
from strategy_a.parsing import parse_response
from strategy_a.results import read_results
from strategy_a.runner import Runner
from strategy_a.tasks import Task
from strategy_a.templates import load_template, path_label, render, target_labels

ROOT = Path(__file__).resolve().parents[1]

H = [
    {"prop_key": "Airport.elevation", "property_name": "elevation", "element_type": "Airport", "path": "Airport"},
    {"prop_key": "Airport.runways", "property_name": "runways", "element_type": "Airport", "path": "Airport"},
]
G = [
    {"prop_key": "Airport.name", "property_name": "name", "element_type": "Airport", "element_id": "SBGR",
     "path": "Airport", "value": "Guarulhos International"},
    {"prop_key": "Airport-[IN]->City.name", "property_name": "name", "element_type": "City", "element_id": "c1",
     "path": "Airport-[IN]->City", "value": "São Paulo"},
]


def write_tasks(path: Path, n: int = 2) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task_id", "task_set_version", "node_type", "node_id", "H", "G", "difficulty"])
        w.writeheader()
        for i in range(n):
            g = [dict(G[0], value=f"Airport {i}", element_id=f"N{i}"), G[1]]  # distinct prompts per task
            w.writerow({"task_id": f"t{i}", "task_set_version": "toy-v1", "node_type": "Airport", "node_id": f"N{i}",
                        "H": json.dumps(H), "G": json.dumps(g), "difficulty": "1.0"})


@pytest.fixture
def setup(tmp_path):
    write_tasks(tmp_path / "tasks.csv")
    (tmp_path / "ground_truth.csv").write_text("task_id,prop_key,value,value_type\n")

    def make(templates=None, models=None, **extra):
        cfg = {
            "inputs": {"tasks": str(tmp_path / "tasks.csv")},
            "runs_per_task": 3,
            "pfe_tokenizer": "simple",
            "output_dir": str(tmp_path / "out"),
            "templates": templates
            or [
                {"id": "base_zero_shot", "file": str(ROOT / "templates/base_zero_shot.v1.txt")},
                {"id": "compact_cot", "file": str(ROOT / "templates/compact_cot.v1.txt")},
            ],
            "models": models or [{"id": "m1", "provider": "mock", "params": {"temperature": 0.7, "max_tokens": 64}}],
            **extra,
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return load_config(path)

    return make


# Templates ------------------------------------------------------------------


def test_labels():
    assert path_label("Airport", "name") == "Airport name"
    assert path_label("Airport-[IN]->City", "name") == "City (the Airport is IN this City) name"
    assert path_label("Airport<-[HAS]-Country", "name") == "Country (this Country HAS the Airport) name"
    assert path_label("Airport-[HAS_ROUTE]->", "distance") == "HAS_ROUTE relationship of the Airport: distance"
    dup = [dict(H[0]), dict(H[0], prop_key="Airport-[IN]->City.elevation", path="Airport-[IN]->City", element_type="City")]
    labels = target_labels(dup)
    assert len(labels) == 2 and set(labels.values()) == {"Airport.elevation", "Airport-[IN]->City.elevation"}


def test_render_all_templates():
    task = Task("t", "v", "Airport", "SBGR", H, G)
    for f in sorted((ROOT / "templates").glob("*.txt")):
        tpl = load_template(f)
        r = render(tpl, task, "", "list")
        assert "- City (the Airport is IN this City) name: São Paulo" in r.user
        assert '"elevation": "<value or null>"' in r.user
        assert "{" + "known_properties}" not in r.user
        assert r.labels == {"elevation": "Airport.elevation", "runways": "Airport.runways"}


def test_template_placeholder_errors(tmp_path):
    bad = tmp_path / "bad.v1.txt"
    bad.write_text("### USER\n{known_properties} {target_properties} {output_format} {oops}")
    with pytest.raises(ConfigError, match="unknown placeholders"):
        load_template(bad)
    bad.write_text("### USER\n{known_properties} {target_properties}")
    with pytest.raises(ConfigError, match="missing required"):
        load_template(bad)
    ok = tmp_path / "ok.v1.txt"
    ok.write_text("### USER\n{known_properties} {target_properties} {output_format} literal {{braces}}")
    assert "literal {braces}" in render(load_template(ok), Task("t", "v", "Airport", "n", H, G), "", "list").user


# Parsing ---------------------------------------------------------------------

LABELS = {"elevation": "Airport.elevation", "runways": "Airport.runways"}


@pytest.mark.parametrize(
    "text, expected, ok",
    [
        ('{"elevation": "2459 ft", "runways": 2}', {"Airport.elevation": "2459 ft", "Airport.runways": 2}, True),
        ('Sure! Here it is: {"elevation": 750, "runways": null} Hope it helps.', {"Airport.elevation": 750, "Airport.runways": None}, True),
        ('Reasoning {"draft": 1}...\n```json\n{"elevation": 750, "runways": 3,}\n```', {"Airport.elevation": 750, "Airport.runways": 3}, True),
        ("{'elevation': 'x', 'runways': None}", {"Airport.elevation": "x", "Airport.runways": None}, True),
        ('{"Elevation": 1, "extra": 5}', {"Airport.elevation": 1, "Airport.runways": None}, True),
        ("I don't know.", {"Airport.elevation": None, "Airport.runways": None}, False),
        ('{"answer": 5}', {"Airport.elevation": None, "Airport.runways": None}, False),
    ],
)
def test_parse(text, expected, ok):
    r = parse_response(text, LABELS, H)
    assert r.predictions == expected
    assert r.ok is ok


def test_parse_events():
    r = parse_response('{"Elevation": 1, "extra": 5}', LABELS, H)
    assert "extra_key:extra" in r.events and "missing_key:Airport.runways" in r.events


# Runner ----------------------------------------------------------------------


def no_ground_truth(monkeypatch):
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if Path(str(file)).name == "ground_truth.csv":
            raise AssertionError("Strategy A opened ground_truth.csv")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)


def test_run_end_to_end_and_resume(setup, monkeypatch):
    no_ground_truth(monkeypatch)
    cfg = setup()
    counts = Runner(cfg, log=lambda m: None).run()
    assert counts == {"ok": 2 * 2 * 3}  # tasks × variants × runs
    out = Path(cfg.output_dir)
    rows = read_results(out / "results.csv")
    keys = {(r["task_id"], r["variant"], r["run_index"]) for r in rows}
    assert len(rows) == len(keys) == 12
    r = rows[0]
    assert r["strategy"] == "A" and r["evidence"] == "{}" and r["search_calls"] == "0"
    assert set(json.loads(r["predictions"])) == {"Airport.elevation", "Airport.runways"}
    assert int(r["pfe_tokens"]) > 0 and r["prompt_version"] == "v1"
    logs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert len(logs) == 12 and logs[0]["label_map"] == LABELS
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status_counts"] == {"ok": 12}

    # Resume: nothing new, no duplicates.
    assert Runner(cfg, log=lambda m: None).run() == {}
    assert len(read_results(out / "results.csv")) == 12


def test_runs_are_cached_separately(setup):
    cfg = setup(templates=[{"id": "base_zero_shot", "file": str(ROOT / "templates/base_zero_shot.v1.txt")}])
    mock = MockProvider(cfg.models[0])
    Runner(cfg, providers={"m1": mock}, log=lambda m: None).run()
    assert len(mock.calls) == 2 * 3
    assert sorted({c.run_index for c in mock.calls}) == [1, 2, 3]


def test_repair_call(setup):
    cfg = setup(templates=[{"id": "base_zero_shot", "file": str(ROOT / "templates/base_zero_shot.v1.txt")}])
    def answer(req):  # first call: no JSON; repair call (3 messages): valid JSON
        return '{"elevation": 10, "runways": 1}' if len(req.messages) == 3 else "no json here"

    mock = MockProvider(cfg.models[0], responder=answer)
    counts = Runner(cfg, providers={"m1": mock}, log=lambda m: None).run()
    assert counts == {"ok": 6}
    logs = [json.loads(line) for line in (Path(cfg.output_dir) / "runs.jsonl").read_text().splitlines()]
    assert all(log["repaired"] and len(log["calls"]) == 2 for log in logs)
    assert json.loads(read_results(Path(cfg.output_dir) / "results.csv")[0]["predictions"])["Airport.elevation"] == 10


def test_parse_error_and_truncated(setup):
    cfg = setup(templates=[{"id": "base_zero_shot", "file": str(ROOT / "templates/base_zero_shot.v1.txt")}], repair=False)
    texts = iter([("nothing", "stop"), ('{"elevation": 1', "length"), ('{"elevation": 1}', "length")] * 10)
    mock = MockProvider(cfg.models[0], responder=lambda req: next(texts))
    counts = Runner(cfg, providers={"m1": mock}, log=lambda m: None).run()
    assert counts["parse_error"] >= 1 and counts["truncated"] >= 1


def test_temperature_zero_rejected():
    with pytest.raises(ConfigError, match="temperature"):
        ModelConfig.from_dict({"id": "x", "provider": "mock", "params": {"temperature": 0}})


def test_model_defaults_merge():
    defaults = {"provider": "ollama", "params": {"temperature": 0.7, "max_tokens": 100}, "options": {"num_ctx": 8192, "seed": 0}}
    m = ModelConfig.from_dict({"id": "q", "model": "qwen3:14b", "options": {"think": False}, "params": {"max_tokens": 50}}, defaults)
    assert m.options == {"num_ctx": 8192, "seed": 0, "think": False}
    assert m.params == {"temperature": 0.7, "max_tokens": 50}
    assert m.max_concurrency == 1  # Ollama default


def test_few_shot_leakage(setup, tmp_path):
    ex = {"version": "v1", "examples": [{"node_id": "N0", "node_type": "Airport", "G": G, "H": H[:1], "answers": {"Airport.elevation": 1}}]}
    (tmp_path / "ex.json").write_text(json.dumps(ex))
    cfg = setup(
        templates=[{"id": "base_few_shot", "file": str(ROOT / "templates/base_few_shot.v1.txt"), "n_examples": [1]}],
        inputs={"tasks": str(tmp_path / "tasks.csv"), "examples": str(tmp_path / "ex.json")},
    )
    with pytest.raises(ConfigError, match="appear in tasks.csv"):
        Runner(cfg, log=lambda m: None).prepare()


def test_few_shot_variants(setup, tmp_path):
    ex = {"version": "v1", "examples": [
        {"node_id": f"X{i}", "node_type": "Airport", "G": G, "H": H[:1], "answers": {"Airport.elevation": i}} for i in range(3)
    ]}
    (tmp_path / "ex.json").write_text(json.dumps(ex))
    cfg = setup(
        templates=[{"id": "base_few_shot", "file": str(ROOT / "templates/base_few_shot.v1.txt"), "n_examples": [1, 3]}],
        inputs={"tasks": str(tmp_path / "tasks.csv"), "examples": str(tmp_path / "ex.json")},
    )
    runner = Runner(cfg, log=lambda m: None)
    runner.prepare()
    assert [v.id for v in runner.variants] == ["base_few_shot_1", "base_few_shot_3"]
    assert runner.pfe[("base_few_shot_3", "m1")] > runner.pfe[("base_few_shot_1", "m1")]
    user = runner.jobs()[0].rendered.user
    assert "Example 1" in user and '{"elevation": 0}' in user


def test_batch_flow(setup):
    cfg = setup()
    runner = Runner(cfg, log=lambda m: None)
    submitted = batch.submit(runner)
    assert sum(b["n"] for b in submitted) == 12
    assert batch.collect(Runner(cfg, log=lambda m: None))["ok"] == 12
    counts = Runner(cfg, log=lambda m: None).run(cache_only=True)
    assert counts == {"ok": 12}
    rows = read_results(Path(cfg.output_dir) / "results.csv")
    assert all(r["latency_s"] == "" for r in rows)  # batch: queue time is not model time


def test_cache_only_skips_missing(setup):
    cfg = setup()
    counts = Runner(cfg, log=lambda m: None).run(cache_only=True)
    assert counts == {"not_cached": 12}
    assert read_results(Path(cfg.output_dir) / "results.csv") == []

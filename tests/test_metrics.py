"""Stored metrics: reasoning tokens, prices, effort texts, results.csv upgrade, EIS (METRICS.md)."""

import csv

import pytest

from evaluation.eis import agent_energy, run_eis
from llmcore.config import Pricing
from llmcore.tokenizers import get_tokenizer, reasoning_tokens
from strategy_a.results import RESULT_COLUMNS, ResultsWriter, read_results
from strategy_a.tasks import Task
from strategy_a.templates import task_effort_texts, text_sizes


def test_reasoning_tokens_by_provider():
    tok = get_tokenizer("simple")
    assert reasoning_tokens({"eval_count": 50, "thinking": "a b c"}, tok) == tok.count("a b c")
    assert reasoning_tokens({"eval_count": 50, "thinking": None}, tok) == 0
    assert reasoning_tokens({"usage": {"completion_tokens_details": {"reasoning_tokens": 7}}}, tok) == 7
    assert reasoning_tokens({"usage_metadata": {"thoughts_token_count": 3}}, tok) == 3
    assert reasoning_tokens({"usage": {"output_tokens": 9}}, tok) is None


def test_pricing_cost():
    p = Pricing(input_per_mtok=0.15, output_per_mtok=0.60)
    assert p.cost(1_000_000, 500_000) == pytest.approx(0.45)
    assert p.cost(1_000_000, 0, "batch") == pytest.approx(0.075)


def task(G, meta=None):
    return Task(task_id="t", task_set_version="v", node_type="Airport", node_id="n1",
                H=[{"prop_key": "Airport.iata", "property_name": "iata", "path": "Airport"}], G=G, meta=meta or {})


def test_effort_texts_split_input_and_context():
    G = [{"prop_key": "Airport.name", "property_name": "name", "path": "Airport", "value": "Guarulhos"},
         {"prop_key": "Airport-[IN]->City.name", "property_name": "name", "path": "Airport-[IN]->City", "value": "Sao Paulo"}]
    texts = task_effort_texts(task(G), False, False)
    assert "Guarulhos" in texts["input"] and "Sao Paulo" not in texts["input"]
    assert "Sao Paulo" in texts["context"] and texts["targets"] == "iata"
    sizes = text_sizes(texts, get_tokenizer("simple"))
    assert sizes["targets"] == {"chars": 4, "words": 1, "tokens": get_tokenizer("simple").count("iata")}


def test_effort_texts_published_input():
    meta = {"published_input": "\n Here is the entity:\n\n  input: altitude:154.0,city:Koyuk\n\n  context: [NO_CONTEXT_ADDED].",
            "published_input_context": "\n Here is the entity:\n\n  input: altitude:154.0,city:Koyuk\n\n"
                                       "  Here are some relationships in the graph: [['node', 'x', 1]].\n"}
    plain = task_effort_texts(task([], meta), False, True)
    assert plain == {"input": "altitude:154.0,city:Koyuk", "context": "", "targets": "iata"}
    ctx = task_effort_texts(task([], meta), True, True)
    assert ctx["input"] == "altitude:154.0,city:Koyuk" and ctx["context"] == "[['node', 'x', 1]]"


def test_old_results_header_is_upgraded(tmp_path):
    old = [c for c in RESULT_COLUMNS if c not in ("tokens_reasoning", "llm_calls", "cost_usd")]
    with open(tmp_path / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=old)
        w.writeheader()
        w.writerow({c: "1" for c in old})
    ResultsWriter(tmp_path).close()
    rows = read_results(tmp_path / "results.csv")
    assert list(rows[0]) == RESULT_COLUMNS and rows[0]["cost_usd"] == "" and rows[0]["tokens_in"] == "1"


def test_agent_energy():
    assert agent_energy(0.002, 2, 3, 0.5) == pytest.approx((2 + 1.5) * 0.001)
    assert agent_energy(0.001, 1, 0, 0.5) == pytest.approx(0.001)  # Strategy A: its LLM energy
    assert agent_energy(None, 2, 1, 0.5) is None
    with pytest.raises(ValueError, match="w_tool"):
        agent_energy(0.001, 1, 0, None)
    row = {"energy_kwh": "0.002", "llm_calls": "2", "search_calls": "3", "search_cache_hits": "1"}
    assert run_eis(row, 1.0) == pytest.approx(0.005)
    assert run_eis(row, 1.0, count_cached_searches=False) == pytest.approx(0.004)

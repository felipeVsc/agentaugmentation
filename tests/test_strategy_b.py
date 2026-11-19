"""Strategy B offline (AGENT_SPEC.md §11): mocked LLM and mocked search."""

import builtins
import io
import json
from pathlib import Path

import pytest
import yaml

from llmcore.providers.mock import MockProvider
from strategy_a.results import read_results
from strategy_a.tasks import Task
from strategy_b import steps
from strategy_b.config import load_config
from strategy_b.runner import Runner
from strategy_b.search import MockSearch, OllamaSearch, SearchResult, TavilySearch, is_blocked

from test_strategy_a import G, H, ROOT, write_tasks

GOOD = [
    {"url": "https://en.wikipedia.org/wiki/Airport_0", "title": "Airport 0", "content": "Airport 0 has an elevation of 2459 ft and 2 runways.", "score": 0.9},
    {"url": "https://ourairports.com/airports/X", "title": "raw table", "content": "elevation 2459 runways 2", "score": 0.99},
]


def plan_json(queries):
    return json.dumps({"queries": queries})


def default_llm(req):
    text = req.messages[-1]["content"]
    if "Write at most" in text or '"queries"' in text and "Web search results" not in text:
        node = "Airport 0" if "Airport 0" in text else "Airport 1"  # task-specific query, as a real plan would be
        return plan_json([{"id": "q1", "query": f"{node} elevation runways", "targets": ["elevation", "runways"]}])
    return json.dumps({"predictions": {"elevation": {"value": "2459", "unit": "ft", "sources": [1]},
                                       "runways": {"value": 2, "unit": None, "sources": [1]}}})


@pytest.fixture
def setup(tmp_path):
    write_tasks(tmp_path / "tasks.csv")
    (tmp_path / "ground_truth.csv").write_text("task_id,prop_key,value,value_type\n")

    def make(llm=default_llm, search=None, **extra):
        cfg = {
            "inputs": {"tasks": str(tmp_path / "tasks.csv")},
            "runs_per_task": 3,
            "output_dir": str(tmp_path / "out"),
            "search": {"tool": "mock"},
            "max_subqueries": 2,
            "blocked_domains": ["ourairports.com"],
            "pfe_tokenizer": "simple",
            "variants": [{"id": "search_evidence", "plan": str(ROOT / "prompts/agent/plan.v1.txt"),
                          "synthesize": str(ROOT / "prompts/agent/synth.v1.txt")}],
            "models": [{"id": "m1", "provider": "mock", "params": {"temperature": 0.7, "max_tokens": 64}, "max_concurrency": 1}],
            **extra,
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg))
        config = load_config(path)
        provider = MockProvider(config.models[0], responder=llm)
        tool = search or MockSearch({}, responder=lambda q: GOOD)
        return Runner(config, providers={"m1": provider}, search_tool=tool, log=lambda m: None), provider, tool

    return make


def logs(runner):
    return [json.loads(line) for line in (runner.out_dir / "runs.jsonl").read_text().splitlines()]


def test_end_to_end(setup, monkeypatch):
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (_ for _ in ()).throw(AssertionError("ground truth read"))
                        if Path(str(f)).name == "ground_truth.csv" else real_open(f, *a, **k))
    runner, provider, tool = setup()
    assert runner.run() == {"ok": 6}
    rows = read_results(runner.out_dir / "results.csv")
    assert len(rows) == 6 and all(r["strategy"] == "B" for r in rows)
    r = rows[0]
    assert json.loads(r["predictions"]) == {"Airport.elevation": "2459 ft", "Airport.runways": 2}
    assert json.loads(r["evidence"]) == {"Airport.elevation": [GOOD[0]["url"]], "Airport.runways": [GOOD[0]["url"]]}
    assert int(r["search_calls"]) == 1 and int(r["pfe_tokens"]) > 0
    log = logs(runner)[0]
    # The blocked source is logged but never passed to synthesis.
    assert any(x["blocked"] for x in log["results"])
    assert all("ourairports" not in e["url"] for e in log["evidence_passed"])
    assert [c["step"] for c in log["calls"]] == ["plan", "synthesize"]
    assert "leak_filtered:https://ourairports.com/airports/X" in log["validation_events"]
    assert int(r["llm_calls"]) == 2 and r["cost_usd"] == ""  # no pricing in this config
    assert set(log["effort"]) == {"template", "input", "context", "targets"}
    assert log["effort"]["template"]["tokens"] == int(r["pfe_tokens"]) or log["effort"]["template"]["tokens"] > 0
    assert log["effort"]["input"]["chars"] > 0


def test_reuse_retrieval_searches_once(setup):
    runner, provider, tool = setup(reuse_retrieval=True)
    runner.run()
    assert len(tool.calls) == 2  # one query per task, runs 2-3 reuse it
    plans = [c for c in provider.calls if "Write at most" in c.messages[-1]["content"]]
    assert len(plans) == 2  # the plan of run 1 is reused through the LLM cache
    rows = read_results(runner.out_dir / "results.csv")
    assert sorted(int(r["search_calls"]) for r in rows) == [0, 0, 0, 0, 1, 1]
    synth_runs = sorted(c.run_index for c in provider.calls if "Web search results" in c.messages[-1]["content"])
    assert synth_runs == [1, 1, 2, 2, 3, 3]  # synthesis still runs 3 times per task


def test_uncited_values_become_null(setup):
    def llm(req):
        if "Write at most" in req.messages[-1]["content"]:
            return plan_json([{"query": "q", "targets": ["elevation", "runways"]}])
        return json.dumps({"predictions": {"elevation": {"value": "100 m", "sources": []},
                                           "runways": {"value": 3, "sources": [7]}}})

    runner, _, _ = setup(llm=llm)
    runner.run()
    row = read_results(runner.out_dir / "results.csv")[0]
    assert json.loads(row["predictions"]) == {"Airport.elevation": None, "Airport.runways": None}
    events = logs(runner)[0]["validation_events"]
    assert "uncited_value:Airport.elevation" in events and any(e.startswith("invalid_citation") for e in events)


def test_memory_mode_keeps_uncited(setup):
    def llm(req):
        if "Write at most" in req.messages[-1]["content"]:
            return plan_json([{"query": "q", "targets": ["elevation", "runways"]}])
        return json.dumps({"predictions": {"elevation": {"value": "100", "unit": "m", "sources": []}, "runways": None}})

    runner, _, _ = setup(llm=llm, variants=[{"id": "mem", "plan": str(ROOT / "prompts/agent/plan.v1.txt"),
                                             "synthesize": str(ROOT / "prompts/agent/synth_memory.v1.txt"),
                                             "synthesis": "evidence_and_memory"}])
    runner.run()
    row = read_results(runner.out_dir / "results.csv")[0]
    assert json.loads(row["predictions"]) == {"Airport.elevation": "100 m", "Airport.runways": None}
    assert "from_memory:Airport.elevation" in logs(runner)[0]["validation_events"]


def test_plan_invalid_after_repair(setup):
    runner, provider, tool = setup(llm=lambda req: "I cannot plan.")
    assert runner.run() == {"plan_invalid": 6}
    log = logs(runner)[0]
    assert [c["step"] for c in log["calls"]] == ["plan", "plan_repair"] and tool.calls == []
    assert json.loads(read_results(runner.out_dir / "results.csv")[0]["predictions"]) == {
        "Airport.elevation": None, "Airport.runways": None}


def test_retrieval_failed(setup):
    from llmcore.types import ProviderError

    runner, _, _ = setup(search=MockSearch({}, responder=lambda q: ProviderError("client", "boom")))
    assert runner.run() == {"retrieval_failed": 6}


def test_parse_plan_defaults_and_truncation():
    task = Task("t", "v", "Airport", "SBGR", H, G)
    labels = {"elevation": "Airport.elevation", "runways": "Airport.runways"}
    text = plan_json([{"query": "a", "targets": ["elevation"]}, {"query": "b", "targets": ["elevation"]},
                      {"query": "c", "targets": ["elevation"]}])
    queries, events, ok = steps.parse_plan(text, task, labels, max_queries=2)
    assert ok and [q["query"] for q in queries][:2] == ["a", "b"]
    assert queries[-1]["source"] == "default" and queries[-1]["targets"] == ["Airport.runways"]
    assert "Guarulhos International" in queries[-1]["query"]
    assert any(e.startswith("plan_truncated") for e in events)


def test_evidence_keeps_one_per_query():
    res = {"q1": [SearchResult("https://a/1", "A1", "x " * 400, 0.9), SearchResult("https://a/2", "A2", "y " * 400, 0.8)],
           "q2": [SearchResult("https://b/1", "B1", "z " * 400, 0.1)]}
    ev, logged, events = steps.build_evidence(res, [], max_tokens=900, count_tokens=lambda s: len(s.split()), max_chars=2000)
    assert [e["url"] for e in ev] == ["https://a/1", "https://b/1"]  # best of each query before higher-scored extras
    assert any(e.startswith("evidence_truncated") for e in events) and len(logged) == 3


def test_blocklist():
    assert is_blocked("https://www.ourairports.com/x", ["ourairports.com"])
    assert is_blocked("https://data.ourairports.com/x", ["ourairports.com"])
    assert is_blocked("https://github.com/jpatokal/openflights/blob/x", ["github.com/jpatokal"])
    assert not is_blocked("https://github.com/other/repo", ["github.com/jpatokal"])
    assert not is_blocked("https://en.wikipedia.org/wiki/X", ["ourairports.com"])


def test_flatten_units():
    assert steps.flatten("2459", "ft") == "2459 ft"
    assert steps.flatten("2459 ft", "ft") == "2459 ft"
    assert steps.flatten("null", "ft") is None


def test_tavily_request(monkeypatch):
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        sent["auth"] = req.headers.get("Authorization")
        return Resp(json.dumps({"results": [{"url": "https://x", "title": "t", "content": "c", "score": 0.5}]}).encode())

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tool = TavilySearch({"depth": "basic"})
    resp = tool.search("Guarulhos elevation", 5, ["ourairports.com", "github.com/jpatokal"])
    assert sent["auth"] == "Bearer tvly-test"
    assert sent["body"]["max_results"] == 5 and sent["body"]["include_answer"] is False
    assert sent["body"]["exclude_domains"] == ["ourairports.com", "github.com"]
    assert resp.results[0].url == "https://x" and resp.credits == 1.0


def test_ollama_search_filters_blocked_domains_and_ranks_chunks(monkeypatch):
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    page = " ".join(["filler " * 70, "Guarulhos elevation is 2459 ft.", "filler " * 70, "runways 2"])
    items = [{"url": "https://ourairports.com/airports/SBGR", "title": "leak", "content": "elevation 2459"},
             {"url": "https://example.org/short", "title": "short", "content": "Guarulhos airport"},
             {"url": "https://en.wikipedia.org/wiki/GRU", "title": "GRU", "content": page},
             {"url": "https://example.org/third", "title": "third", "content": "more filler"}]

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        sent["auth"] = req.headers.get("Authorization")
        return Resp(json.dumps({"results": items}).encode())

    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tool = OllamaSearch({"chunk_chars": 100, "chunks_per_source": 2})
    # Stand-in for the embedding model: a chunk is as similar to the query as its share of "elevation"/"runways".
    monkeypatch.setattr(tool, "_similarities", lambda q, chunks: [
        sum(w in c for w in ("elevation", "runways")) / 2 for c in chunks])
    resp = tool.search("Guarulhos elevation", 2, ["ourairports.com"])
    assert sent["auth"] == "Bearer ollama-test"
    assert sent["body"] == {"query": "Guarulhos elevation", "max_results": 10}  # over-fetch to replace blocked results
    assert [r.url for r in resp.results] == ["https://en.wikipedia.org/wiki/GRU", "https://example.org/short"]
    best = resp.results[0]
    assert best.score == 0.5 and best.content.count(" [...] ") == 1
    assert "2459 ft" in best.content and "runways 2" in best.content and len(best.content) <= 2 * 100 + 7
    assert resp.raw["blocked_dropped"] == 1 and resp.credits == 1.0
    assert tool.search("x", 3, []) and sent["body"]["max_results"] == 3  # no blocklist: ask for k


def test_ollama_switches_keys_on_rate_limit(monkeypatch):
    import io
    import urllib.error

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    limited = {"Bearer key-a"}
    used = []

    def fake_urlopen(req, timeout):
        auth = req.headers.get("Authorization")
        used.append(auth)
        if auth in limited:
            raise urllib.error.HTTPError(req.full_url, 429, "limit", {"Retry-After": "120"}, io.BytesIO(b"session limit"))
        return Resp(json.dumps({"results": [{"url": "https://a.org", "title": "A", "content": "alpha beta"}]}).encode())

    monkeypatch.setenv("KEY_A", "key-a")
    monkeypatch.setenv("KEY_B", "key-b")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tool = OllamaSearch({"api_key_env": ["KEY_A", "KEY_B"], "ranking": "none"})
    assert tool.search("q", 1, []).results[0].url == "https://a.org"
    assert used == ["Bearer key-a", "Bearer key-b"]  # the limited key is skipped within the same search
    tool.search("q2", 1, [])
    assert used[-1] == "Bearer key-b" and len(used) == 3  # and not tried again until its Retry-After has passed
    limited.add("Bearer key-b")
    with pytest.raises(Exception) as info:
        tool.search("q3", 1, [])
    assert info.value.kind == "rate_limit" and 100 < info.value.retry_after <= 120  # both limited: the earliest reset
    assert tool.key_stats["switches"] == 2


def test_ollama_chunks_cut_between_words_and_cap_the_page(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test")
    tool = OllamaSearch({"chunk_chars": 20, "max_page_chars": 60, "ranking": "none"})
    chunks = tool.chunk("alpha  beta\ngamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi")
    assert all(len(c) <= 20 for c in chunks) and " ".join(chunks).startswith("alpha beta gamma delta")
    assert len(" ".join(chunks)) <= 60
    with pytest.raises(Exception):
        OllamaSearch({"ranking": "bm25"})


def test_search_buffer_normalizes_and_dedupes(tmp_path):
    import threading
    import time as _time

    from strategy_b.search import CachedSearch

    def slow(q):
        _time.sleep(0.05)
        return GOOD[:1]

    tool = MockSearch({}, responder=slow)
    cache = CachedSearch(tool, tmp_path / "sc")
    threads = [threading.Thread(target=cache.search, args=("Perth airport elevation", 5, [])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(tool.calls) == 1 and cache.stats["cache_hits"] == 3  # concurrent duplicates wait for the first
    assert cache.search("  perth AIRPORT   elevation? ", 5, []).cached and len(tool.calls) == 1
    # A new session with --no-search-cache searches again once, then reuses its own result.
    fresh = CachedSearch(tool, tmp_path / "sc", read_cache=False)
    fresh.search("Perth airport elevation", 5, [])
    fresh.search("Perth airport elevation", 5, [])
    assert len(tool.calls) == 2


def test_search_credit_cap(tmp_path):
    from llmcore.types import ProviderError
    from strategy_b.search import CachedSearch

    tool = MockSearch({}, responder=lambda q: GOOD[:1])
    tool.credits_per_call = lambda: 1.0
    cache = CachedSearch(tool, tmp_path / "sc", max_credits=2)
    orig = tool.search

    def billed(q, k, ex):
        r = orig(q, k, ex)
        r.credits = 1.0
        return r

    tool.search = billed
    cache.search("a", 5, [])
    cache.search("b", 5, [])
    assert cache.search("a", 5, []).cached  # cached searches are free under the cap
    with pytest.raises(ProviderError) as err:
        cache.search("c", 5, [])
    assert err.value.kind == "budget" and not err.value.retryable and cache.stats["budget_refused"] == 1


def test_plan_v2_identifies_the_entity_and_partial_searches_are_retried(setup, tmp_path):
    from llmcore.types import ProviderError

    notes = tmp_path / "notes.yaml"
    notes.write_text("runways: number of runways of the airport\n")

    def llm(req):
        text = req.messages[-1]["content"]
        if "Identify the entity" in text:
            assert "- runways: number of runways of the airport" in text
            return json.dumps({"entity": "Guarulhos", "queries": [
                {"id": "q1", "query": "Guarulhos elevation", "targets": ["elevation"]},
                {"id": "q2", "query": "Guarulhos runways", "targets": ["runways"]}]})
        assert "- runways: number of runways of the airport" in text
        return default_llm(req)

    def search(q):
        return ProviderError("rate_limit", "429") if "runways" in q else GOOD

    runner, _, _ = setup(llm=llm, search=MockSearch({}, responder=search), max_search_retries=0,
                         inputs={"tasks": str(tmp_path / "tasks.csv"), "property_notes": str(notes)},
                         variants=[{"id": "search_evidence", "plan": str(ROOT / "prompts/agent/plan.v2.txt"),
                                    "synthesize": str(ROOT / "prompts/agent/synth.v2.txt")}])
    assert runner.run() == {"search_partial": 6}
    log = logs(runner)[0]
    assert log["plan_entity"] == "Guarulhos"
    assert [q["error"] is None for q in log["queries"]] == [True, False]
    from strategy_b.cli import FAILED_STATUSES
    assert "search_partial" in FAILED_STATUSES


class _StubEmbedder:
    """Bag-of-words vectors over a fixed vocabulary; counts how many texts it embeds."""
    VOCAB = ["elevation", "runways", "filler", "airport", "guarulhos"]

    def __init__(self):
        self.n = 0

    def _vec(self, text):
        self.n += 1
        return [text.lower().count(w) + 0.01 for w in self.VOCAB]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def query_embed(self, text):
        return [self._vec(t) for t in ([text] if isinstance(text, str) else text)]


def test_ollama_embedding_cache_gives_identical_results_without_embedding_again(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test")
    pages = [{"url": f"https://x/{i}", "title": "t", "content": " ".join(["filler"] * 60 + ["elevation 2459 ft"] + ["filler"] * 60)}
             for i in range(3)]
    first = OllamaSearch({"chunk_chars": 80, "embed_cache": str(tmp_path / "emb.sqlite")})
    first._embedder = stub = _StubEmbedder()
    a = first._results("Guarulhos elevation", pages)
    assert stub.n > 0 and first.embed_stats["from_cache"] == 0
    second = OllamaSearch({"chunk_chars": 80, "embed_cache": str(tmp_path / "emb.sqlite")})
    second._embedder = stub2 = _StubEmbedder()
    b = second._results("Guarulhos elevation", pages)
    assert stub2.n == 0 and second.embed_stats["embedded"] == 0  # everything came from the disk cache
    assert [(r.content, r.score) for r in a] == [(r.content, r.score) for r in b]


def test_ollama_bm25_prefilter_embeds_only_the_best_lexical_chunks(monkeypatch):
    from strategy_b.search import bm25

    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test")
    assert bm25("runway elevation", ["the elevation is 10 ft", "nothing here", "runway and elevation"])[1] == 0.0
    text = " ".join([f"filler{i}" for i in range(200)] + ["Guarulhos elevation 2459 ft"] + [f"filler{i}" for i in range(200, 400)])
    tool = OllamaSearch({"chunk_chars": 60, "prefilter_chunks": 4})
    tool._embedder = stub = _StubEmbedder()
    [r] = tool._results("Guarulhos elevation", [{"url": "u", "title": "t", "content": text}])
    assert "2459" in r.content
    assert stub.n == 4 + 1  # 4 chunks and the query, out of many more chunks


def test_search_waits_out_a_rate_limit_with_retry_after(tmp_path):
    from llmcore.types import ProviderError
    from strategy_b.search import CachedSearch

    calls, slept = [], []

    def responder(q):
        calls.append(q)
        if len(calls) <= 2:
            err = ProviderError("rate_limit", "hourly search limit", 429)
            err.retry_after = 600.0
            return err
        return GOOD[:1]

    cache = CachedSearch(MockSearch({}, responder=responder), tmp_path / "sc", max_retries=0, sleep=slept.append,
                         max_rate_limit_wait_s=7200)
    assert cache.search("Guarulhos elevation", 5, []).results  # waited twice, then succeeded
    assert slept == [601.0, 601.0] and cache.stats["rate_limit_waits"] == 2
    no_wait = CachedSearch(MockSearch({}, responder=lambda q: responder(q) if False else (
        setattr(e := ProviderError("rate_limit", "limit", 429), "retry_after", 600.0) or e)), tmp_path / "sc2",
        max_retries=0, sleep=slept.append)
    with pytest.raises(ProviderError):
        no_wait.search("x", 5, [])  # default: no waiting, the run is marked failed and retried later

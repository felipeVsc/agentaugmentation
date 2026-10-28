"""Ollama provider against a fake Ollama server (no Ollama needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from llmcore.config import ModelConfig
from llmcore.providers import load_provider
from llmcore.types import ProviderError, Request
from strategy_a.config import load_config
from strategy_a.results import read_results
from strategy_a.runner import Runner

from test_strategy_a import ROOT, write_tasks


class FakeOllama(BaseHTTPRequestHandler):
    chats: list = []
    models = ["llama3.2:1b"]

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(200, {"version": "0.33.3"})
        if self.path == "/api/tags":
            return self._send(200, {"models": [
                {"name": m, "model": m, "digest": "abc123def4567890", "size": 1, "details": {"quantization_level": "Q8_0"}}
                for m in self.models
            ]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.auth = self.headers.get("Authorization")
        if self.path == "/api/show":
            return self._send(200, {
                "details": {"parameter_size": "1.2B", "quantization_level": "Q8_0", "family": "llama"},
                "model_info": {"llama.context_length": 131072},
                "capabilities": ["completion"],
            })
        if self.path == "/api/chat":
            FakeOllama.chats.append(payload)
            if payload["model"] not in self.models:
                return self._send(404, {"error": f"model '{payload['model']}' not found"})
            return self._send(200, {
                "model": payload["model"],
                "message": {"role": "assistant", "content": '{"elevation": 750, "runways": 2}'},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 120,
                "eval_count": 12,
                "total_duration": 1_000_000,
            })
        self._send(404, {"error": "not found"})


@pytest.fixture
def server():
    FakeOllama.chats = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def model(url, **kw):
    entry = {"id": "llama", "provider": "ollama", "model": "llama3.2:1b", "params": {"temperature": 0.7, "max_tokens": 64},
             "options": {"base_url": url, "num_ctx": 4096, "seed": 10, "think": False, **kw.pop("options", {})}, **kw}
    return ModelConfig.from_dict(entry)


def test_generate_body_and_usage(server):
    p = load_provider(model(server))
    r = p.generate(Request(system="sys", messages=[{"role": "user", "content": "hi"}], params=p.cfg.params, run_index=3))
    sent = FakeOllama.chats[-1]
    assert sent["stream"] is False and sent["think"] is False and sent["keep_alive"] == "30m"
    assert sent["options"] == {"num_predict": 64, "num_ctx": 4096, "temperature": 0.7, "seed": 13}
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert r.text == '{"elevation": 750, "runways": 2}'
    assert (r.tokens_in, r.tokens_out, r.finish_reason) == (120, 12, "stop")
    assert r.model_version == "llama3.2:1b@abc123def456 (Q8_0)"
    assert p.describe()["context_length"] == 131072


def test_extra_options_and_tokenizer_counting(server):
    cfg = model(server, tokenizer="simple", params={"temperature": 0.5, "max_tokens": 10, "extra": {"top_k": 20, "format": "json"}})
    p = load_provider(cfg)
    r = p.generate(Request(system=None, messages=[{"role": "user", "content": "x"}], params=cfg.params, run_index=1))
    sent = FakeOllama.chats[-1]
    assert sent["options"]["top_k"] == 20 and sent["format"] == "json" and "format" not in sent["options"]
    assert r.tokens_in is None  # tokenizer set: the runner counts the full prompt (prompt cache makes Ollama undercount)
    assert r.raw["prompt_eval_count"] == 120


def test_missing_model_and_server(server):
    p = load_provider(model(server, model="qwen3:14b"))
    with pytest.raises(ProviderError, match="ollama pull qwen3:14b"):
        p.describe()
    with pytest.raises(ProviderError) as err:
        p.generate(Request(system=None, messages=[{"role": "user", "content": "x"}], params=p.cfg.params, run_index=1))
    assert err.value.kind == "client" and not err.value.retryable
    down = load_provider(model("http://127.0.0.1:9"))
    with pytest.raises(ProviderError) as err:
        down.generate(Request(system=None, messages=[{"role": "user", "content": "x"}], params=down.cfg.params, run_index=1))
    assert err.value.kind == "connection" and "ollama serve" in str(err.value)


def test_runner_with_ollama(server, tmp_path):
    write_tasks(tmp_path / "tasks.csv")
    cfg = {
        "inputs": {"tasks": str(tmp_path / "tasks.csv")},
        "runs_per_task": 2,
        "pfe_tokenizer": "simple",
        "output_dir": str(tmp_path / "out"),
        "templates": [{"id": "base_zero_shot", "file": str(ROOT / "templates/base_zero_shot.v1.txt")}],
        "model_defaults": {"provider": "ollama", "params": {"temperature": 0.7, "max_tokens": 64},
                           "options": {"base_url": server, "seed": 0}},
        "models": [{"id": "llama", "model": "llama3.2:1b", "tokenizer": "simple",
                    "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0, "as_of": "2026-10-27"}}],
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    runner = Runner(load_config(tmp_path / "cfg.yaml"), log=lambda m: None)
    assert runner.run() == {"ok": 4}
    rows = read_results(tmp_path / "out" / "results.csv")
    assert all(json.loads(r["predictions"]) == {"Airport.elevation": 750, "Airport.runways": 2} for r in rows)
    assert all(int(r["tokens_in"]) > 0 and r["tokens_out"] == "12" for r in rows)
    assert all(r["llm_calls"] == "1" and r["tokens_reasoning"] == "0" for r in rows)  # no thinking text
    assert all(abs(float(r["cost_usd"]) - (int(r["tokens_in"]) * 1.0 + 12 * 2.0) / 1e6) < 1e-12 for r in rows)
    assert sorted(c["options"]["seed"] for c in FakeOllama.chats) == [1, 1, 2, 2]
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["model_details"]["llama"]["digest"] == "abc123def4567890"
    assert manifest["model_versions"]["llama"] == ["llama3.2:1b@abc123def456 (Q8_0)"]
    log = json.loads((tmp_path / "out" / "runs.jsonl").read_text().splitlines()[0])
    assert log["usage"]["prompt_eval_count"] == 120 and log["usage"]["tokens_in_counted_with"] == "simple"
    assert log["effort"]["template"]["tokens"] == int(rows[0]["pfe_tokens"])
    assert set(log["effort"]) == {"template", "input", "context", "targets"}


def test_cloud_sends_api_key(server, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-key")
    p = load_provider(model(server))
    p.cloud = True  # the fake server stands in for https://ollama.com
    p.api_key = "secret-key"
    p.describe()
    r = p.generate(Request(system=None, messages=[{"role": "user", "content": "x"}], params=p.cfg.params, run_index=1))
    assert FakeOllama.auth == "Bearer secret-key"
    assert r.model_version == "llama3.2:1b (Ollama Cloud)"


def test_cloud_requires_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        load_provider(model("https://ollama.com"))

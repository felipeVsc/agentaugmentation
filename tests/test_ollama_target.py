"""Choosing between Ollama Cloud and an on-premise Ollama server (llmcore.config.use_ollama, --ollama)."""

import pytest

from llmcore.cache import cache_key
from llmcore.config import ConfigError, ModelConfig, endpoint_output_dir, ollama_url, use_ollama
from llmcore.types import Request
from strategy_a.config import load_config as load_a
from strategy_b.config import load_config as load_b

T = {"temperature": 0.7}
CLOUD = {"id": "cloud-gpt-oss-20b", "provider": "ollama", "model": "gpt-oss:20b", "params": T,
         "options": {"base_url": "https://ollama.com", "think": "low"},
         "pricing": {"input_per_mtok": 0.07, "output_per_mtok": 0.3}}


def cloud_model():
    return ModelConfig.from_dict(dict(CLOUD))


def test_ollama_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_url("cloud") == "https://ollama.com"
    assert ollama_url("local") == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    assert ollama_url("local") == "http://gpu-box:11434"
    assert ollama_url("http://10.0.0.5:11434/") == "http://10.0.0.5:11434"
    with pytest.raises(ConfigError):
        ollama_url("somewhere")


def test_cloud_config_to_local(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    m = cloud_model()
    assert use_ollama([m], "local") == "local"
    assert m.id == "local-gpt-oss-20b"
    assert m.options == {"base_url": "http://localhost:11434", "think": "low"}
    assert m.pricing is None


def test_same_endpoint_changes_nothing():
    m = cloud_model()
    before = cache_key(m, Request(None, [{"role": "user", "content": "x"}], m.params, 1))
    assert use_ollama([m], "cloud") is None
    assert m.id == "cloud-gpt-oss-20b" and m.pricing is not None
    # the cached cloud responses of the paper stay valid
    assert cache_key(m, Request(None, [{"role": "user", "content": "x"}], m.params, 1)) == before
    assert use_ollama([m], None) is None


def test_cache_key_differs_between_endpoints():
    cloud, local = cloud_model(), cloud_model()
    use_ollama([local], "local")
    req = Request(None, [{"role": "user", "content": "x"}], cloud.params, 1)
    assert cache_key(cloud, req) != cache_key(local, req)


def test_local_config_to_cloud_and_non_ollama_untouched():
    local = ModelConfig.from_dict({"id": "gemma", "provider": "ollama", "model": "gemma4:31b", "params": T})
    other = ModelConfig.from_dict({"id": "mock-1", "provider": "mock", "params": T})
    assert use_ollama([local, other], "cloud") == "cloud"
    assert local.id == "cloud-gemma" and local.options["base_url"] == "https://ollama.com"
    assert other.id == "mock-1" and "base_url" not in other.options


def test_endpoint_output_dir():
    assert endpoint_output_dir("results/a/strategy_a/", "local") == "results/a/strategy_a_local/"
    assert endpoint_output_dir("results/a/strategy_a", None) == "results/a/strategy_a"


def test_strategy_configs(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    tasks = tmp_path / "tasks.csv"
    tasks.write_text("task_id\n")
    a = tmp_path / "a.yaml"
    a.write_text(f"""inputs: {{tasks: {tasks}}}
output_dir: results/x/strategy_a/
templates: [{{id: zero, file: t.v1.txt}}]
model_defaults: {{provider: ollama, params: {{temperature: 0.7}}}}
models:
  - {{id: cloud-gpt-oss-20b, model: "gpt-oss:20b", options: {{base_url: "https://ollama.com"}}}}
""")
    cfg = load_a(a)
    assert cfg.models[0].id == "cloud-gpt-oss-20b" and cfg.output_dir == "results/x/strategy_a/"
    cfg = load_a(a, ollama="local")
    assert cfg.models[0].id == "local-gpt-oss-20b" and cfg.output_dir == "results/x/strategy_a_local/"

    b = tmp_path / "b.yaml"
    b.write_text(f"""inputs: {{tasks: {tasks}}}
output_dir: results/x/strategy_b/
search_cache_dir: results/x/search_cache/
ollama: local
variants: [{{id: search_evidence, plan: p.v1.txt, synthesize: s.v1.txt}}]
model_defaults: {{provider: ollama, params: {{temperature: 0.7}}}}
models:
  - {{id: cloud-gemma4-31b, model: "gemma4:31b", options: {{base_url: "https://ollama.com"}}}}
""")
    cfg = load_b(b)
    assert cfg.models[0].id == "local-gemma4-31b" and cfg.output_dir == "results/x/strategy_b_local/"
    assert cfg.search_cache_dir == "results/x/search_cache/"   # searches are shared between endpoints
    cfg = load_b(b, ollama="cloud")
    assert cfg.models[0].id == "cloud-gemma4-31b" and cfg.output_dir == "results/x/strategy_b/"

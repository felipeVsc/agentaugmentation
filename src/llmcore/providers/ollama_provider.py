"""Ollama, through its native API (``/api/chat``).

The native API is used instead of Ollama's OpenAI-compatible endpoint because
it lets us set the context window (``num_ctx``), turn thinking off, and get the
model digest, quantization and timings that a reproducible run needs.

Options (``options:`` in the model entry), all optional:

- ``base_url``: server URL; default ``$OLLAMA_HOST`` or ``http://localhost:11434``
  (the top-level ``base_url`` field works too).
- ``num_ctx``: context window in tokens (default 8192). Ollama silently drops the
  start of a prompt longer than this, so ``strategy-a check`` warns when a
  prompt may not fit. Keep it fixed per model: changing it reloads the model.
- ``keep_alive``: how long the model stays loaded between calls (default ``30m``).
- ``think``: ``false`` turns thinking off on thinking models (qwen3, deepseek-r1,
  gpt-oss); leave unset for models without a thinking mode.
- ``seed`` and ``seed_per_run`` (default true): send ``seed + run_index``, so the
  5 runs differ (temperature > 0) but each run can be reproduced exactly.
- ``tokens_in_from``: ``reported`` (Ollama's ``prompt_eval_count``, which counts
  the chat template too) or ``tokenizer`` (count system + user text with the
  model's ``tokenizer``). Some Ollama versions report only the tokens they did
  not find in the prompt cache; Ollama 0.33.3 reported the full count on repeated
  prompts. ``prompt_eval_count`` is always kept in the run log.
  Default: ``tokenizer`` when the model entry sets ``tokenizer``, else ``reported``.

Ollama-specific generation options (``top_k``, ``repeat_penalty``, ``min_p``, ...)
go under ``params.extra`` and are sent in ``options``; ``format`` goes there too
and is sent at the top level.

Ollama Cloud: set ``base_url: https://ollama.com`` (in ``options`` or the model entry) and put the
key in ``OLLAMA_API_KEY`` (or the variable named by ``api_key_env``); it is sent as a Bearer token.
Use model names without the ``-cloud`` suffix (``gpt-oss:120b``). Cloud models are not pulled, so
``describe`` reports what ``/api/show`` gives, if anything; CodeCarbon cannot measure them, use
``energy: {method: ecologits, ...}`` or ``none``. (Alternatively, ``ollama signin`` on the local
server and use the ``-cloud`` names through ``localhost``; no key is needed then.)

Energy: ``energy: {method: codecarbon}`` measures this machine (one call at a
time, ``max_concurrency: 1``). EcoLogits can give a datacenter-style estimate
instead with ``ecologits_provider: huggingface_hub`` and the closest
``ecologits_model`` (e.g. ``meta-llama/Llama-3.2-1B-Instruct``).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ..config import ConfigError
from ..types import ProviderError, Request, Response, http_error_kind
from .base import Provider

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"
DEFAULT_NUM_CTX = 8192
TOP_LEVEL = {"format", "think", "keep_alive"}


def _full_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


class OllamaProvider(Provider):
    supports_batch = False
    ecologits_provider = "huggingface_hub"

    def __init__(self, cfg):
        super().__init__(cfg)
        o = cfg.options
        url = o.get("base_url") or cfg.base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_URL
        if not url.startswith("http"):
            url = f"http://{url}"  # OLLAMA_HOST is often given as host:port
        self.base_url = url.rstrip("/")
        self.cloud = "ollama.com" in self.base_url
        key_env = cfg.api_key_env or "OLLAMA_API_KEY"
        self.api_key = os.environ.get(key_env)
        if self.cloud and not self.api_key:
            raise ConfigError(f"model {cfg.id!r}: Ollama Cloud needs an API key; set {key_env} in the environment or .env")
        self.num_ctx = int(o.get("num_ctx", DEFAULT_NUM_CTX))
        self.keep_alive = o.get("keep_alive", "30m")
        self.think = o.get("think")
        self.seed = o.get("seed", 0)
        self.seed_per_run = bool(o.get("seed_per_run", True))
        self.tokens_in_from = o.get("tokens_in_from", "tokenizer" if cfg.tokenizer else "reported")
        if self.tokens_in_from not in ("reported", "tokenizer"):
            raise ValueError(f"model {cfg.id!r}: options.tokens_in_from must be reported or tokenizer")
        self._info: dict | None = None
        self._info_lock = threading.Lock()

    # HTTP ---------------------------------------------------------------

    def _http(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.cfg.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(http_error_kind(exc.code), f"HTTP {exc.code} from Ollama: {body}", exc.code) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise ProviderError("timeout", f"Ollama did not answer within {timeout or self.cfg.timeout_s}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ProviderError("timeout", f"Ollama timeout: {exc.reason}") from exc
            raise ProviderError(
                "connection", f"cannot reach Ollama at {self.base_url} ({exc.reason}); is `ollama serve` running?"
            ) from exc

    # Model information ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Server version, model digest, size, quantization and context length (for the manifest)."""
        with self._info_lock:
            if self._info is not None:
                return self._info
            if self.cloud:  # nothing is pulled; record what /api/show offers, without failing
                name = self.cfg.model
                try:
                    show = self._http("POST", "/api/show", {"model": name}, timeout=30)
                except ProviderError as exc:
                    show = {"error": str(exc)}
                self._info = {"model": name, "cloud": True, "base_url": self.base_url,
                              "details": show.get("details"), "capabilities": show.get("capabilities"),
                              "show_error": show.get("error")}
                return self._info
            version = self._http("GET", "/api/version", timeout=10).get("version")
            name = _full_name(self.cfg.model)
            tags = self._http("GET", "/api/tags", timeout=10).get("models", [])
            match = next((m for m in tags if m.get("name") == name or m.get("model") == name), None)
            if match is None:
                raise ProviderError(
                    "client",
                    f"model {self.cfg.model!r} is not in Ollama at {self.base_url}; run: ollama pull {self.cfg.model}",
                )
            show = self._http("POST", "/api/show", {"model": name}, timeout=30)
            model_info = show.get("model_info") or {}
            ctx_len = next((v for k, v in model_info.items() if k.endswith(".context_length")), None)
            self._info = {
                "ollama_version": version,
                "model": name,
                "digest": match.get("digest"),
                "size_bytes": match.get("size"),
                "details": show.get("details") or match.get("details"),
                "context_length": ctx_len,
                "num_ctx": self.num_ctx,
                "capabilities": show.get("capabilities"),
            }
            if ctx_len and self.num_ctx > ctx_len:
                log.warning("model %s: num_ctx %d exceeds its context length %d", name, self.num_ctx, ctx_len)
            return self._info

    @property
    def model_version(self) -> str:
        try:
            info = self.describe()
        except ProviderError:
            return _full_name(self.cfg.model)
        if info.get("cloud"):
            return f"{info['model']} (Ollama Cloud)"
        digest = (info.get("digest") or "")[:12]
        quant = (info.get("details") or {}).get("quantization_level")
        return f"{info['model']}@{digest}" + (f" ({quant})" if quant else "")

    # Generation -----------------------------------------------------------

    def body(self, request: Request) -> dict[str, Any]:
        temperature, top_p, max_tokens, extra = self.gen_params(request)
        options: dict[str, Any] = {"num_predict": max_tokens, "num_ctx": self.num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if self.seed is not None:
            options["seed"] = int(self.seed) + (request.run_index if self.seed_per_run else 0)
        top = {k: extra.pop(k) for k in list(extra) if k in TOP_LEVEL}
        options.update(extra)
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + request.messages
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if self.think is not None:
            body["think"] = self.think
        body.update(top)
        return body

    def generate(self, request: Request) -> Response:
        body = self.body(request)
        version = self.model_version  # fetched once; also checks the model is pulled
        start = time.perf_counter()
        data = self._http("POST", "/api/chat", body)
        latency = time.perf_counter() - start
        message = data.get("message") or {}
        reason = data.get("done_reason")
        reported_in = data.get("prompt_eval_count")
        raw = {k: v for k, v in data.items() if k != "message"}
        raw["thinking"] = message.get("thinking")
        raw["seed"] = body["options"].get("seed")
        raw["num_ctx"] = self.num_ctx
        if reported_in is not None and reported_in >= self.num_ctx:
            raw["prompt_may_be_truncated"] = True
            log.warning("model %s: prompt filled num_ctx=%d and was probably truncated", self.cfg.model, self.num_ctx)
        return Response(
            text=message.get("content") or "",
            tokens_in=reported_in if self.tokens_in_from == "reported" else None,
            tokens_out=data.get("eval_count"),  # includes thinking tokens when thinking is on
            latency_s=latency,
            finish_reason="stop" if reason == "stop" else "length" if reason == "length" else f"other:{reason}",
            model_version=version,
            raw=raw,
            execution="realtime",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

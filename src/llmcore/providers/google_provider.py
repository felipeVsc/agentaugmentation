"""Google Gemini through the ``google-genai`` SDK (Gemini API or Vertex AI), with batch mode.

Options:
- ``vertexai: true`` with ``project`` and ``location`` to use Vertex AI
  (credentials from ``gcloud auth application-default login``; no API key).
- Fields of ``GenerateContentConfig`` go under ``params.extra``, e.g.
  ``thinking_config: {thinking_budget: 0}`` to turn thinking off on Flash models.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from ..types import ProviderError, Request, Response, http_error_kind
from .base import Provider

_BATCH_MAX_BYTES = 15 * 1024 * 1024  # inline requests are limited to 20 MB per job


def _finish(reason: str | None) -> str:
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return f"other:{reason}"


class GoogleProvider(Provider):
    supports_batch = True
    ecologits_provider = "google_genai"

    def __init__(self, cfg):
        super().__init__(cfg)
        from google import genai
        from google.genai import errors, types

        self._types, self._errors = types, errors
        http_options = types.HttpOptions(timeout=int(cfg.timeout_s * 1000))
        if cfg.options.get("vertexai"):
            self.client = genai.Client(
                vertexai=True,
                project=cfg.options.get("project"),
                location=cfg.options.get("location", "global"),
                http_options=http_options,
            )
        else:
            self.client = genai.Client(api_key=cfg.api_key(), http_options=http_options)

    def _contents(self, request: Request) -> list[dict]:
        role = {"user": "user", "assistant": "model"}
        return [{"role": role[m["role"]], "parts": [{"text": m["content"]}]} for m in request.messages]

    def _config(self, request: Request) -> dict[str, Any]:
        temperature, top_p, max_tokens, extra = self.gen_params(request)
        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if request.system:
            config["system_instruction"] = request.system
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        config.update(extra)
        return config

    def _parse(self, data: dict, latency: float | None, execution: str) -> Response:
        candidates = data.get("candidates") or [{}]
        first = candidates[0] or {}
        parts = (first.get("content") or {}).get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if not p.get("thought"))
        usage = data.get("usage_metadata") or {}
        out = usage.get("candidates_token_count")
        if out is not None or usage.get("thoughts_token_count"):
            out = (out or 0) + (usage.get("thoughts_token_count") or 0)  # thinking tokens are billed as output
        return Response(
            text=text,
            tokens_in=usage.get("prompt_token_count"),
            tokens_out=out,
            latency_s=latency,
            finish_reason=_finish(first.get("finish_reason")),
            model_version=data.get("model_version"),
            raw=data,
            execution=execution,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, self._errors.APIError):
            return ProviderError(http_error_kind(exc.code), str(exc), exc.code)
        name = type(exc).__name__
        if "Timeout" in name:
            return ProviderError("timeout", str(exc))
        if "Connect" in name or "Network" in name:
            return ProviderError("connection", str(exc))
        return ProviderError("client", f"{name}: {exc}")

    def generate(self, request: Request) -> Response:
        start = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self.cfg.model, contents=self._contents(request), config=self._config(request)
            )
        except Exception as exc:  # the SDK raises httpx errors directly on timeouts
            raise self._error(exc) from exc
        return self._parse(resp.model_dump(mode="json"), time.perf_counter() - start, "realtime")

    def count_tokens(self, text: str) -> int | None:
        return self.client.models.count_tokens(model=self.cfg.model, contents=text).total_tokens

    # Batch API with inline requests; each request carries its custom_id in metadata.
    def batch_submit(self, items):
        jobs, current, size = [], [], 0

        def flush():
            job = self.client.batches.create(
                model=self.cfg.model, src=[r for _, r in current], config={"display_name": f"strategy-{self.cfg.id}"}
            )
            jobs.append({"name": job.name, "custom_ids": [cid for cid, _ in current], "n": len(current)})

        for custom_id, request in items:
            inline = {
                "contents": self._contents(request),
                "config": self._config(request),
                "metadata": {"custom_id": custom_id},
            }
            n = len(json.dumps(inline, ensure_ascii=False).encode())
            if current and size + n > _BATCH_MAX_BYTES:
                flush()
                current, size = [], 0
            current.append((custom_id, inline))
            size += n
        if current:
            flush()
        return {"jobs": jobs}

    def batch_poll(self, state):
        states = [self.client.batches.get(name=j["name"]).state.name for j in state["jobs"]]
        if any(s in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED") for s in states):
            return "running"
        return "failed" if all(s == "JOB_STATE_FAILED" for s in states) else "done"

    def batch_fetch(self, state):
        out: dict[str, Response | ProviderError] = {}
        for job_state in state["jobs"]:
            job = self.client.batches.get(name=job_state["name"])
            responses = (job.dest.inlined_responses if job.dest else None) or []
            for i, item in enumerate(responses):
                cid = (item.metadata or {}).get("custom_id") or job_state["custom_ids"][i]
                if item.error:
                    out[cid] = ProviderError("client", str(item.error))
                elif item.response is not None:
                    out[cid] = self._parse(item.response.model_dump(mode="json"), None, "batch")
        return out

"""OpenAI Chat Completions, and any OpenAI-compatible server through ``base_url``
(vLLM, Ollama ``/v1``, OpenRouter, Groq, Together, ...).

Options (``options:`` in the model entry):
- ``max_tokens_param``: ``max_completion_tokens`` (default for api.openai.com) or
  ``max_tokens`` (default when ``base_url`` is set; most compatible servers use it).
- ``batch``: allow the Batch API (default true for api.openai.com, false otherwise).
- ``api_key_optional``: true for local servers that need no key.

Reasoning: put e.g. ``reasoning_effort: none`` under ``params.extra``; it is sent as-is.
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from typing import Any

from ..types import ProviderError, Request, Response, http_error_kind
from .base import Provider

_BATCH_ENDPOINT = "/v1/chat/completions"
_BATCH_MAX_REQUESTS = 50_000
_BATCH_MAX_BYTES = 180 * 1024 * 1024


def _finish(reason: str | None) -> str:
    if reason == "stop":
        return "stop"
    if reason == "length":
        return "length"
    return f"other:{reason}"


class OpenAIProvider(Provider):
    ecologits_provider = "openai"

    def __init__(self, cfg):
        super().__init__(cfg)
        import openai

        self._openai = openai
        optional = cfg.options.get("api_key_optional", cfg.base_url is not None)
        key = cfg.api_key(required=not optional) or "not-needed"
        self.client = openai.OpenAI(api_key=key, base_url=cfg.base_url, timeout=cfg.timeout_s, max_retries=0)
        self.supports_batch = bool(cfg.options.get("batch", cfg.base_url is None))
        self.max_tokens_param = cfg.options.get(
            "max_tokens_param", "max_tokens" if cfg.base_url else "max_completion_tokens"
        )

    def body(self, request: Request) -> dict[str, Any]:
        temperature, top_p, max_tokens, extra = self.gen_params(request)
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + request.messages
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages, self.max_tokens_param: max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        body.update(extra)
        return body

    def _parse(self, data: dict, latency: float | None, execution: str) -> Response:
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return Response(
            text=(choice.get("message") or {}).get("content") or "",
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),  # includes reasoning tokens when present
            latency_s=latency,
            finish_reason=_finish(choice.get("finish_reason")),
            model_version=data.get("model"),
            raw=data,
            execution=execution,
            created_at=datetime.fromtimestamp(data["created"], timezone.utc).isoformat()
            if data.get("created")
            else None,
        )

    def _error(self, exc: Exception) -> ProviderError:
        o = self._openai
        if isinstance(exc, o.APITimeoutError):
            return ProviderError("timeout", str(exc))
        if isinstance(exc, o.APIConnectionError):
            return ProviderError("connection", str(exc))
        if isinstance(exc, o.APIStatusError):
            return ProviderError(http_error_kind(exc.status_code), str(exc), exc.status_code)
        return ProviderError("client", f"{type(exc).__name__}: {exc}")

    def generate(self, request: Request) -> Response:
        body = self.body(request)
        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**body)
        except self._openai.OpenAIError as exc:
            raise self._error(exc) from exc
        return self._parse(resp.model_dump(mode="json"), time.perf_counter() - start, "realtime")

    # Batch API
    def batch_submit(self, items):
        chunks, current, size = [], [], 0
        for custom_id, request in items:
            line = json.dumps(
                {"custom_id": custom_id, "method": "POST", "url": _BATCH_ENDPOINT, "body": self.body(request)},
                ensure_ascii=False,
            ).encode()
            if current and (len(current) >= _BATCH_MAX_REQUESTS or size + len(line) > _BATCH_MAX_BYTES):
                chunks.append(current)
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
        if current:
            chunks.append(current)
        jobs = []
        for lines in chunks:
            payload = io.BytesIO(b"\n".join(lines) + b"\n")
            uploaded = self.client.files.create(file=("batch.jsonl", payload), purpose="batch")
            batch = self.client.batches.create(
                input_file_id=uploaded.id, endpoint=_BATCH_ENDPOINT, completion_window="24h"
            )
            jobs.append({"batch_id": batch.id, "input_file_id": uploaded.id, "n": len(lines)})
        return {"jobs": jobs}

    def batch_poll(self, state):
        statuses = [self.client.batches.retrieve(j["batch_id"]).status for j in state["jobs"]]
        if any(s in ("validating", "in_progress", "finalizing", "cancelling") for s in statuses):
            return "running"
        # "expired" and "cancelled" may still have partial output; fetch what exists.
        return "failed" if all(s == "failed" for s in statuses) else "done"

    def batch_fetch(self, state):
        out: dict[str, Response | ProviderError] = {}
        for job in state["jobs"]:
            batch = self.client.batches.retrieve(job["batch_id"])
            for file_id in (batch.output_file_id, batch.error_file_id):
                if not file_id:
                    continue
                for line in self.client.files.content(file_id).text.splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    resp, err = entry.get("response") or {}, entry.get("error")
                    if err or resp.get("status_code", 200) >= 400:
                        code = resp.get("status_code")
                        out[entry["custom_id"]] = ProviderError(
                            http_error_kind(code), json.dumps(err or resp.get("body"))[:500], code
                        )
                    else:
                        out[entry["custom_id"]] = self._parse(resp["body"], None, "batch")
        return out

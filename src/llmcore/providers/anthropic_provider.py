"""Anthropic Messages API, with the Message Batches API.

``top_p`` is only sent when set: recent Claude models reject temperature and top_p together.
Other fields (e.g. ``thinking``) go under ``params.extra`` and are sent as-is.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..types import ProviderError, Request, Response, http_error_kind
from .base import Provider

_BATCH_MAX_REQUESTS = 100_000


def _finish(reason: str | None) -> str:
    if reason in ("end_turn", "stop_sequence"):
        return "stop"
    if reason == "max_tokens":
        return "length"
    return f"other:{reason}"


class AnthropicProvider(Provider):
    supports_batch = True
    ecologits_provider = "anthropic"

    def __init__(self, cfg):
        super().__init__(cfg)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=cfg.api_key(), base_url=cfg.base_url, timeout=cfg.timeout_s, max_retries=0
        )

    def body(self, request: Request) -> dict[str, Any]:
        temperature, top_p, max_tokens, extra = self.gen_params(request)
        body: dict[str, Any] = {"model": self.cfg.model, "max_tokens": max_tokens, "messages": request.messages}
        if request.system:
            body["system"] = request.system
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        body.update(extra)
        return body

    def _parse(self, data: dict, latency: float | None, execution: str) -> Response:
        usage = data.get("usage") or {}
        tokens_in = usage.get("input_tokens")
        if tokens_in is not None:
            tokens_in += (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
        text = "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
        return Response(
            text=text,
            tokens_in=tokens_in,
            tokens_out=usage.get("output_tokens"),
            latency_s=latency,
            finish_reason=_finish(data.get("stop_reason")),
            model_version=data.get("model"),
            raw=data,
            execution=execution,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _error(self, exc: Exception) -> ProviderError:
        a = self._anthropic
        if isinstance(exc, a.APITimeoutError):
            return ProviderError("timeout", str(exc))
        if isinstance(exc, a.APIConnectionError):
            return ProviderError("connection", str(exc))
        if isinstance(exc, a.APIStatusError):
            kind = "server" if exc.status_code == 529 else http_error_kind(exc.status_code)  # 529 = overloaded
            return ProviderError(kind, str(exc), exc.status_code)
        return ProviderError("client", f"{type(exc).__name__}: {exc}")

    def generate(self, request: Request) -> Response:
        body = self.body(request)
        start = time.perf_counter()
        try:
            msg = self.client.messages.create(**body)
        except self._anthropic.AnthropicError as exc:
            raise self._error(exc) from exc
        return self._parse(msg.model_dump(mode="json"), time.perf_counter() - start, "realtime")

    def count_tokens(self, text: str) -> int | None:
        result = self.client.messages.count_tokens(model=self.cfg.model, messages=[{"role": "user", "content": text}])
        return result.input_tokens

    # Batch API (custom_id must match ^[a-zA-Z0-9_-]{1,64}$; cache keys are 64 hex chars)
    def batch_submit(self, items):
        jobs = []
        for start in range(0, len(items), _BATCH_MAX_REQUESTS):
            chunk = items[start : start + _BATCH_MAX_REQUESTS]
            batch = self.client.messages.batches.create(
                requests=[{"custom_id": cid, "params": self.body(req)} for cid, req in chunk]
            )
            jobs.append({"batch_id": batch.id, "n": len(chunk)})
        return {"jobs": jobs}

    def batch_poll(self, state):
        statuses = [self.client.messages.batches.retrieve(j["batch_id"]).processing_status for j in state["jobs"]]
        return "done" if all(s == "ended" for s in statuses) else "running"

    def batch_fetch(self, state):
        out: dict[str, Response | ProviderError] = {}
        for job in state["jobs"]:
            for entry in self.client.messages.batches.results(job["batch_id"]):
                result = entry.result
                if result.type == "succeeded":
                    out[entry.custom_id] = self._parse(result.message.model_dump(mode="json"), None, "batch")
                else:
                    detail = getattr(result, "error", None)
                    out[entry.custom_id] = ProviderError(
                        "server" if result.type == "expired" else "client", f"{result.type}: {detail}"
                    )
        return out

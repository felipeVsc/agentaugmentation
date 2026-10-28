"""Offline provider for tests and dry runs (LLM_SPEC.md §10).

Options:
- ``texts``: list of responses, used in turn (cycled);
- ``text``: one fixed response;
- default: find the last JSON object in the prompt (the required output format)
  and answer it with every value set to null.

Tests can also pass a callable: ``MockProvider(cfg, responder=lambda request: "...")``.
"""

from __future__ import annotations

import itertools
import json
import re
import threading

from ..types import Request, Response
from .base import Provider

_LAST_OBJECT = re.compile(r"\{[^{}]*\}(?![\s\S]*\{[^{}]*\})")


def null_answer(prompt: str) -> str:
    match = _LAST_OBJECT.search(prompt)
    if not match:
        return "{}"
    try:
        keys = json.loads(match.group(0)).keys()
    except json.JSONDecodeError:
        return "{}"
    return json.dumps({k: None for k in keys})


class MockProvider(Provider):
    supports_batch = True
    ecologits_provider = "openai"

    def __init__(self, cfg, responder=None):
        super().__init__(cfg)
        self.calls: list[Request] = []
        self._lock = threading.Lock()
        if responder is not None:
            self.responder = responder
        elif "texts" in cfg.options:
            cycle = itertools.cycle(cfg.options["texts"])
            self.responder = lambda request: next(cycle)
        elif "text" in cfg.options:
            self.responder = lambda request: cfg.options["text"]
        else:
            self.responder = lambda request: null_answer(request.messages[0]["content"])

    def generate(self, request: Request) -> Response:
        with self._lock:
            self.calls.append(request)
            text = self.responder(request)
        finish = "stop"
        if isinstance(text, tuple):
            text, finish = text
        prompt = (request.system or "") + "".join(m["content"] for m in request.messages)
        return Response(
            text=text,
            tokens_in=len(prompt.split()),
            tokens_out=len(text.split()),
            latency_s=0.0,
            finish_reason=finish,
            model_version=f"mock-{self.cfg.model}",
        )

    def batch_submit(self, items):
        return {"jobs": [{"items": [[cid, req.as_dict()] for cid, req in items]}]}

    def batch_poll(self, state):
        return "done"

    def batch_fetch(self, state):
        out = {}
        for job in state["jobs"]:
            for cid, req in job["items"]:
                resp = self.generate(Request(**req))
                resp.latency_s, resp.execution = None, "batch"
                out[cid] = resp
        return out

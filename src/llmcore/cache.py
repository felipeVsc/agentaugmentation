"""On-disk response cache (LLM_SPEC.md §6.4).

Key = hash of (provider, model, base_url, provider options, generation params, system,
messages, run_index). Options are included because some change the answer
(Ollama ``num_ctx``, ``think``, ``seed``).
``run_index`` is part of the key, so the 5 runs of a configuration stay separate.
A cached response keeps its original usage, latency and energy values.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .config import ModelConfig
from .types import Request, Response


def cache_key(cfg: ModelConfig, request: Request) -> str:
    payload = {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "options": cfg.options,
        "params": request.params,
        "system": request.system,
        "messages": request.messages,
        "run_index": request.run_index,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Response | None:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            resp = Response.from_dict(json.load(fh)["response"])
        resp.cached = True
        return resp

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()

    def put(self, key: str, request: Request, response: Response) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "request": request.as_dict(), "response": response.as_dict()}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
        os.replace(tmp, path)

"""Web search tools (AGENT_SPEC.md §5.2, §7).

A tool implements ``SearchTool.search(query, k, exclude_domains) -> list[SearchResult]``.
Built in: ``tavily`` and ``mock`` (offline). Plug in your own with
``search.tool: "my_module:MyTool"`` (a ``SearchTool`` subclass).

Every search goes through ``CachedSearch``: results are stored on disk keyed by
(tool, normalized query, parameters, blocked domains), so a repeated query costs no
credits and a run can be replayed exactly. The cache is shared by all models and runs;
concurrent identical queries wait for the first one, and ``max_credits`` caps the
credits one run may spend.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llmcore.config import ConfigError
from llmcore.types import ProviderError, http_error_kind


@dataclass
class SearchResult:
    url: str
    title: str
    content: str
    score: float | None = None
    query_id: str | None = None
    blocked: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResponse:
    results: list[SearchResult]
    latency_s: float | None
    cached: bool = False
    credits: float = 0.0  # credits spent by this call (0 when cached)
    raw: dict = field(default_factory=dict)


class SearchTool:
    name = "base"

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def credits_per_call(self) -> float:
        return 1.0

    def search(self, query: str, k: int, exclude_domains: list[str]) -> SearchResponse:
        raise NotImplementedError


class TavilySearch(SearchTool):
    """Tavily Search API (https://docs.tavily.com). ``depth: basic`` = 1 credit, ``advanced`` = 2."""

    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, params):
        super().__init__(params)
        env = params.get("api_key_env", "TAVILY_API_KEY")
        self.api_key = os.environ.get(env)
        if not self.api_key:
            raise ConfigError(f"Tavily API key missing: set {env} in the environment or in .env")
        self.depth = params.get("depth", "basic")
        if self.depth not in ("basic", "advanced"):
            raise ConfigError("search.depth must be basic or advanced")
        self.timeout = float(params.get("timeout_s", 30))

    def credits_per_call(self) -> float:
        return 2.0 if self.depth == "advanced" else 1.0

    def search(self, query, k, exclude_domains):
        body = {
            "query": query,
            "search_depth": self.depth,
            "max_results": k,
            "topic": self.params.get("topic", "general"),
            "include_answer": False,  # the answer must come from the model under test, not from Tavily
            "include_raw_content": False,
        }
        if exclude_domains:
            body["exclude_domains"] = [d.split("/")[0] for d in exclude_domains]
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            kind = "rate_limit" if exc.code in (429, 432, 433) else http_error_kind(exc.code)
            raise ProviderError(kind, f"Tavily HTTP {exc.code}: {detail}", exc.code) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise ProviderError("timeout", "Tavily timeout") from exc
        except urllib.error.URLError as exc:
            raise ProviderError("connection", f"Tavily unreachable: {exc.reason}") from exc
        results = [
            SearchResult(url=r.get("url", ""), title=r.get("title") or "", content=r.get("content") or "", score=r.get("score"))
            for r in data.get("results") or []
        ]
        return SearchResponse(results, time.perf_counter() - start, credits=self.credits_per_call(),
                              raw={"response_time": data.get("response_time")})


class MockSearch(SearchTool):
    """Offline search: ``params.results`` maps a substring of the query to a list of results
    (``{url, title, content, score}``); ``params.default`` is returned otherwise. Tests can pass ``responder``."""

    name = "mock"

    def __init__(self, params, responder=None):
        super().__init__(params)
        self.responder = responder
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def credits_per_call(self):
        return 0.0

    def search(self, query, k, exclude_domains):
        with self._lock:
            self.calls.append(query)
        if self.responder is not None:
            items = self.responder(query)
        else:
            items = next((v for key, v in (self.params.get("results") or {}).items() if key.lower() in query.lower()),
                         self.params.get("default", []))
        if isinstance(items, Exception):
            raise items
        results = [SearchResult(**{k2: r.get(k2) for k2 in ("url", "title", "content", "score")}) for r in items[:k]]
        return SearchResponse(results, 0.0)


_BUILTIN = {"tavily": TavilySearch, "mock": MockSearch}


def load_search_tool(params: dict[str, Any]) -> SearchTool:
    name = params.get("tool", "tavily")
    if name in _BUILTIN:
        return _BUILTIN[name](params)
    if ":" in name:
        module, _, attr = name.partition(":")
        cls = getattr(importlib.import_module(module), attr)
        if not (isinstance(cls, type) and issubclass(cls, SearchTool)):
            raise ConfigError(f"search tool {name!r} must subclass strategy_b.search.SearchTool")
        return cls(params)
    raise ConfigError(f"unknown search tool {name!r}; built-in: {sorted(_BUILTIN)}, or 'module:Class'")


# Blocklist ----------------------------------------------------------------------


def is_blocked(url: str, blocked_domains: list[str]) -> bool:
    """``example.org`` blocks the domain and its subdomains; ``github.com/user`` blocks a path prefix."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.lower()
    for entry in blocked_domains:
        entry = entry.lower().removeprefix("www.")
        domain, _, prefix = entry.partition("/")
        if host == domain or host.endswith("." + domain):
            if not prefix or path.startswith("/" + prefix):
                return True
    return False


# Cache --------------------------------------------------------------------------


def normalize_query(query: str) -> str:
    """Case, spacing and surrounding punctuation do not change what a search engine returns."""
    return re.sub(r"\s+", " ", query).strip().strip("?.!,;:").strip().casefold()


class CachedSearch:
    """Disk cache + retries + statistics + credit cap around a search tool."""

    def __init__(self, tool: SearchTool, cache_dir: Path | None, max_retries: int = 2, read_cache: bool = True,
                 sleep=time.sleep, max_credits: float | None = None):
        self.tool = tool
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.read_cache = read_cache
        self.max_credits = max_credits
        self._sleep = sleep
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._memory: dict[str, SearchResponse] = {}  # searches of this session, reused even with read_cache off
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "credits": 0.0, "budget_refused": 0}

    def _key(self, query: str, k: int, exclude: list[str]) -> str:
        params = {key: v for key, v in self.tool.params.items() if key not in ("api_key_env", "cache_dir", "timeout_s")}
        payload = {"tool": self.tool.name, "query": normalize_query(query), "k": k, "exclude": sorted(exclude),
                   "params": params}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cached(self, key: str) -> SearchResponse | None:
        hit = self._memory.get(key)
        if hit is not None:
            return SearchResponse(hit.results, hit.latency_s, cached=True, raw=hit.raw)
        if self.cache_dir is None or not self.read_cache or not self._path(key).exists():
            return None
        data = json.loads(self._path(key).read_text(encoding="utf-8"))
        return SearchResponse([SearchResult(**r) for r in data["results"]], data.get("latency_s"), cached=True,
                              raw=data.get("raw", {}))

    def search(self, query: str, k: int, exclude: list[str]) -> SearchResponse:
        key = self._key(query, k, exclude)
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        # One search per key at a time: a concurrent identical query waits and then reads the cache.
        with key_lock:
            hit = self._cached(key)
            if hit is not None:
                with self._lock:
                    self.stats["cache_hits"] += 1
                return hit
            resp = self._search(query, k, exclude)
            self._store(key, query, k, exclude, resp)
            return resp

    def _search(self, query: str, k: int, exclude: list[str]) -> SearchResponse:
        with self._lock:
            if self.max_credits is not None and self.stats["credits"] + self.tool.credits_per_call() > self.max_credits:
                self.stats["budget_refused"] += 1
                raise ProviderError("budget", f"search credit cap reached ({self.max_credits:g} credits this run)")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._lock:
                    self.stats["calls"] += 1
                resp = self.tool.search(query, k, exclude)
                break
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == self.max_retries:
                    with self._lock:
                        self.stats["failures"] += 1
                    raise
                self._sleep(min(30.0, 2.0 * 2 ** attempt))
        else:  # pragma: no cover
            raise last
        with self._lock:
            self.stats["credits"] += resp.credits
        return resp

    def _store(self, key: str, query: str, k: int, exclude: list[str], resp: SearchResponse) -> None:
        self._memory[key] = resp
        if self.cache_dir is not None:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"query": query, "k": k, "exclude": exclude, "tool": self.tool.name,
                      "results": [r.as_dict() for r in resp.results], "latency_s": resp.latency_s,
                      "raw": resp.raw, "created_at": datetime.now(timezone.utc).isoformat()}
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
            os.replace(tmp, path)

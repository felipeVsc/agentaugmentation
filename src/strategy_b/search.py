"""Web search tools (AGENT_SPEC.md §5.2, §7).

A tool implements ``SearchTool.search(query, k, exclude_domains) -> list[SearchResult]``.
Built in: ``tavily``, ``ollama`` (Ollama web search with local chunk ranking) and ``mock`` (offline). Plug in your own with
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
import math
import os
import re
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
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


class OllamaSearch(SearchTool):
    """Ollama web search API (https://docs.ollama.com/capabilities/web-search), made to behave like Tavily ``advanced``.

    The API returns up to 10 results with the page text (from a few hundred characters to a whole PDF) and has no
    domain filter or relevance score. So, locally:

    - blocked domains: up to 10 results are requested, blocked URLs are dropped and the first ``k`` are kept
      (Tavily's ``exclude_domains`` also returns ``k`` allowed results);
    - chunks: each page (its first ``max_page_chars``) is split into chunks of at most ``chunk_chars`` and the
      ``chunks_per_source`` chunks most similar to the query are kept, in page order, joined by `` [...] `` (what
      Tavily ``advanced`` returns as ``content``);
    - score: the cosine similarity of the page's best chunk; results are returned by decreasing score.

    ``ranking: embedding`` uses ``embed_model`` (fastembed, ONNX on CPU, deterministic); ``ranking: none`` keeps the
    first chunks. One API call counts as 1 credit, so ``search_max_credits`` caps the number of searches.

    Speed (ranking is CPU-bound and already uses every core):
    - ``embed_cache``: an SQLite file of chunk and query embeddings, keyed by model and text. The same pages come back
      for the same task across models and runs, so most chunks are embedded once; results are identical.
    - ``prefilter_chunks``: on a page with more chunks than this, only the chunks with the best BM25 score for the
      query are embedded (95% of the top-3 chunks unchanged on real pages, 1.6× faster). 0 = every chunk.
    """

    name = "ollama"
    URL = "https://ollama.com/api/web_search"
    MAX_RESULTS = 10
    SEPARATOR = " [...] "

    def __init__(self, params):
        super().__init__(params)
        # One key, or a list of keys (accounts): a search uses the first key that is not rate-limited and moves to
        # the next on a rate limit or a payment error, so the accounts' limits add up. Results do not depend on the key.
        envs = params.get("api_key_env", "OLLAMA_API_KEY")
        envs = [envs] if isinstance(envs, str) else list(envs)
        self.api_keys = [os.environ.get(env) for env in envs]
        missing = [env for env, key in zip(envs, self.api_keys) if not key]
        if missing:
            raise ConfigError(f"Ollama API key missing: set {', '.join(missing)} in the environment or in .env")
        self.api_key = self.api_keys[0]
        self._blocked_until = [0.0] * len(self.api_keys)  # time.time() until which each key is rate-limited
        self._key_lock = threading.Lock()
        self.key_stats = {"switches": 0, "calls_per_key": [0] * len(self.api_keys)}
        self.timeout = float(params.get("timeout_s", 30))
        self.ranking = params.get("ranking", "embedding")
        if self.ranking not in ("embedding", "none"):
            raise ConfigError("search.ranking must be embedding or none")
        self.embed_model = params.get("embed_model", "BAAI/bge-small-en-v1.5")
        self.chunk_chars = int(params.get("chunk_chars", 500))
        self.chunks_per_source = int(params.get("chunks_per_source", 3))
        self.max_page_chars = int(params.get("max_page_chars", 30_000))
        self.prefilter_chunks = int(params.get("prefilter_chunks", 0))
        self.embed_cache = params.get("embed_cache")
        self._embedder = None
        self._embed_lock = threading.Lock()
        self._db = None
        self._vectors: dict[str, Any] = {}  # embeddings of this session
        self.embed_stats = {"embedded": 0, "from_cache": 0}
        # Pacing: at least this many seconds between two API calls (all threads), to stay under Ollama's hourly and
        # session search limits instead of hitting them and waiting hours. Cached searches never reach the API.
        self.min_interval_s = float(params.get("min_interval_s", 0))
        self._next_call = 0.0
        self._pace_lock = threading.Lock()

    def _pace(self):
        if self.min_interval_s <= 0:
            return
        with self._pace_lock:
            now = time.time()
            slot = max(now, self._next_call)
            self._next_call = slot + self.min_interval_s
        if slot > now:
            time.sleep(slot - now)

    BLOCK_WITHOUT_RETRY_AFTER_S = 600.0  # a rate limit without Retry-After, or a payment error (402), parks the key

    def search(self, query, k, exclude_domains):
        body = {"query": query, "max_results": self.MAX_RESULTS if exclude_domains else min(k, self.MAX_RESULTS)}
        self._pace()   # before the timer: latency does not include the pacing wait
        start = time.perf_counter()
        while True:
            with self._key_lock:
                now = time.time()
                free = [i for i, until in enumerate(self._blocked_until) if until <= now]
                if not free:  # every account is limited: report the earliest reset, which CachedSearch waits out
                    err = ProviderError("rate_limit", "Ollama search: every API key is rate-limited", 429)
                    err.retry_after = max(1.0, min(self._blocked_until) - now)
                    raise err
                i = free[0]
                self.key_stats["calls_per_key"][i] += 1
            try:
                data = self._post(body, self.api_keys[i])
                break
            except ProviderError as exc:
                if exc.kind == "rate_limit" or (exc.status_code == 402 and len(self.api_keys) > 1):
                    wait = getattr(exc, "retry_after", None) or self.BLOCK_WITHOUT_RETRY_AFTER_S
                    with self._key_lock:
                        self._blocked_until[i] = time.time() + wait
                        self.key_stats["switches"] += 1
                    if len(self.api_keys) > 1:
                        continue
                raise
        search_s = time.perf_counter() - start
        items = data.get("results") or []
        unblocked = [r for r in items if not is_blocked(r.get("url", ""), exclude_domains)]
        allowed = unblocked[:k]
        rank_start = time.perf_counter()
        results = self._results(query, allowed)
        results.sort(key=lambda r: -(r.score or 0.0))
        rank_s = time.perf_counter() - rank_start
        return SearchResponse(results, search_s + rank_s, credits=self.credits_per_call(),
                              raw={"search_s": search_s, "rank_s": rank_s, "returned": len(items),
                                   "blocked_dropped": len(items) - len(unblocked),
                                   "page_chars": [len(r.get("content") or "") for r in allowed]})

    def _post(self, body: dict, api_key: str) -> dict:
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            kind = "rate_limit" if exc.code == 429 else http_error_kind(exc.code)
            err = ProviderError(kind, f"Ollama search HTTP {exc.code}: {detail}", exc.code)
            try:  # the hourly search limit says when it resets
                err.retry_after = float(exc.headers.get("Retry-After") or 0) or None
            except (TypeError, ValueError):
                err.retry_after = None
            raise err from exc
        except (socket.timeout, TimeoutError) as exc:
            raise ProviderError("timeout", "Ollama search timeout") from exc
        except urllib.error.URLError as exc:
            raise ProviderError("connection", f"Ollama search unreachable: {exc.reason}") from exc

    def _results(self, query: str, items: list[dict]) -> list[SearchResult]:
        """One result per page: its best chunks (in page order) and the similarity of the best one.
        Pages are embedded one at a time: one call for all pages was slower on CPU (3.9 s against 2.7 s for 5 pages)."""
        pages = [self.chunk(item.get("content") or "") for item in items]
        out = []
        for item, chunks in zip(items, pages):
            url, title = item.get("url", ""), item.get("title") or ""
            if not chunks:
                out.append(SearchResult(url=url, title=title, content="", score=0.0))
                continue
            if self.ranking == "none":
                keep, score = list(range(min(self.chunks_per_source, len(chunks)))), None
            else:
                ids = list(range(len(chunks)))
                if self.prefilter_chunks and len(chunks) > self.prefilter_chunks:
                    lexical = bm25(query, chunks)
                    ids = sorted(sorted(ids, key=lambda i: -lexical[i])[:self.prefilter_chunks])
                sims = self._similarities(query, [chunks[i] for i in ids])
                page = {i: float(v) for i, v in zip(ids, sims)}
                best = sorted(ids, key=lambda i: -page[i])[:self.chunks_per_source]
                keep, score = sorted(best), float(page[best[0]])
            out.append(SearchResult(url=url, title=title, content=self.SEPARATOR.join(chunks[i] for i in keep), score=score))
        return out

    def chunk(self, text: str) -> list[str]:
        """Whitespace-normalized chunks of at most ``chunk_chars``, cut between words."""
        words = " ".join(text[:self.max_page_chars].split()).split(" ")
        chunks, cur = [], ""
        for w in words:
            if not w:
                continue
            w = w[:self.chunk_chars]
            if cur and len(cur) + 1 + len(w) > self.chunk_chars:
                chunks.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            chunks.append(cur)
        return chunks

    def _similarities(self, query: str, chunks: list[str]):
        import numpy as np

        with self._embed_lock:  # one ONNX session (it already uses every core) and one cache connection
            vecs = np.array(self._vectors_for(chunks, "passage"))
            qv = np.array(self._vectors_for([query], "query"))[0]
        return vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-9)

    def _vectors_for(self, texts: list[str], kind: str) -> list:
        """Embeddings of ``texts`` (``kind``: passage or query), from the session, the disk cache, or the model."""
        import numpy as np

        keys = [hashlib.sha256(f"{self.embed_model}|{kind}|{t}".encode()).hexdigest() for t in texts]
        missing = [k for k in dict.fromkeys(keys) if k not in self._vectors]
        if missing and self.embed_cache:
            db = self._cache_db()
            for i in range(0, len(missing), 500):
                part = missing[i:i + 500]
                rows = db.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part).fetchall()
                for key, blob in rows:
                    self._vectors[key] = np.frombuffer(blob, dtype=np.float32)
            self.embed_stats["from_cache"] += sum(k in self._vectors for k in missing)
        todo = {k: t for k, t in zip(keys, texts) if k not in self._vectors}
        if todo:
            if self._embedder is None:
                from fastembed import TextEmbedding

                self._embedder = TextEmbedding(self.embed_model)
            embed = self._embedder.query_embed if kind == "query" else self._embedder.embed
            new = [np.asarray(v, dtype=np.float32) for v in embed(list(todo.values()))]
            self._vectors.update(zip(todo, new))
            self.embed_stats["embedded"] += len(new)
            if self.embed_cache:
                db = self._cache_db()
                db.executemany("INSERT OR IGNORE INTO emb (key, vec) VALUES (?, ?)", [(k, v.tobytes()) for k, v in zip(todo, new)])
                db.commit()
        return [self._vectors[k] for k in keys]

    def _cache_db(self):
        if self._db is None:
            import sqlite3

            Path(self.embed_cache).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.embed_cache, check_same_thread=False, timeout=60)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        return self._db


def bm25(query: str, chunks: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of each chunk for the query, with the page's chunks as the corpus."""
    tok = lambda text: re.findall(r"[a-z0-9]+", text.lower())  # noqa: E731
    docs = [tok(c) for c in chunks]
    avg = sum(map(len, docs)) / max(len(docs), 1) or 1.0
    df = Counter(w for d in docs for w in set(d))
    terms = set(tok(query))
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for w in terms & tf.keys():
            idf = math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5))
            score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(score)
    return scores


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


_BUILTIN = {"tavily": TavilySearch, "ollama": OllamaSearch, "mock": MockSearch}


def load_search_tool(params: dict[str, Any]) -> SearchTool:
    """Search tool from the config: built-in (ollama, tavily, mock) or 'module:Class'."""
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
                 sleep=time.sleep, max_credits: float | None = None, max_rate_limit_wait_s: float = 0.0):
        self.tool = tool
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.read_cache = read_cache
        self.max_credits = max_credits
        # A rate limit with Retry-After (Ollama's hourly search limit) is waited out, up to this many seconds in total
        # per search, instead of failing the run; 0 = fail as any other error.
        self.max_rate_limit_wait_s = max_rate_limit_wait_s
        self._sleep = sleep
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._memory: dict[str, SearchResponse] = {}  # searches of this session, reused even with read_cache off
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "credits": 0.0, "budget_refused": 0,
                      "rate_limit_waits": 0, "rate_limit_wait_s": 0.0}

    def _key(self, query: str, k: int, exclude: list[str]) -> str:
        params = {key: v for key, v in self.tool.params.items()
                  if key not in ("api_key_env", "cache_dir", "timeout_s", "embed_cache", "min_interval_s")}  # do not change results
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
        waited = 0.0
        attempt = 0
        while True:
            try:
                with self._lock:
                    self.stats["calls"] += 1
                resp = self.tool.search(query, k, exclude)
                break
            except ProviderError as exc:
                retry_after = getattr(exc, "retry_after", None)
                if exc.kind == "rate_limit" and retry_after and waited + retry_after <= self.max_rate_limit_wait_s:
                    with self._lock:
                        self.stats["rate_limit_waits"] += 1
                        self.stats["rate_limit_wait_s"] += retry_after
                    self._sleep(retry_after + 1)
                    waited += retry_after
                    continue  # does not count as a retry
                if not exc.retryable or attempt >= self.max_retries:
                    with self._lock:
                        self.stats["failures"] += 1
                    raise
                self._sleep(min(30.0, 2.0 * 2 ** attempt))
                attempt += 1
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

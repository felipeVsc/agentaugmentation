"""Cache, rate limit, retries and energy around one provider (LLM_SPEC.md §6.3–6.5)."""

from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timezone

from .cache import DiskCache, cache_key
from .config import ModelConfig
from .energy import EnergyMeter, make_meter
from .providers import Provider, load_provider
from .types import CallFailed, ProviderError, Request, Response


class RateLimiter:
    """At most ``rpm`` call starts per minute (evenly spaced)."""

    def __init__(self, rpm: int | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


class Caller:
    def __init__(
        self,
        cfg: ModelConfig,
        cache: DiskCache | None,
        provider: Provider | None = None,
        meter: EnergyMeter | None = None,
        read_cache: bool = True,
        sleep=time.sleep,
    ):
        self.cfg = cfg
        self.provider = provider or load_provider(cfg)
        self.meter = meter or make_meter(cfg, self.provider.ecologits_provider)
        self.cache = cache
        self.read_cache = read_cache
        self.limiter = RateLimiter(cfg.rpm)
        self._sleep = sleep
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "failures": 0}
        self._stats_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def key(self, request: Request) -> str:
        return cache_key(self.cfg, request)

    def cached(self, request: Request) -> Response | None:
        if self.cache is None or not self.read_cache:
            return None
        return self.cache.get(self.key(request))

    def _count(self, name: str) -> None:
        with self._stats_lock:
            self.stats[name] += 1

    def call(self, request: Request, cache_only: bool = False) -> Response | None:
        """Return the response, from cache if possible. ``cache_only`` returns None on a miss.

        A request identical to one in flight waits for it and then reads the cache, so concurrent
        duplicates (e.g. the shared plan of Strategy B) cost one call.
        """
        if cache_only:
            hit = self.cached(request)
            if hit is not None:
                self._count("cache_hits")
            return hit
        key = self.key(request)
        with self._stats_lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            hit = self.cached(request)
            if hit is not None:
                self._count("cache_hits")
                return hit
            response = self._call_with_retries(request)
            if response.created_at is None:
                response.created_at = datetime.now(timezone.utc).isoformat()
            if self.cache is not None:
                self.cache.put(key, request, response)
            return response

    def _call_with_retries(self, request: Request) -> Response:
        attempts = self.cfg.max_retries + 1
        for attempt in range(1, attempts + 1):
            self.limiter.wait()
            self._count("calls")
            try:
                return self.meter.measure(lambda: self.provider.generate(request))
            except ProviderError as exc:
                if not exc.retryable or attempt == attempts:
                    self._count("failures")
                    status = "timeout" if exc.kind == "timeout" else "api_error"
                    raise CallFailed(status, f"{exc.kind}: {exc}", attempt) from exc
                self._count("retries")
                # Exponential backoff with jitter; rate limits wait longer.
                base = 4.0 if exc.kind == "rate_limit" else 1.0
                self._sleep(min(120.0, base * 2 ** (attempt - 1)) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self.meter.close()
        self.provider.close()

    def store_batch_result(self, key: str, request: Request, response: Response) -> None:
        response.energy = self.meter.estimate(response)
        if response.created_at is None:
            response.created_at = datetime.now(timezone.utc).isoformat()
        self.cache.put(key, request, response)

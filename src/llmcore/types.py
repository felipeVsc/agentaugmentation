"""Request/response types and errors shared by every provider."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Request:
    """One model call.

    ``messages`` alternates ``user``/``assistant`` turns (a repair call appends
    the bad answer and a new user turn). ``params`` holds ``temperature``,
    ``top_p``, ``max_tokens`` and ``extra`` (provider-specific fields passed through).
    """

    system: str | None
    messages: list[dict[str, str]]
    params: dict[str, Any]
    run_index: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Response:
    text: str
    tokens_in: int | None
    tokens_out: int | None
    latency_s: float | None  # None for batch calls: the provider's queue time is not model time
    finish_reason: str  # "stop" | "length" | "error" | "other:<provider value>"
    model_version: str | None = None
    raw: dict = field(default_factory=dict)
    execution: str = "realtime"  # "realtime" | "batch"
    # Filled by the caller: {"energy_kwh", "co2e_kg", ...} or None when not measured.
    energy: dict | None = None
    created_at: str | None = None  # ISO 8601, when the provider answered
    cached: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("cached")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Response":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


class ProviderError(Exception):
    """Raised by a provider. ``kind`` drives retries and the final run status."""

    RETRYABLE = {"rate_limit", "timeout", "server", "connection"}

    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.kind in self.RETRYABLE


class CallFailed(Exception):
    """A call that still failed after the last retry."""

    def __init__(self, status: str, message: str, attempts: int):
        super().__init__(message)
        self.status = status  # "timeout" | "api_error"
        self.attempts = attempts


def http_error_kind(status_code: int | None) -> str:
    if status_code is None:
        return "connection"
    if status_code == 429:
        return "rate_limit"
    if status_code in (408, 504):
        return "timeout"
    if status_code in (409, 425) or status_code >= 500:
        return "server"
    if status_code in (401, 403):
        return "auth"
    return "client"

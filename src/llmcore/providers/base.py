"""Provider interface (LLM_SPEC.md §6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import ModelConfig
from ..types import ProviderError, Request, Response


class Provider(ABC):
    """One model behind one API.

    ``generate`` makes a single call and either returns a ``Response`` or raises
    ``ProviderError`` (the caller retries the retryable kinds). Providers must
    not retry on their own: SDK retries are disabled so that every attempt is
    counted by the caller.

    Batch support (optional) is three steps, all keyed by ``custom_id`` (the cache key):
    ``batch_submit`` returns a JSON-serializable state, ``batch_poll`` returns
    ``"running"``, ``"done"`` or ``"failed"``, and ``batch_fetch`` returns
    ``{custom_id: Response | ProviderError}`` for the requests that finished.
    """

    supports_batch: bool = False
    # Name used by EcoLogits for this provider, when it differs from the registry name.
    ecologits_provider: str | None = None

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @abstractmethod
    def generate(self, request: Request) -> Response: ...

    def describe(self) -> dict[str, Any]:
        """Model details for the manifest (digest, quantization, ...); empty when the provider has none."""
        return {}

    def close(self) -> None:
        """Release resources at the end of a run."""

    def count_tokens(self, text: str) -> int | None:
        """Provider-side token count, if the provider offers one (used by ``tokenizer: provider``)."""
        return None

    # Batch API (optional)
    def batch_submit(self, items: list[tuple[str, Request]]) -> dict[str, Any]:
        raise NotImplementedError(f"provider {self.cfg.provider!r} has no batch API")

    def batch_poll(self, state: dict[str, Any]) -> str:
        raise NotImplementedError

    def batch_fetch(self, state: dict[str, Any]) -> dict[str, Response | ProviderError]:
        raise NotImplementedError

    # Helpers for subclasses
    def gen_params(self, request: Request) -> tuple[float | None, float | None, int, dict]:
        p = request.params
        return p.get("temperature"), p.get("top_p"), int(p.get("max_tokens", 1024)), dict(p.get("extra") or {})

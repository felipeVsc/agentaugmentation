"""Token counters for PFE (LLM_SPEC.md §5.6) and for usage the provider doesn't report.

Specs:
- ``cl100k_base``, ``o200k_base`` or ``tiktoken:<encoding>``: tiktoken encoding;
- ``tiktoken-model:<model>``: tiktoken encoding for an OpenAI model name;
- ``hf:<repo>``: Hugging Face tokenizer (needs the ``tokenizers`` package);
- ``provider``: the provider's own count endpoint (Anthropic, Gemini), one API call per text;
- ``simple``: words and punctuation, offline and dependency-free (tests, rough counts).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

from .config import ConfigError

_SIMPLE = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    def __init__(self, name: str, count: Callable[[str], int]):
        self.name = name
        self._count = count

    def count(self, text: str) -> int:
        return self._count(text) if text else 0


@lru_cache(maxsize=None)
def get_tokenizer(spec: str) -> Tokenizer:
    kind, _, arg = spec.partition(":")
    if spec == "simple":
        return Tokenizer(spec, lambda text: len(_SIMPLE.findall(text)))
    if kind in ("tiktoken", "tiktoken-model") or (not arg and spec.endswith("_base")):
        try:
            import tiktoken
        except ImportError as exc:
            raise ConfigError(f"tokenizer {spec!r} needs tiktoken: pip install tiktoken") from exc
        enc = tiktoken.encoding_for_model(arg) if kind == "tiktoken-model" else tiktoken.get_encoding(arg or spec)
        return Tokenizer(spec, lambda text: len(enc.encode(text, disallowed_special=())))
    if kind == "hf":
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:
            raise ConfigError(f"tokenizer {spec!r} needs the tokenizers package: pip install tokenizers") from exc
        tok = HFTokenizer.from_pretrained(arg)
        return Tokenizer(spec, lambda text: len(tok.encode(text, add_special_tokens=False).ids))
    raise ConfigError(f"unknown tokenizer {spec!r}")


def provider_tokenizer(provider) -> Tokenizer:
    """Counts with the provider's API (``tokenizer: provider``); results are memoized."""
    memo: dict[str, int] = {}

    def count(text: str) -> int:
        if text not in memo:
            n = provider.count_tokens(text)
            if n is None:
                raise ConfigError(f"provider {provider.cfg.provider!r} cannot count tokens; choose another tokenizer")
            memo[text] = n
        return memo[text]

    return Tokenizer(f"provider:{provider.cfg.provider}/{provider.cfg.model}", count)


def reasoning_tokens(raw: dict, tokenizer: Tokenizer) -> int | None:
    """Reasoning tokens of one call, a part of its tokens_out (METRICS.md §4.1); None when unknown.

    Ollama returns the thinking text but no count, so it is counted with the tokenizer
    (0 when the model did not think). OpenAI, Gemini and Anthropic report it in their usage.
    """
    if "thinking" in raw and "eval_count" in raw:  # Ollama
        return tokenizer.count(raw["thinking"]) if raw["thinking"] else 0
    usage = raw.get("usage") or raw.get("usage_metadata") or raw.get("usageMetadata") or {}
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    for value in (details.get("reasoning_tokens"), usage.get("thoughts_token_count"), usage.get("thoughtsTokenCount")):
        if value is not None:
            return int(value)
    return None

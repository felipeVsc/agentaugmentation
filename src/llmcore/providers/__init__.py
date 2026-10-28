"""Provider registry.

Built-in names map to lazily imported classes, so an SDK is only needed when a
model in the config uses it. Plug in your own provider either way:

- in the config: ``provider: "my_package.my_module:MyProvider"``;
- in code: ``register_provider("myname", MyProvider)`` before loading the config.

A provider subclasses ``Provider`` and implements ``generate``; batch support is optional.
"""

from __future__ import annotations

import importlib

from ..config import ConfigError, ModelConfig
from .base import Provider

_BUILTIN = {
    "openai": "llmcore.providers.openai_provider:OpenAIProvider",
    "anthropic": "llmcore.providers.anthropic_provider:AnthropicProvider",
    "google": "llmcore.providers.google_provider:GoogleProvider",
    "ollama": "llmcore.providers.ollama_provider:OllamaProvider",
    "mock": "llmcore.providers.mock:MockProvider",
}
_REGISTERED: dict[str, type[Provider]] = {}


def register_provider(name: str, cls: type[Provider]) -> None:
    if not (isinstance(cls, type) and issubclass(cls, Provider)):
        raise TypeError("provider class must subclass llmcore.providers.Provider")
    _REGISTERED[name] = cls


def _import(path: str) -> type[Provider]:
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ConfigError(f"provider {path!r}: expected 'module:Class'")
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load provider {path!r}: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, Provider)):
        raise ConfigError(f"provider {path!r} must subclass llmcore.providers.Provider")
    return cls


def provider_class(name: str) -> type[Provider]:
    if name in _REGISTERED:
        return _REGISTERED[name]
    if name in _BUILTIN:
        return _import(_BUILTIN[name])
    if ":" in name:
        return _import(name)
    raise ConfigError(f"unknown provider {name!r}; built-in: {sorted(_BUILTIN)}, or use 'module:Class'")


def load_provider(cfg: ModelConfig) -> Provider:
    return provider_class(cfg.provider)(cfg)


__all__ = ["Provider", "register_provider", "provider_class", "load_provider"]

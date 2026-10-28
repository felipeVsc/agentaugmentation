"""Model configuration (one entry of ``models:`` in a strategy config).

API keys are never stored in configs: ``api_key_env`` names the environment
variable to read (optionally loaded from ``.env`` by the CLI).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


class ConfigError(ValueError):
    pass


DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
}

GENERATION_KEYS = {"temperature", "top_p", "max_tokens", "extra"}


@dataclass
class EnergyConfig:
    # "ecologits" (API models, estimated), "codecarbon" (local models, measured) or "none".
    method: str = "none"
    # EcoLogits: provider and model to look up. Defaults come from the model entry;
    # set them to the closest registered model when the exact one is missing.
    ecologits_provider: str | None = None
    ecologits_model: str | None = None
    electricity_mix_zone: str | None = None  # ISO 3166-1 alpha-3, None = provider default
    # CodeCarbon
    country_iso_code: str = "FRA"


@dataclass
class Pricing:
    """USD per million tokens: ``cost_usd`` of every run (METRICS.md §4.5), ``estimate``, and the manifest."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    batch_discount: float = 0.5  # batch price = realtime price × (1 - discount)
    as_of: str | None = None  # date of the price list, recorded in the manifest

    def cost(self, tokens_in: int, tokens_out: int, execution: str = "realtime") -> float:
        usd = tokens_in / 1e6 * self.input_per_mtok + tokens_out / 1e6 * self.output_per_mtok
        return usd * (1 - self.batch_discount) if execution == "batch" else usd


@dataclass
class ModelConfig:
    id: str  # written to results.csv `model`
    provider: str  # registry name ("openai", "anthropic", "google", "mock") or "module:Class"
    model: str | None = None  # name sent to the API; defaults to id
    params: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.7, "max_tokens": 1024})
    max_concurrency: int = 4
    rpm: int | None = None  # requests per minute, None = no limit
    timeout_s: float = 60.0
    max_retries: int = 3
    api_key_env: str | None = None
    base_url: str | None = None  # OpenAI-compatible servers: vLLM, Ollama, OpenRouter, ...
    options: dict[str, Any] = field(default_factory=dict)  # provider-specific settings
    tokenizer: str | None = None  # for PFE and missing usage counts; None = config default
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    pricing: Pricing | None = None
    enabled: bool = True

    def __post_init__(self):
        if self.model is None:
            self.model = self.id

    @property
    def key_env(self) -> str | None:
        return self.api_key_env or DEFAULT_KEY_ENV.get(self.provider)

    def api_key(self, required: bool = True) -> str | None:
        name = self.key_env
        value = os.environ.get(name) if name else None
        if required and not value:
            raise ConfigError(
                f"model {self.id!r}: API key missing, set {name} in the environment or in .env"
                if name
                else f"model {self.id!r}: no api_key_env set"
            )
        return value

    def validate(self) -> None:
        unknown = set(self.params) - GENERATION_KEYS
        if unknown:
            raise ConfigError(f"model {self.id!r}: unknown params {sorted(unknown)}; put provider fields under params.extra")
        temperature = self.params.get("temperature")
        if temperature is None or temperature <= 0:
            raise ConfigError(
                f"model {self.id!r}: temperature must be > 0 (LLM_SPEC §6.2), otherwise the 5 runs are identical"
            )
        if self.max_concurrency < 1:
            raise ConfigError(f"model {self.id!r}: max_concurrency must be >= 1")
        if self.energy.method not in ("none", "ecologits", "codecarbon"):
            raise ConfigError(f"model {self.id!r}: energy.method must be none, ecologits or codecarbon")
        if self.energy.method == "codecarbon" and self.max_concurrency != 1:
            raise ConfigError(
                f"model {self.id!r}: codecarbon measures the whole machine, so max_concurrency must be 1"
            )

    def public_dict(self) -> dict:
        """For manifests: everything in the config (which never holds secrets)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, defaults: dict | None = None) -> "ModelConfig":
        defaults = defaults or {}

        def mapping(value):
            return {"method": value} if isinstance(value, str) else dict(value or {})

        # Nested mappings merge key by key, so a model can override one option of model_defaults.
        nested = {
            name: {**mapping(defaults.get(name)), **mapping(d.get(name))} for name in ("params", "options", "energy")
        }
        d = {**defaults, **d}
        if "id" not in d or "provider" not in d:
            raise ConfigError(f"model entry needs id and provider: {d}")
        if d["provider"] == "ollama":
            # One local server: calls queue on it anyway, and CodeCarbon needs them one at a time.
            d.setdefault("max_concurrency", 1)
            d.setdefault("timeout_s", 600.0)  # first call loads the model; CPU inference is slow
        d.pop("params", None)
        d.pop("energy", None)
        d["options"] = nested["options"]
        params, energy = nested["params"], nested["energy"]
        pricing = d.pop("pricing", None)
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise ConfigError(f"model {d['id']!r}: unknown fields {sorted(extra)}")
        cfg = cls(
            **d,
            params=params,
            energy=EnergyConfig(**(energy or {})),
            pricing=Pricing(**pricing) if pricing else None,
        )
        cfg.validate()
        return cfg

"""Strategy B configuration (AGENT_SPEC.md §9). Models are declared exactly as in Strategy A."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llmcore.config import ConfigError, ModelConfig, endpoint_output_dir, use_ollama

_VERSION = re.compile(r"\.(v\d+[\w-]*)\.[^.]+$")
SYNTHESIS_MODES = ("evidence_only", "evidence_and_memory")


@dataclass
class VariantSpec:
    """One agent configuration, written to results.csv ``variant``."""

    id: str
    plan: str  # prompt file of the planning call
    synthesize: str  # prompt file of the synthesis call
    # evidence_only: every value must cite retrieved evidence (AGENT_SPEC §5.3), isolating the effect of search.
    # evidence_and_memory: the model may also answer from its own knowledge when the evidence is silent.
    synthesis: str = "evidence_only"
    with_context: bool = False  # also give the graph context of the task (published with-context condition)
    version: str | None = None

    def __post_init__(self):
        if self.synthesis not in SYNTHESIS_MODES:
            raise ConfigError(f"variant {self.id!r}: synthesis must be one of {SYNTHESIS_MODES}")
        if self.version is None:
            versions = {m.group(1) for f in (self.plan, self.synthesize) if (m := _VERSION.search(f))}
            if len(versions) != 1:
                raise ConfigError(f"variant {self.id!r}: set version, or name both prompt files like name.v1.txt")
            self.version = versions.pop()


@dataclass
class StrategyBConfig:
    tasks: str
    variants: list[VariantSpec]
    models: list[ModelConfig]
    output_dir: str
    search: dict[str, Any] = field(default_factory=lambda: {"tool": "tavily", "depth": "basic"})
    runs_per_task: int = 5
    max_subqueries: int = 4
    results_per_query: int = 5
    max_evidence_tokens: int = 3000
    max_chars_per_result: int = 1500
    # true: plan and search once per (task, model, variant), then synthesize runs_per_task times on that evidence
    # (AGENT_SPEC §6). Separates model variance from search variance and divides search credits by runs_per_task.
    reuse_retrieval: bool = False
    blocked_domains: list[str] = field(default_factory=list)
    pfe_tokenizer: str = "cl100k_base"
    cache_dir: str | None = None  # LLM cache, default <output_dir>/cache
    search_cache_dir: str | None = None  # default <output_dir>/search_cache; share it across models to save credits
    max_search_retries: int = 2
    # 'cloud', 'local' or a URL: where the Ollama models run (llmcore.config.use_ollama); None = their own base_url.
    # The Ollama web search always goes to ollama.com and needs an Ollama key, wherever the models run.
    ollama: str | None = None
    max_rate_limit_wait_s: float = 0.0  # wait out a search rate limit (Retry-After) up to this long instead of failing
    search_max_credits: float | None = None  # cap on search credits per `strategy-b run`; over it, searches fail
    max_parallel_queries: int = 4
    property_notes: str | None = None  # YAML {property name: short description}, for {property_notes}
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    base_dir: Path = field(default_factory=Path.cwd, repr=False)

    def notes(self) -> dict[str, str]:
        """The property descriptions of ``inputs.property_notes`` (empty when unset)."""
        if not self.property_notes:
            return {}
        with open(self.path(self.property_notes), encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise ConfigError(f"{self.property_notes}: expected a mapping property name -> description")
        return data

    def path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        path = Path(p)
        return path if path.is_absolute() else self.base_dir / path

    @property
    def cache_path(self) -> Path:
        return self.path(self.cache_dir) if self.cache_dir else self.path(self.output_dir) / "cache"

    @property
    def search_cache_path(self) -> Path:
        return self.path(self.search_cache_dir) if self.search_cache_dir else self.path(self.output_dir) / "search_cache"

    def select_models(self, ids: list[str] | None) -> list[ModelConfig]:
        if ids:
            unknown = set(ids) - {m.id for m in self.models}
            if unknown:
                raise ConfigError(f"unknown models {sorted(unknown)}")
            return [m for m in self.models if m.id in ids]
        return [m for m in self.models if m.enabled]

    def select_variants(self, ids: list[str] | None) -> list[VariantSpec]:
        if not ids:
            return self.variants
        unknown = set(ids) - {v.id for v in self.variants}
        if unknown:
            raise ConfigError(f"unknown variants {sorted(unknown)}")
        return [v for v in self.variants if v.id in ids]


def load_config(path: str | Path, ollama: str | None = None) -> StrategyBConfig:
    """Read and validate a Strategy B YAML config. Refuses a ground_truth.csv as input.

    ``ollama`` ('cloud', 'local' or a URL) overrides the config's ``ollama`` key; see ``llmcore.config.use_ollama``.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    inputs = raw.get("inputs") or {}
    if "tasks" not in inputs:
        raise ConfigError("inputs.tasks is required")
    if Path(inputs["tasks"]).name == "ground_truth.csv":
        raise ConfigError("Strategy B must never read ground_truth.csv")
    defaults = raw.get("model_defaults") or {}
    models = [ModelConfig.from_dict(m, defaults) for m in raw.get("models") or []]
    if not models:
        raise ConfigError("no models configured")
    variants = [VariantSpec(**v) for v in raw.get("variants") or []]
    if not variants:
        raise ConfigError("no variants configured")
    for name, items in (("model", [m.id for m in models]), ("variant", [v.id for v in variants])):
        if len(set(items)) != len(items):
            raise ConfigError(f"{name} ids must be unique")
    known = set(StrategyBConfig.__dataclass_fields__) - {"tasks", "variants", "models", "raw", "base_dir", "property_notes"}
    extra = set(raw) - known - {"inputs", "model_defaults", "models", "variants"}
    if extra:
        raise ConfigError(f"unknown top-level keys {sorted(extra)}")
    cfg = StrategyBConfig(
        tasks=inputs["tasks"],
        property_notes=inputs.get("property_notes"),
        variants=variants,
        models=models,
        raw=raw,
        **{k: raw[k] for k in known if k in raw},
    )
    if ollama:
        cfg.ollama = ollama
    endpoint = use_ollama(cfg.models, cfg.ollama)
    if endpoint:
        # Searches do not depend on where the model runs: keep the search cache where it was.
        cfg.search_cache_dir = cfg.search_cache_dir or str(Path(cfg.output_dir) / "search_cache")
        cfg.output_dir = endpoint_output_dir(cfg.output_dir, endpoint)
    if cfg.runs_per_task < 1 or cfg.max_subqueries < 1 or cfg.results_per_query < 1:
        raise ConfigError("runs_per_task, max_subqueries and results_per_query must be >= 1")
    return cfg

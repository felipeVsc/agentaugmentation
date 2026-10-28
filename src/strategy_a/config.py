"""Strategy A configuration (LLM_SPEC.md §9)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llmcore.config import ConfigError, ModelConfig

_VERSION = re.compile(r"\.(v\d+[\w-]*)\.[^.]+$")


@dataclass
class TemplateSpec:
    id: str
    file: str
    # Few-shot: number(s) of examples. A list expands into one variant per value
    # (``base_few_shot_1``, ``base_few_shot_3``, ...). None/0 = no examples.
    n_examples: list[int] = field(default_factory=list)
    context_format: str = "list"  # list | json
    version: str | None = None  # default: taken from the file name (name.v1.txt -> v1)
    expected_output_tokens: int = 80  # only for `estimate`
    output: str = "json"  # json | csv (the published prompts ask for CSV `property,value`)
    with_context: bool = False  # {published_input}: the with-context input of the published run

    def __post_init__(self):
        if self.output not in ("json", "csv"):
            raise ConfigError(f"template {self.id!r}: output must be json or csv")
        if isinstance(self.n_examples, int):
            self.n_examples = [self.n_examples] if self.n_examples else []
        self.n_examples = [int(n) for n in self.n_examples or [] if int(n) > 0]
        if self.context_format not in ("list", "json"):
            raise ConfigError(f"template {self.id!r}: context_format must be list or json")
        if self.version is None:
            m = _VERSION.search(self.file)
            if not m:
                raise ConfigError(f"template {self.id!r}: set version, or name the file like name.v1.txt")
            self.version = m.group(1)


@dataclass
class Variant:
    """One distinct configuration, written to results.csv `variant`."""

    id: str
    template: TemplateSpec
    n_examples: int


@dataclass
class StrategyAConfig:
    tasks: str
    templates: list[TemplateSpec]
    models: list[ModelConfig]
    output_dir: str
    examples: str | None = None
    runs_per_task: int = 5
    pfe_tokenizer: str = "cl100k_base"
    cache_dir: str | None = None  # default: <output_dir>/cache
    repair: bool = True  # one repair call on parse failures (LLM_SPEC §7.2)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    base_dir: Path = field(default_factory=Path.cwd, repr=False)

    def path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        path = Path(p)
        return path if path.is_absolute() else self.base_dir / path

    @property
    def cache_path(self) -> Path:
        return self.path(self.cache_dir) if self.cache_dir else self.path(self.output_dir) / "cache"

    def variants(self) -> list[Variant]:
        out = []
        for t in self.templates:
            if t.n_examples:
                out.extend(Variant(f"{t.id}_{n}", t, n) for n in t.n_examples)
            else:
                out.append(Variant(t.id, t, 0))
        return out

    def select_models(self, ids: list[str] | None) -> list[ModelConfig]:
        if ids:
            unknown = set(ids) - {m.id for m in self.models}
            if unknown:
                raise ConfigError(f"unknown models {sorted(unknown)}")
            return [m for m in self.models if m.id in ids]
        return [m for m in self.models if m.enabled]

    def select_variants(self, ids: list[str] | None) -> list[Variant]:
        variants = self.variants()
        if not ids:
            return variants
        chosen = [v for v in variants if v.id in ids or v.template.id in ids]
        unknown = set(ids) - {v.id for v in chosen} - {v.template.id for v in chosen}
        if unknown:
            raise ConfigError(f"unknown templates/variants {sorted(unknown)}")
        return chosen


def load_config(path: str | Path) -> StrategyAConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    inputs = raw.get("inputs") or {}
    if "tasks" not in inputs:
        raise ConfigError("inputs.tasks is required")
    if Path(inputs["tasks"]).name == "ground_truth.csv":
        raise ConfigError("Strategy A must never read ground_truth.csv")
    model_defaults = raw.get("model_defaults") or {}
    models = [ModelConfig.from_dict(m, model_defaults) for m in raw.get("models") or []]
    if not models:
        raise ConfigError("no models configured")
    ids = [m.id for m in models]
    if len(set(ids)) != len(ids):
        raise ConfigError("model ids must be unique")
    templates = [TemplateSpec(**t) for t in raw.get("templates") or []]
    if not templates:
        raise ConfigError("no templates configured")
    cfg = StrategyAConfig(
        tasks=inputs["tasks"],
        examples=inputs.get("examples"),
        templates=templates,
        models=models,
        output_dir=raw.get("output_dir") or "results/strategy_a",
        runs_per_task=int(raw.get("runs_per_task", 5)),
        pfe_tokenizer=raw.get("pfe_tokenizer", raw.get("tokenizer_for_pfe", "cl100k_base")),
        cache_dir=raw.get("cache_dir"),
        repair=bool(raw.get("repair", True)),
        raw=raw,
        base_dir=Path(raw.get("base_dir") or Path.cwd()),
    )
    variant_ids = [v.id for v in cfg.variants()]
    if len(set(variant_ids)) != len(variant_ids):
        raise ConfigError(f"variant ids must be unique: {variant_ids}")
    if any(t.n_examples for t in templates) and not cfg.examples:
        raise ConfigError("few-shot templates need inputs.examples")
    if cfg.runs_per_task < 1:
        raise ConfigError("runs_per_task must be >= 1")
    return cfg

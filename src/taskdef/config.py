"""Configuration of a task-set build (TASKDEFINITION_SPEC.md §8).

Secrets never live in the YAML file: the Neo4j user and password are read
from environment variables (optionally loaded from a ``.env`` file by the CLI).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    database: str | None = None
    # Environment variables that override / provide the connection settings.
    uri_env: str = "NEO4J_URI"
    database_env: str = "NEO4J_DATABASE"
    user_env: str = "NEO4J_USER"
    password_env: str = "NEO4J_PASSWORD"
    # "elementId" for Neo4j 5, "id" for Neo4j 4.
    id_function: str = "elementId"
    # Optional export filters. None = export the whole graph, which keeps
    # PageRank computed on the real graph rather than on a subgraph.
    labels: list[str] | None = None
    relationship_types: list[str] | None = None
    fetch_size: int = 10000

    def connection(self) -> tuple[str, str | None, str, str]:
        """Return ``(uri, database, user, password)``, reading secrets from the environment."""
        uri = os.environ.get(self.uri_env) or self.uri
        database = os.environ.get(self.database_env) or self.database
        user = os.environ.get(self.user_env)
        password = os.environ.get(self.password_env)
        missing = [name for name, value in ((self.user_env, user), (self.password_env, password)) if not value]
        if missing:
            raise ConfigError(
                "Neo4j credentials missing: set " + " and ".join(missing)
                + " in the environment or in a .env file (see .env.example)."
            )
        if self.id_function not in ("elementId", "id"):
            raise ConfigError("neo4j.id_function must be 'elementId' or 'id'")
        return uri, database, user, password


@dataclass
class DatasetConfig:
    name: str
    node_type: str
    context_max_hops: int = 3
    # "neo4j": export a snapshot from Neo4j (or reuse the one in output_dir/intermediate).
    # "snapshot": read an existing snapshot directory (snapshot_dir).
    source: str = "neo4j"
    snapshot_dir: str | None = None
    # Element type of a node with several labels: the first label of this list
    # it carries; otherwise node_type if present; otherwise the first label alphabetically.
    label_priority: list[str] = field(default_factory=list)
    # Optional node property used as node_id for nodes of node_type (e.g. a
    # domain id). Must be unique. Default: the database id.
    node_id_property: str | None = None


@dataclass
class CentralityConfig:
    measure: str = "pagerank"          # pagerank | degree
    epsilon: float = 1.0e-6
    directed: bool = True
    damping: float = 0.85
    max_iter: int = 200
    tol: float = 1.0e-10
    edge_measure: str = "endpoint_mean"


@dataclass
class SamplingConfig:
    max_n: int = 5
    max_H: int = 3
    max_G: int = 3
    seed: int = 42
    # Explicit sizes instead of 1..max_H / 1..max_G (e.g. sizes_G: [1, 3, 5]).
    sizes_H: list[int] | None = None
    sizes_G: list[int] | None = None
    # False: G is drawn from n's own properties only, and the G quartiles are computed over N only.
    use_context: bool = True
    # How properties are stratified. "popularity": H and G share one partition, the quartiles of pop(p)
    # over the distinct (element type, property) pairs of N ∪ context(N) (the draft's single set of
    # property-score pairs, with H stratified by popularity; TASKDEFINITION_SPEC.md §5.1).
    # "score": H by pop(p), G by its own quartiles of pop(p) * cent_p(e).
    G_strata_by: str = "popularity"

    @property
    def H_sizes(self) -> list[int]:
        return list(self.sizes_H) if self.sizes_H else list(range(1, self.max_H + 1))

    @property
    def G_sizes(self) -> list[int]:
        return list(self.sizes_G) if self.sizes_G else list(range(1, self.max_G + 1))


@dataclass
class PropertiesConfig:
    # Entries can be a property name ("id"), a type-qualified name
    # ("Airport.id", any element of type Airport) or a full prop_key.
    exclude_properties: list[str] = field(default_factory=list)
    never_hide: list[str] = field(default_factory=list)
    # Groups of prop_keys that reveal each other and are never split across H and G.
    leakage_pairs: list[list[str]] = field(default_factory=list)
    empty_string_is_null: bool = True
    # A string H value at least this long is also a leak when contained in a G value.
    leak_min_substring_len: int = 4


PUBLISHED_PROFILES: dict[str, dict[str, Any]] = {
    # What the published code (llmaugmentation, context.ipynb) actually did.
    "code": {
        "node_strata": "terciles",
        "property_classes": "threshold",
        "min_G": 2,
        "hidden_requires_value": False,
        "redraw_H_per_task": True,
    },
    # Algorithm 2 as printed in the DOLAP 2026 paper.
    "paper": {
        "node_strata": "quartiles",
        "property_classes": "node_quartiles",
        "min_G": None,
        "hidden_requires_value": True,
        "redraw_H_per_task": False,
    },
}


@dataclass
class PublishedConfig:
    """Task design of the published DOLAP 2026 paper (``tasks build-published``, ``tasks import-published``).

    ``profile`` sets the defaults of the five switches below; any of them can be overridden.
    """

    profile: str = "code"  # code | paper
    nodes_per_stratum: int = 3
    size_H: int = 3
    size_G: int = 3
    seed: int = 42
    node_strata: str | None = None  # terciles | quartiles
    property_classes: str | None = None  # threshold | node_quartiles
    easy_threshold: float = 0.8  # threshold classes: pop >= threshold is easy
    min_G: int | None = None  # code: top G up to 2 from the class pool; paper: exactly size_G
    hidden_requires_value: bool | None = None  # H drawn only among properties n has a value for
    redraw_H_per_task: bool | None = None  # code: new H for each of the 4 tasks; paper: H1, H4 fixed per node
    apply_leakage_pairs: bool = False  # the published design had no leakage check
    # Properties of the node type that can be drawn (the published EXPECTED_PROPERTIES).
    # Empty = every scalar property of the node type, minus properties.exclude_properties.
    properties: list[str] = field(default_factory=list)
    # Node property that identifies nodes in the published files (and in centrality_file).
    node_key: str = "icao"
    # Optional files from the published repo, for an exact replication:
    # datasets/ranks/ranks_pagerank.csv and datasets/ranks/rank_properties.csv.
    centrality_file: str | None = None
    centrality_id_column: str = "icao"
    centrality_score_column: str = "score"
    popularity_file: str | None = None  # columns prop,value; value in percent
    popularity_scale: float = 100.0
    # With-context condition: entries[start : randint(lo, hi)] of the node's context (published: 2, 5, 10).
    context_slice: list[int] = field(default_factory=lambda: [2, 5, 10])

    def resolved(self) -> dict[str, Any]:
        if self.profile not in PUBLISHED_PROFILES:
            raise ConfigError(f"published.profile must be one of {sorted(PUBLISHED_PROFILES)}")
        out = dict(PUBLISHED_PROFILES[self.profile])
        for key in out:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if out["min_G"] is None:
            out["min_G"] = self.size_G
        return out

    def validate(self) -> None:
        r = self.resolved()
        if r["node_strata"] not in ("terciles", "quartiles"):
            raise ConfigError("published.node_strata must be terciles or quartiles")
        if r["property_classes"] not in ("threshold", "node_quartiles"):
            raise ConfigError("published.property_classes must be threshold or node_quartiles")
        if min(self.nodes_per_stratum, self.size_H, self.size_G) < 1:
            raise ConfigError("published.nodes_per_stratum, size_H and size_G must be >= 1")
        if not 1 <= r["min_G"] <= self.size_G:
            raise ConfigError("published.min_G must be between 1 and size_G")
        if len(self.context_slice) != 3 or not 0 <= self.context_slice[0] <= self.context_slice[1] <= self.context_slice[2]:
            raise ConfigError("published.context_slice must be [start, lo, hi] with start <= lo <= hi")


@dataclass
class TaskDefConfig:
    dataset: DatasetConfig
    output_dir: str
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    centrality: CentralityConfig = field(default_factory=CentralityConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    properties: PropertiesConfig = field(default_factory=PropertiesConfig)
    published: PublishedConfig = field(default_factory=PublishedConfig)
    task_set_version: str | None = None

    @property
    def version(self) -> str:
        return self.task_set_version or Path(self.output_dir).name

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.dataset.source not in ("neo4j", "snapshot"):
            raise ConfigError("dataset.source must be 'neo4j' or 'snapshot'")
        if self.dataset.source == "snapshot" and not self.dataset.snapshot_dir:
            raise ConfigError("dataset.snapshot_dir is required when dataset.source is 'snapshot'")
        if self.dataset.context_max_hops < 1:
            raise ConfigError("dataset.context_max_hops must be >= 1")
        if self.centrality.measure not in ("pagerank", "degree"):
            raise ConfigError("centrality.measure must be 'pagerank' or 'degree'")
        if self.centrality.edge_measure != "endpoint_mean":
            raise ConfigError("centrality.edge_measure: only 'endpoint_mean' is implemented")
        if not 0 < self.centrality.epsilon < 1:
            raise ConfigError("centrality.epsilon must be in (0, 1)")
        for name in ("max_n", "max_H", "max_G"):
            if getattr(self.sampling, name) < 1:
                raise ConfigError(f"sampling.{name} must be >= 1")
        if self.sampling.G_strata_by not in ("score", "popularity"):
            raise ConfigError("sampling.G_strata_by must be 'score' or 'popularity'")
        for name in ("sizes_H", "sizes_G"):
            sizes = getattr(self.sampling, name)
            if sizes is not None and (
                not isinstance(sizes, list) or not sizes or any(not isinstance(x, int) or x < 1 for x in sizes)
                or len(set(sizes)) != len(sizes)
            ):
                raise ConfigError(f"sampling.{name} must be a list of distinct integers >= 1")
        for group in self.properties.leakage_pairs:
            if not isinstance(group, list) or len(group) < 2:
                raise ConfigError("properties.leakage_pairs must be a list of lists of at least two prop_keys")
        self.published.validate()


def _section(cls, data: Any, name: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {', '.join(sorted(unknown))}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid '{name}' section: {exc}") from exc


def load_config(path: str | Path) -> TaskDefConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("The config file must be a mapping")
    sections = {"dataset", "neo4j", "centrality", "sampling", "properties", "published", "output_dir", "task_set_version"}
    unknown = set(raw) - sections
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {', '.join(sorted(unknown))}")
    if "dataset" not in raw or "output_dir" not in raw:
        raise ConfigError("'dataset' and 'output_dir' are required")
    cfg = TaskDefConfig(
        dataset=_section(DatasetConfig, raw["dataset"], "dataset"),
        output_dir=str(raw["output_dir"]),
        neo4j=_section(Neo4jConfig, raw.get("neo4j"), "neo4j"),
        centrality=_section(CentralityConfig, raw.get("centrality"), "centrality"),
        sampling=_section(SamplingConfig, raw.get("sampling"), "sampling"),
        properties=_section(PropertiesConfig, raw.get("properties"), "properties"),
        published=_section(PublishedConfig, raw.get("published"), "published"),
        task_set_version=raw.get("task_set_version"),
    )
    cfg.validate()
    return cfg

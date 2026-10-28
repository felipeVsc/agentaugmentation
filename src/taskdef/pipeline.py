"""End-to-end task-set build (TASKDEFINITION_SPEC.md §3)."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .centrality import compute_centrality
from .config import TaskDefConfig
from .context import detect_n1, n1_keys
from .graph import Graph
from .leakage import LeakagePairs
from .output import ground_truth_rows, write_ground_truth, write_json, write_tasks
from .popularity import compute_popularity
from .sampling import BuildResult, TaskBuilder
from .validation import validate_tasks

log = logging.getLogger(__name__)

ASSUMPTIONS = [
    "Graph source is Neo4j; all later steps read the snapshot, not the live database.",
    "Centrality is normalized per element type (divided by the type's maximum); 0 is replaced by epsilon.",
    "Edge centrality is the mean of the normalized centralities of its two endpoints.",
    "Direct neighbors are taken in both directions; n:1 paths follow outgoing relationships only.",
    "Only scalar properties (string, integer, float, boolean, date) are eligible.",
    "H strata use pop(p) over the distinct H-eligible properties of the node type, each counted once.",
    "G strata use pop(p) * cent_p(e) over the set of elements in N ∪ context(N), each (element, property) once.",
    "Strata: low = [Q0, Q1], mid = (Q1, Q3], high = (Q3, Q4]; if Q3 == Q4, values equal to Q4 go to high.",
    "A combination with fewer candidates than size_H or size_G is skipped, not filled with fewer properties.",
    "G holds distinct prop_keys; a prop_key shared by several elements is drawn at most once per task.",
    "G excludes values that equal an H value, or contain a string H value of at least leak_min_substring_len chars.",
    "task_id hashes (dataset, node_id, sorted H prop_keys, sorted G (prop_key, element_id)).",
    "Random draws order candidates by sha256(seed, combination, candidate id).",
]


@dataclass
class BuildOutput:
    output_dir: Path
    result: BuildResult
    manifest: dict


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def snapshot_dir(cfg: TaskDefConfig) -> Path:
    return Path(cfg.output_dir) / "intermediate" / "snapshot"


def load_snapshot(cfg: TaskDefConfig, refresh: bool = False) -> tuple[Graph, dict]:
    """Read the snapshot, exporting it from Neo4j first if needed."""
    if cfg.dataset.source == "snapshot":
        path = Path(cfg.dataset.snapshot_dir)
        log.info("Reading snapshot from %s", path)
        return Graph.load(path), {"source": "snapshot", "path": str(path)}

    target = snapshot_dir(cfg)
    if Graph.exists(target) and not refresh:
        log.info("Reusing snapshot in %s (use --refresh-snapshot to export again)", target)
        return Graph.load(target), {"source": "neo4j", "path": str(target), "reused": True}

    from .neo4j_source import export_graph

    graph = export_graph(cfg.neo4j, cfg.dataset)
    graph.save(target)
    uri, database, _user, _password = cfg.neo4j.connection()
    log.info("Snapshot written to %s", target)
    return graph, {"source": "neo4j", "path": str(target), "reused": False, "uri": uri, "database": database}


def build(cfg: TaskDefConfig, refresh_snapshot: bool = False, overwrite: bool = False) -> BuildOutput:
    out_dir = Path(cfg.output_dir)
    if (out_dir / "tasks.csv").exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds a task set. Task sets are immutable: use a new output_dir "
            "(new task_set_version), or --overwrite while developing."
        )

    graph, source_info = load_snapshot(cfg, refresh=refresh_snapshot)
    inter = out_dir / "intermediate"
    inter.mkdir(parents=True, exist_ok=True)

    log.info("Computing centrality (%s)", cfg.centrality.measure)
    centrality = compute_centrality(graph, cfg.centrality)
    centrality.to_frame(graph).to_parquet(inter / "centrality.parquet", index=False)

    detected = detect_n1(graph)
    log.info("Computing popularity")
    popularity = compute_popularity(graph, cfg.properties.empty_string_is_null)
    popularity.to_frame().to_parquet(inter / "popularity.parquet", index=False)

    builder = TaskBuilder(graph, cfg, centrality, popularity, n1_keys(detected))
    log.info("Computing quartile partitions")
    partitions = builder.partitions()
    log.info("Sampling tasks")
    result = builder.build(partitions)

    import pandas as pd

    pd.DataFrame(result.context_rows).to_parquet(inter / "context.parquet", index=False)

    gt = ground_truth_rows(result.tasks)
    validate_tasks(
        result.tasks,
        gt,
        result.node_quartiles,
        result.h_quartiles,
        result.g_quartiles,
        LeakagePairs(cfg.properties.leakage_pairs),
        cfg.properties.leak_min_substring_len,
        builder.g_value,
    )

    write_tasks(out_dir / "tasks.csv", result.tasks, cfg.version)
    write_ground_truth(out_dir / "ground_truth.csv", gt)

    manifest = {
        "task_set_version": cfg.version,
        "dataset": cfg.dataset.name,
        "node_type": cfg.dataset.node_type,
        "config": cfg.as_dict(),
        "master_seed": cfg.sampling.seed,
        "code_version": code_version(),
        "package_version": __version__,
        "snapshot": {
            "hash": graph.content_hash(),
            "source": source_info.get("source"),
            "path": source_info.get("path"),
            "stats": graph.stats(),
        },
        "n1_relationships": [
            {"source_type": k[0], "relationship": k[1], "target_type": k[2], **v} for k, v in detected.items()
        ],
        "quartiles": {
            "node_centrality": list(result.node_quartiles.q),
            "H_property_popularity": list(result.h_quartiles.q),
            ("G_property_popularity" if cfg.sampling.G_strata_by == "popularity" else "G_property_score"): list(
                result.g_quartiles.q
            ),
            "intervals": "low=[Q0,Q1) mid=[Q1,Q3) high=[Q3,Q4]; Q0==Q1 ties go to low",
        },
        "strata": result.strata_summary,
        "report": result.report,
        "assumptions": ASSUMPTIONS,
    }
    write_json(out_dir / "manifest.json", manifest)
    log.info("Wrote %d tasks to %s", len(result.tasks), out_dir)
    return BuildOutput(out_dir, result, manifest)

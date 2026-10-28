"""Command line: ``tasks check-connection | snapshot | build --config <file>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _load_env(env_file: str) -> None:
    path = Path(env_file)
    if path.exists():
        from dotenv import load_dotenv

        load_dotenv(path, override=False)


def cmd_check_connection(args) -> int:
    from .neo4j_source import check_connection

    cfg = load_config(args.config)
    uri, database, user, _password = cfg.neo4j.connection()
    print(f"Connecting to {uri} (database: {database or 'default'}, user: {user})")
    info = check_connection(cfg.neo4j)
    print("Connected.")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if cfg.dataset.node_type not in info["labels"]:
        print(f"WARNING: no nodes with label {cfg.dataset.node_type!r} (dataset.node_type)")
        return 1
    return 0


def cmd_snapshot(args) -> int:
    from .pipeline import load_snapshot

    cfg = load_config(args.config)
    if cfg.dataset.source != "neo4j":
        print("dataset.source is not 'neo4j'; nothing to export.")
        return 1
    graph, info = load_snapshot(cfg, refresh=args.refresh)
    print(json.dumps({"snapshot": info, "stats": graph.stats(), "hash": graph.content_hash()}, indent=2))
    return 0


def cmd_build(args) -> int:
    from .pipeline import build

    cfg = load_config(args.config)
    output = build(cfg, refresh_snapshot=args.refresh_snapshot, overwrite=args.overwrite)
    report = output.manifest["report"]
    print(f"Task set {cfg.version}: {report['tasks_total']} tasks written to {output.output_dir}")
    print(f"  expected before skips: {report['expected_before_skips']}")
    print(f"  skipped: {report['skipped_total']} {report['skipped_by_reason']}")
    print(f"  duplicates removed: {report['duplicates_removed']}, zero difficulty: {report['zero_difficulty']}")
    print(f"  nodes available per stratum: {report['nodes_available']}")
    return 0


def _print_published(manifest: dict) -> None:
    summary = manifest["summary"]
    print(f"Task set {manifest['task_set_version']} ({manifest['design']}, profile {manifest['profile']}): {summary['tasks']} tasks")
    print(f"  per node stratum: {summary['per_node_stratum']}")
    print(f"  per H/G difficulty: {summary['per_difficulty_prop']}")
    print(f"  |G| given: {summary['size_G_given']}, |G| drawn: {summary['size_G_drawn']}")
    print(f"  H properties without ground truth: {summary['H_properties_without_ground_truth']}")
    if manifest.get("report"):
        print(f"  report: {manifest['report']}")
    check = manifest.get("difficulty_check")
    if check and check.get("tasks_compared"):
        print(f"  difficulty vs published: max |diff| = {check['max_abs_difference_vs_published']:.3g} over {check['tasks_compared']} tasks")


def cmd_build_published(args) -> int:
    from .published import build_published

    cfg = load_config(args.config)
    _print_published(build_published(cfg, refresh_snapshot=args.refresh_snapshot, overwrite=args.overwrite))
    return 0


def cmd_import_published(args) -> int:
    from .published import import_published

    cfg = load_config(args.config)
    manifest = import_published(
        cfg,
        Path(args.results),
        entities_context=Path(args.entities_context) if args.entities_context else None,
        node_values=Path(args.node_values) if args.node_values else None,
        use_graph=not args.no_graph,
        overwrite=args.overwrite,
    )
    _print_published(manifest)
    print(f"  node values: {manifest['node_values_source']}; ground_truth.csv written: {manifest['ground_truth_written']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tasks", description="Build benchmark task sets (TASKDEFINITION_SPEC.md)")
    parser.add_argument("--env-file", default=".env", help="file with NEO4J_* variables (default: .env)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-connection", help="connect to Neo4j and list labels and relationship types")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_check_connection)

    p = sub.add_parser("snapshot", help="export the graph snapshot from Neo4j only")
    p.add_argument("--config", required=True)
    p.add_argument("--refresh", action="store_true", help="export again even if a snapshot exists")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("build", help="build the task set")
    p.add_argument("--config", required=True)
    p.add_argument("--refresh-snapshot", action="store_true", help="export from Neo4j even if a snapshot exists")
    p.add_argument("--overwrite", action="store_true", help="replace an existing task set (development only)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("build-published", help="draw a task set with the published DOLAP 2026 design")
    p.add_argument("--config", required=True)
    p.add_argument("--refresh-snapshot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_build_published)

    p = sub.add_parser("import-published", help="convert the published task file (test_results.csv)")
    p.add_argument("--config", required=True)
    p.add_argument("--results", required=True, help="published test_results.csv")
    p.add_argument("--node-values", help="CSV of node values keyed by published.node_key (published ground_truth.csv)")
    p.add_argument("--entities-context", help="published context_entities.csv (alternative source of node values)")
    p.add_argument("--no-graph", action="store_true", help="do not read the graph (needs the published rank files)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_import_published)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _load_env(args.env_file)
    try:
        return args.func(args)
    except (ConfigError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

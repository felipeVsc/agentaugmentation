"""Command line for Strategy B.

    strategy-b check  --config C          config, search key, models ready, PFE, search credit budget
    strategy-b render --config C [--task ID] [--variant V]   print the plan prompt of one task
    strategy-b run    --config C [--models a,b] [--variants v] [--limit N] [--runs N] [--no-cache] [--no-search-cache]
                      [--retry-failed]
    strategy-b replay --config C --source-dir D --source-variant V [--models a,b] [--limit N] [--runs N]
                      new synthesis call on the plan and evidence of finished runs; no search

The search key is read from TAVILY_API_KEY or OLLAMA_API_KEY (environment or .env), never from the config.
--ollama cloud|local|URL runs the Ollama models on Ollama Cloud or on an on-premise server; the Ollama web search
itself always runs on ollama.com.
"""

from __future__ import annotations

import argparse
import sys

from llmcore.config import ConfigError

from . import steps
from .config import load_config
from .runner import Runner


FAILED_STATUSES = {"api_error", "timeout", "retrieval_failed", "search_partial"}


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _runner(args, **kw) -> Runner:
    cfg = load_config(args.config, ollama=getattr(args, "ollama", None))
    if getattr(args, "runs", None):
        cfg.runs_per_task = args.runs
    return Runner(cfg, _csv(args.models), _csv(args.variants), args.limit, **kw)


def cmd_check(args) -> int:
    """`strategy-b check`: load everything, check models and the search tool, and report problems."""
    r = _runner(args)
    r.prepare()
    cfg = r.cfg
    problems = 0
    print(f"tasks: {len(r.tasks)} (task set {r.task_set_version}); variants: {', '.join(v.id for v in r.variants)}")
    print(f"search: {r.search.tool.name} {dict((k, v) for k, v in cfg.search.items() if k != 'api_key_env')}")
    print(f"blocked domains: {cfg.blocked_domains or 'none'}")
    for m in r.models:
        try:
            d = r.callers[m.id].provider.describe()
            det = d.get("details") or {}
            print(f"model {m.id}: {d.get('model', m.model)} params={det.get('parameter_size')} quant={det.get('quantization_level')}")
        except Exception as exc:
            print(f"model {m.id}: NOT READY: {exc}")
            problems += 1
    for (v, m), n in sorted(r.pfe.items()):
        print(f"pfe {v} × {m}: {n} tokens")
    # Variants with the same plan prompt share the plan (LLM cache) and therefore the searches.
    plans = len({(v.plan, v.with_context) for v in r.variants})
    retrievals = len(r.tasks) * len(r.models) * plans * (1 if cfg.reuse_retrieval else cfg.runs_per_task)
    per_call = r.search.tool.credits_per_call()
    # The plan has at most max_subqueries model queries plus default queries for untargeted H properties.
    worst = retrievals * (cfg.max_subqueries + max(len(t.H) for t in r.tasks)) * per_call
    typical = retrievals * max(len(t.H) for t in r.tasks) * per_call
    print(f"retrieval steps: {retrievals} (reuse_retrieval={cfg.reuse_retrieval}); "
          f"search credits ≈ {typical:.0f} typical (one query per hidden property), ≤ {worst:.0f} worst case, "
          f"before cache hits (cache: {cfg.search_cache_path})"
          + (f"; cap {cfg.search_max_credits:g} credits per run" if cfg.search_max_credits is not None else ""))
    print(f"runs: {len(r.jobs())} planned, {len(r.pending())} pending")
    return 1 if problems else 0


def cmd_render(args) -> int:
    """`strategy-b render`: print the plan prompt of one task and variant."""
    r = _runner(args)
    r.prepare()
    jobs = [j for j in r.jobs() if j.run_index == 1]
    if args.task:
        jobs = [j for j in jobs if j.task.task_id == args.task]
    if args.variant:
        jobs = [j for j in jobs if j.variant.id == args.variant]
    if not jobs:
        print("no matching task/variant", file=sys.stderr)
        return 1
    j = jobs[0]
    system, user, labels = steps.render_plan(r.plan_tpls[j.variant.id], j.task, j.variant.with_context, r.cfg.max_subqueries,
                                             r.notes)
    print(f"# task {j.task.task_id} variant {j.variant.id}\n")
    if system:
        print("=== PLAN SYSTEM ===\n" + system + "\n")
    print("=== PLAN USER ===\n" + user)
    print(f"\n=== labels → prop_key ===\n{labels}")
    return 0


def cmd_run(args) -> int:
    """`strategy-b run`: run the agent on the pending jobs (resumable)."""
    counts = _runner(args, read_cache=not args.no_cache, read_search_cache=not args.no_search_cache).run(
        retry_statuses=FAILED_STATUSES if args.retry_failed else None)
    print(counts)
    return 0


def cmd_replay(args) -> int:
    """`strategy-b replay`: redo only the synthesis call of earlier runs on the same evidence (search_memory)."""
    from .replay import ReplayRunner

    cfg = load_config(args.config, ollama=getattr(args, "ollama", None))
    if args.runs:
        cfg.runs_per_task = args.runs
    runner = ReplayRunner(cfg, args.source_dir, args.source_variant, model_ids=_csv(args.models),
                          variant_ids=_csv(args.variants), limit=args.limit, read_cache=not args.no_cache)
    print(runner.run(retry_statuses=FAILED_STATUSES if args.retry_failed else None))
    return 0


def main(argv=None) -> int:
    """Entry point of the `strategy-b` command."""
    parser = argparse.ArgumentParser(prog="strategy-b", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--models")
        p.add_argument("--variants")
        p.add_argument("--limit", type=int, help="only the first N tasks")
        p.add_argument("--runs", type=int, help="runs per task (overrides runs_per_task, e.g. 1 for a test)")
        p.add_argument("--env-file", default=".env")
        p.add_argument("--ollama", help="where the Ollama models run: cloud, local ($OLLAMA_HOST or localhost:11434) "
                       "or a URL; overrides the config (ids become local-*/cloud-*, output_dir gets _local/_cloud)")
        return p

    common(sub.add_parser("check")).set_defaults(func=cmd_check)
    p = common(sub.add_parser("render"))
    p.add_argument("--task")
    p.add_argument("--variant")
    p.set_defaults(func=cmd_render)
    p = common(sub.add_parser("run"))
    p.add_argument("--no-cache", action="store_true", help="ignore cached LLM answers (new ones are stored)")
    p.add_argument("--no-search-cache", action="store_true",
                   help="search again instead of reading cached searches (spends credits; new ones are stored)")
    p.add_argument("--retry-failed", action="store_true",
                   help="rerun rows with status api_error/timeout/retrieval_failed (cached steps are reused)")
    p.set_defaults(func=cmd_run)
    p = common(sub.add_parser("replay", help="new synthesis call on the plan and evidence of finished runs (no search)"))
    p.add_argument("--source-dir", required=True, help="output_dir of the runs to replay (its runs.jsonl)")
    p.add_argument("--source-variant", required=True)
    p.add_argument("--no-cache", action="store_true", help="ignore cached LLM answers (new ones are stored)")
    p.add_argument("--retry-failed", action="store_true", help="rerun rows with status api_error/timeout")
    p.set_defaults(func=cmd_replay)
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    except ImportError:
        pass
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

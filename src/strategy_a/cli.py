"""Command line for Strategy A.

    strategy-a check    --config C [--ping]        validate config, keys, templates, EcoLogits models
    strategy-a render   --config C [--task ID] [--variant V]   print a rendered prompt
    strategy-a estimate --config C [--pending]     tokens and cost, no model calls
    strategy-a run      --config C [--cache-only] [--retry-failed] [--no-cache]
    strategy-a batch submit|status|collect --config C [--dry-run]

Common filters: --models a,b  --templates t1,t2 (template or variant ids)  --limit N (first N tasks)
                --runs N (runs per task, overrides runs_per_task).
API keys are read from the environment or from .env, never from the config.
"""

from __future__ import annotations

import argparse
import sys

from llmcore.config import ConfigError
from llmcore.types import Request

from . import batch
from .config import load_config
from .estimate import estimate, print_estimate
from .runner import Runner

FAILED_STATUSES = {"api_error", "timeout"}


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _runner(args, read_cache: bool = True, connect: bool = True) -> Runner:
    cfg = load_config(args.config)
    if getattr(args, "runs", None):
        cfg.runs_per_task = args.runs
    return Runner(cfg, _csv(args.models), _csv(args.templates), args.limit, read_cache=read_cache, connect=connect)


def cmd_check(args) -> int:
    runner = _runner(args)
    runner.prepare()  # loads tasks, templates, examples, providers (keys), energy models, tokenizers
    print(f"tasks: {len(runner.tasks)} (task set {runner.task_set_version})")
    print(f"variants: {', '.join(v.id for v in runner.variants)}")
    jobs = runner.jobs()
    problems = 0
    for m in runner.models:
        caller = runner.callers[m.id]
        batch_ok = "batch" if caller.provider.supports_batch else "no batch"
        print(f"model {m.id}: provider={m.provider} model={m.model} energy={caller.meter.method} {batch_ok} tokenizer={runner.tokenizers[m.id].name}")
        try:
            details = caller.provider.describe()
        except Exception as exc:
            print(f"  NOT READY: {exc}")
            problems += 1
            details = {}
        if details:
            d = details.get("details") or {}
            print(
                f"  {details.get('model')} digest={str(details.get('digest'))[:12]} "
                f"params={d.get('parameter_size')} quant={d.get('quantization_level')} "
                f"context_length={details.get('context_length')} num_ctx={details.get('num_ctx')} "
                f"server={details.get('ollama_version')}"
            )
        num_ctx = getattr(caller.provider, "num_ctx", None)
        if num_ctx:
            # Ollama drops the start of a prompt longer than num_ctx without an error.
            tok = runner.tokenizers[m.id]
            longest = max(
                (tok.count((j.rendered.system or "") + "\n" + j.rendered.user) for j in jobs if j.model.id == m.id and j.run_index == 1),
                default=0,
            )
            need = longest + int(m.params.get("max_tokens", 1024))
            note = "OK" if need <= num_ctx * 0.9 else "TOO SMALL: raise options.num_ctx"
            print(f"  longest prompt ≈ {longest} tokens ({tok.name}) + max_tokens {m.params.get('max_tokens')} vs num_ctx {num_ctx}: {note}")
            problems += note != "OK"
        if args.ping:
            req = Request(system=None, messages=[{"role": "user", "content": "Reply with: ok"}], params={**m.params, "max_tokens": 16}, run_index=0)
            try:
                resp = caller.provider.generate(req)
                print(f"  ping ok: {resp.text.strip()[:40]!r} in={resp.tokens_in} out={resp.tokens_out} version={resp.model_version}")
            except Exception as exc:
                print(f"  ping FAILED: {exc}")
    for (v, m), n in sorted(runner.pfe.items()):
        print(f"pfe {v} × {m}: {n} tokens")
    print(f"runs: {len(jobs)} planned, {len(runner.pending())} pending")
    if problems:
        print(f"{problems} problem(s) found")
    return 1 if problems else 0


def cmd_render(args) -> int:
    runner = _runner(args, connect=False)
    jobs = [j for j in runner.jobs() if j.run_index == 1]
    if args.task:
        jobs = [j for j in jobs if j.task.task_id == args.task]
    if args.variant:
        jobs = [j for j in jobs if j.variant.id == args.variant]
    if not jobs:
        print("no matching task/variant", file=sys.stderr)
        return 1
    j = jobs[0]
    print(f"# task {j.task.task_id}  variant {j.variant.id}  model {j.model.id}\n")
    if j.rendered.system:
        print("=== SYSTEM ===\n" + j.rendered.system + "\n")
    print("=== USER ===\n" + j.rendered.user)
    print(f"\n=== labels → prop_key ===\n{j.rendered.labels}")
    return 0


def cmd_estimate(args) -> int:
    print_estimate(estimate(_runner(args, connect=False), pending_only=args.pending))
    return 0


def cmd_run(args) -> int:
    runner = _runner(args, read_cache=not args.no_cache)
    counts = runner.run(cache_only=args.cache_only, retry_statuses=FAILED_STATUSES if args.retry_failed else None)
    if counts.get("not_cached"):
        print(f"{counts['not_cached']} runs skipped: not in cache (collect the batches, or run without --cache-only)", file=sys.stderr)
    return 0


def cmd_batch(args) -> int:
    runner = _runner(args)
    if args.action == "submit":
        batch.submit(runner, dry_run=args.dry_run)
    elif args.action == "status":
        batch.status(runner)
    else:
        print(batch.collect(runner))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="strategy-a", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--models", help="comma-separated model ids (default: all enabled)")
        p.add_argument("--templates", help="comma-separated template or variant ids")
        p.add_argument("--limit", type=int, help="only the first N tasks")
        p.add_argument("--runs", type=int, help="runs per task (overrides runs_per_task, e.g. 1 for a test)")
        p.add_argument("--env-file", default=".env")
        return p

    p = common(sub.add_parser("check"))
    p.add_argument("--ping", action="store_true", help="one tiny real call per model")
    p.set_defaults(func=cmd_check)
    p = common(sub.add_parser("render"))
    p.add_argument("--task")
    p.add_argument("--variant")
    p.set_defaults(func=cmd_render)
    p = common(sub.add_parser("estimate"))
    p.add_argument("--pending", action="store_true", help="only runs not yet in results.csv")
    p.set_defaults(func=cmd_estimate)
    p = common(sub.add_parser("run"))
    p.add_argument("--cache-only", action="store_true", help="never make a first call; use cached/batch responses only")
    p.add_argument("--retry-failed", action="store_true", help="rerun rows with status api_error/timeout")
    p.add_argument("--no-cache", action="store_true", help="ignore cached responses (new ones are still stored)")
    p.set_defaults(func=cmd_run)
    p = common(sub.add_parser("batch"))
    p.add_argument("action", choices=["submit", "status", "collect"])
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_batch)

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

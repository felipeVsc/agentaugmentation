"""Batch mode: send the first call of every pending run through the provider's
batch API (about half the price), then let ``run`` build results.csv from the cache.

    strategy-a batch submit  --config ...   # one batch per model
    strategy-a batch status  --config ...
    strategy-a batch collect --config ...   # downloads finished batches into the cache
    strategy-a run --cache-only --config ...

Batch results go into the same cache as realtime calls, under the same key, so
``run`` cannot tell them apart except for ``execution: batch`` and an empty
``latency_s`` (queue time is not model time). Repair calls, if needed, are made
in real time by ``run``.

State lives in ``<output_dir>/batches/``: ``batches.json`` and one
``<local_id>.requests.jsonl`` per batch (needed to store the results in the cache).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from llmcore.types import ProviderError, Request

from .runner import Runner
from llmcore.util import now_iso


def _state_path(runner: Runner) -> Path:
    return runner.out_dir / "batches" / "batches.json"


def _load(runner: Runner) -> dict:
    path = _state_path(runner)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"batches": []}


def _save(runner: Runner, state: dict) -> None:
    path = _state_path(runner)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _open_keys(runner: Runner, state: dict) -> set[str]:
    keys = set()
    for b in state["batches"]:
        if b["status"] != "collected":
            with open(runner.out_dir / "batches" / b["requests_file"], encoding="utf-8") as fh:
                keys.update(json.loads(line)["key"] for line in fh)
    return keys


def submit(runner: Runner, dry_run: bool = False) -> list[dict]:
    runner.prepare()
    state = _load(runner)
    in_flight = _open_keys(runner, state)
    per_model: dict[str, dict[str, Request]] = defaultdict(dict)
    for job in runner.pending():
        caller = runner.callers[job.model.id]
        req = runner.request(job)
        key = caller.key(req)
        if key in runner.cache or key in in_flight:
            continue
        per_model[job.model.id][key] = req
    submitted = []
    for model in runner.models:
        items = list(per_model.get(model.id, {}).items())
        caller = runner.callers[model.id]
        if not items:
            runner.log(f"{model.id}: nothing to submit")
            continue
        if not caller.provider.supports_batch:
            runner.log(f"{model.id}: provider has no batch API, {len(items)} calls left for `run`")
            continue
        if dry_run:
            runner.log(f"{model.id}: would submit {len(items)} requests")
            continue
        local_id = f"{model.id}-{now_iso().replace(':', '').replace('+', 'Z')[:17]}-{len(state['batches'])}"
        local_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in local_id)
        requests_file = f"{local_id}.requests.jsonl"
        path = runner.out_dir / "batches" / requests_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for key, req in items:
                fh.write(json.dumps({"key": key, "request": req.as_dict()}, ensure_ascii=False) + "\n")
        provider_state = caller.provider.batch_submit(items)
        entry = {
            "local_id": local_id,
            "model": model.id,
            "n": len(items),
            "submitted_at": now_iso(),
            "status": "submitted",
            "requests_file": requests_file,
            "provider_state": provider_state,
        }
        state["batches"].append(entry)
        _save(runner, state)  # save after each model, so a crash never loses a submitted batch
        submitted.append(entry)
        runner.log(f"{model.id}: submitted {len(items)} requests ({local_id})")
    return submitted


def status(runner: Runner) -> list[dict]:
    runner.prepare()
    state = _load(runner)
    rows = []
    for b in state["batches"]:
        if b["status"] != "collected" and b["model"] in runner.callers:
            b["status"] = runner.callers[b["model"]].provider.batch_poll(b["provider_state"])
        rows.append({k: b[k] for k in ("local_id", "model", "n", "submitted_at", "status")})
        runner.log(f"{b['local_id']}: {b['status']} ({b['n']} requests)")
    _save(runner, state)
    return rows


def collect(runner: Runner) -> dict:
    runner.prepare()
    state = _load(runner)
    summary = defaultdict(int)
    for b in state["batches"]:
        if b["status"] == "collected" or b["model"] not in runner.callers:
            continue
        caller = runner.callers[b["model"]]
        poll = caller.provider.batch_poll(b["provider_state"])
        if poll == "running":
            runner.log(f"{b['local_id']}: still running")
            summary["running"] += 1
            continue
        requests = {}
        with open(runner.out_dir / "batches" / b["requests_file"], encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                requests[rec["key"]] = Request(**rec["request"])
        results = caller.provider.batch_fetch(b["provider_state"]) if poll != "failed" else {}
        ok = errors = 0
        for key, result in results.items():
            if key not in requests:
                continue
            if isinstance(result, ProviderError):
                errors += 1
                continue
            caller.store_batch_result(key, requests[key], result)
            ok += 1
        missing = len(requests) - ok - errors
        b.update(status="collected", collected_at=now_iso(), ok=ok, errors=errors, missing=missing)
        summary["ok"] += ok
        summary["errors"] += errors
        summary["missing"] += missing
        _save(runner, state)
        runner.log(
            f"{b['local_id']}: {ok} responses cached, {errors} errors, {missing} missing"
            + (" — submit again to retry them in a new batch" if errors or missing else "")
        )
    return dict(summary)

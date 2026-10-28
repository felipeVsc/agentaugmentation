"""Strategy A runner (LLM_SPEC.md §4, §6, §7, §8).

for model × variant × task × run_index: render → call → parse (→ one repair call) → write a row.
Calls run concurrently per model; finished runs are skipped on restart.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from llmcore.cache import DiskCache
from llmcore.caller import Caller
from llmcore.config import ConfigError, ModelConfig
from llmcore.providers import Provider
from llmcore.tokenizers import Tokenizer, get_tokenizer, provider_tokenizer, reasoning_tokens
from llmcore.types import CallFailed, Request, Response
from llmcore.util import code_version, file_sha256, now_iso

from .config import StrategyAConfig, Variant
from .parsing import repair_instruction, parse_response
from .results import ResultsWriter, drop_statuses, load_done
from .tasks import Example, Task, check_example_leakage, load_examples, load_tasks
from .templates import (
    Rendered,
    Template,
    load_template,
    pfe_text,
    placeholders,
    render,
    render_examples,
    render_output_format,
    task_effort_texts,
    text_sizes,
)

STRATEGY = "A"


@dataclass
class Job:
    task: Task
    variant: Variant
    model: ModelConfig
    run_index: int
    rendered: Rendered

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.task.task_id, self.model.id, self.variant.id, self.run_index)


class Runner:
    def __init__(
        self,
        cfg: StrategyAConfig,
        model_ids: list[str] | None = None,
        variant_ids: list[str] | None = None,
        limit: int | None = None,
        read_cache: bool = True,
        providers: dict[str, Provider] | None = None,
        log: Callable[[str], None] | None = None,
        connect: bool = True,
    ):
        self.cfg = cfg
        self.models = cfg.select_models(model_ids)
        self.variants = cfg.select_variants(variant_ids)
        self.out_dir = cfg.path(cfg.output_dir)
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        self.read_cache = read_cache
        self._providers = providers or {}
        self._limit = limit
        self._prepared = False
        # connect=False: no providers are created (no API keys needed), for render/estimate.
        self.connect = connect

    # Setup ----------------------------------------------------------------

    def prepare(self) -> None:
        if self._prepared:
            return
        cfg = self.cfg
        self.tasks = load_tasks(cfg.path(cfg.tasks))
        if self._limit:
            self.tasks = self.tasks[: self._limit]
        self.task_set_version = self.tasks[0].task_set_version

        self.examples_version, self.examples = None, []
        if cfg.examples and any(v.n_examples for v in self.variants):
            self.examples_version, self.examples = load_examples(cfg.path(cfg.examples))
            check_example_leakage(self.examples, load_tasks(cfg.path(cfg.tasks)))

        self.templates: dict[str, Template] = {}
        self.examples_text: dict[str, str] = {}
        for v in self.variants:
            tpl = self.templates.setdefault(v.template.id, load_template(cfg.path(v.template.file)))
            if v.n_examples:
                if not tpl.uses_examples:
                    raise ConfigError(f"template {v.template.id!r} has n_examples but no {{examples}} placeholder")
                if v.n_examples > len(self.examples):
                    raise ConfigError(f"variant {v.id!r} needs {v.n_examples} examples, the file has {len(self.examples)}")
            chosen: list[Example] = self.examples[: v.n_examples]
            self.examples_text[v.id] = render_examples(tpl, chosen, v.template.context_format) if chosen else ""

        self.cache = DiskCache(cfg.cache_path)
        self.callers: dict[str, Caller] = {}
        if self.connect:
            for m in self.models:
                self.callers[m.id] = Caller(m, self.cache, provider=self._providers.get(m.id), read_cache=self.read_cache)

        self.tokenizers: dict[str, Tokenizer] = {}
        for m in self.models:
            spec = m.tokenizer or cfg.pfe_tokenizer
            if spec == "provider":
                self.tokenizers[m.id] = (
                    provider_tokenizer(self.callers[m.id].provider) if self.connect else get_tokenizer(cfg.pfe_tokenizer)
                )
            else:
                self.tokenizers[m.id] = get_tokenizer(spec)
        # PFE per variant × model, with the model's tokenizer (§5.6).
        self.pfe: dict[tuple[str, str], int] = {}
        self.pfe_texts: dict[str, str] = {}
        for v in self.variants:
            text = self.pfe_texts[v.id] = pfe_text(self.templates[v.template.id], self.examples_text[v.id])
            for m in self.models:
                self.pfe[(v.id, m.id)] = self.tokenizers[m.id].count(text)
        self._prepared = True

    # Planning -------------------------------------------------------------

    def jobs(self) -> list[Job]:
        self.prepare()
        jobs = []
        for v in self.variants:
            tpl = self.templates[v.template.id]
            for task in self.tasks:
                rendered = render(tpl, task, self.examples_text[v.id], v.template.context_format, v.template.with_context)
                for m in self.models:
                    for r in range(1, self.cfg.runs_per_task + 1):
                        jobs.append(Job(task, v, m, r, rendered))
        return jobs

    def request(self, job: Job) -> Request:
        return Request(
            system=job.rendered.system,
            messages=[{"role": "user", "content": job.rendered.user}],
            params=job.model.params,
            run_index=job.run_index,
        )

    def pending(self) -> list[Job]:
        self.prepare()
        done = load_done(self.out_dir / "results.csv", self.task_set_version)
        return [j for j in self.jobs() if j.key not in done]

    # Execution ------------------------------------------------------------

    def run(self, cache_only: bool = False, retry_statuses: set[str] | None = None) -> dict:
        self.prepare()
        if retry_statuses:
            n = drop_statuses(self.out_dir / "results.csv", self.out_dir / "runs.jsonl", retry_statuses)
            self.log(f"removed {n} runs with status in {sorted(retry_statuses)}; they will run again")
        pending = self.pending()
        total = len(self.jobs())
        self.log(f"{total} runs planned, {total - len(pending)} already in results.csv, {len(pending)} to do")
        started = now_iso()
        writer = ResultsWriter(self.out_dir)
        self._counts: Counter = Counter()
        self._model_versions: dict[str, set] = {m.id: set() for m in self.models}
        self._lock = threading.Lock()
        executors = {m.id: ThreadPoolExecutor(max_workers=m.max_concurrency) for m in self.models}
        try:
            futures = [executors[j.model.id].submit(self._process, j, writer, cache_only) for j in pending]
            for f in futures:
                f.result()  # re-raise unexpected errors
        except KeyboardInterrupt:
            self.log("interrupted; finished runs are saved, run again to resume")
            raise
        finally:
            for ex in executors.values():
                ex.shutdown(wait=True, cancel_futures=True)
            writer.close()
            for caller in self.callers.values():
                try:
                    caller.close()
                except Exception as exc:  # never lose the manifest over a meter shutdown
                    self.log(f"closing {caller.cfg.id}: {exc}")
            self.write_manifest(started)
        self.log(f"done: {dict(self._counts)}")
        return dict(self._counts)

    def _process(self, job: Job, writer: ResultsWriter, cache_only: bool) -> None:
        caller = self.callers[job.model.id]
        request = self.request(job)
        calls: list[dict] = []
        responses: list[Response] = []
        requests: list[Request] = []
        status, events, parsed, repaired = "ok", [], None, False
        predictions = {h["prop_key"]: None for h in job.task.H}
        try:
            first = caller.call(request, cache_only=cache_only)
            if first is None:
                self._count("not_cached")
                return
            responses.append(first)
            requests.append(request)
            calls.append({"request": request.as_dict(), "response": first.as_dict(), "cached": first.cached})
            output = job.variant.template.output
            result = parse_response(first.text, job.rendered.labels, job.task.H, output)
            events += result.events
            if not result.ok and self.cfg.repair:
                repair_req = Request(
                    system=request.system,
                    messages=request.messages
                    + [
                        {"role": "assistant", "content": first.text},
                        {
                            "role": "user",
                            "content": repair_instruction(output, job.rendered.labels),
                        },
                    ],
                    params=job.model.params,
                    run_index=job.run_index,
                )
                second = caller.call(repair_req)
                responses.append(second)
                requests.append(repair_req)
                calls.append({"request": repair_req.as_dict(), "response": second.as_dict(), "cached": second.cached, "repair": True})
                repaired = True
                result2 = parse_response(second.text, job.rendered.labels, job.task.H, output)
                events += [f"repair:{e}" for e in result2.events]
                if result2.ok:
                    result = result2
            if result.ok:
                predictions, parsed = result.predictions, result.parsed
                status = "truncated" if first.finish_reason == "length" else "ok"
            else:
                status = "parse_error"
        except CallFailed as exc:
            status = exc.status
            events.append(f"call_failed:{exc} (attempts={exc.attempts})")
        except Exception as exc:  # a provider bug must not kill the whole batch
            status = "api_error"
            events.append(f"unexpected:{type(exc).__name__}: {exc}")

        usage = self._usage(job, requests, responses)
        row = {
            "task_set_version": job.task.task_set_version,
            "task_id": job.task.task_id,
            "strategy": STRATEGY,
            "variant": job.variant.id,
            "model": job.model.id,
            "run_index": job.run_index,
            "status": status,
            "predictions": json.dumps(predictions, ensure_ascii=False),
            "evidence": "{}",
            "pfe_tokens": self.pfe[(job.variant.id, job.model.id)],
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "tokens_reasoning": "" if usage["tokens_reasoning"] is None else usage["tokens_reasoning"],
            "llm_calls": usage["calls"],
            "search_calls": 0,
            "latency_s": _fmt(usage["latency_s"]),
            "energy_kwh": _fmt(usage["energy_kwh"]),
            "co2e_kg": _fmt(usage["co2e_kg"]),
            "cost_usd": _fmt(usage["cost_usd"]),
            "prompt_version": job.variant.template.version,
            "timestamp": responses[0].created_at if responses else now_iso(),
        }
        log = {
            "task_id": job.task.task_id,
            "model": job.model.id,
            "variant": job.variant.id,
            "template": job.variant.template.id,
            "run_index": job.run_index,
            "status": status,
            "system": request.system,
            "user": request.messages[0]["content"],
            "label_map": job.rendered.labels,
            "calls": calls,
            "parsed": parsed,
            "predictions": predictions,
            "parse_events": events,
            "repaired": repaired,
            "usage": usage,
            "effort": self._effort(job),
            "timestamp": row["timestamp"],
        }
        writer.write(row, log)
        with self._lock:
            for r in responses:
                if r.model_version:
                    self._model_versions[job.model.id].add(r.model_version)
        self._count(status)

    def _count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1
            n = sum(v for k, v in self._counts.items() if k != "not_cached")
            if n and n % 100 == 0:
                self.log(f"{n} runs written: {dict(self._counts)}")

    def _usage(self, job: Job, requests: list[Request], responses: list[Response]) -> dict:
        """Totals over the run's calls; counts missing from the provider come from the model's tokenizer
        (Ollama with ``tokens_in_from: tokenizer`` always leaves tokens_in to it)."""
        tok = self.tokenizers[job.model.id]
        tokens_in = tokens_out = 0
        estimated = False
        reported_in, cached_in, reasoning, costs = [], [], [], []
        for req, r in zip(requests, responses):
            if r.raw.get("prompt_eval_count") is not None:
                reported_in.append(r.raw["prompt_eval_count"])
            if r.raw.get("prompt_eval_cached_count") is not None:
                cached_in.append(r.raw["prompt_eval_cached_count"])
            if r.tokens_in is None:
                prompt = "\n".join([req.system or ""] + [m["content"] for m in req.messages])
                call_in = tok.count(prompt)
                estimated = True
            else:
                call_in = r.tokens_in
            if r.tokens_out is None:
                call_out = tok.count(r.text)
                estimated = True
            else:
                call_out = r.tokens_out
            tokens_in += call_in
            tokens_out += call_out
            reasoning.append(reasoning_tokens(r.raw, tok))
            costs.append(job.model.pricing.cost(call_in, call_out, r.execution) if job.model.pricing else None)
        lat = [r.latency_s for r in responses]
        energy = [r.energy or {} for r in responses]
        e_kwh = [e.get("energy_kwh") for e in energy]
        co2 = [e.get("co2e_kg") for e in energy]
        return {
            "tokens_in": tokens_in if responses else 0,
            "tokens_out": tokens_out if responses else 0,
            "usage_estimated": estimated,
            "tokens_in_counted_with": tok.name if estimated else "provider",
            "prompt_eval_count": sum(reported_in) if reported_in else None,
            "latency_s": sum(lat) if responses and None not in lat else None,
            "energy_kwh": sum(e_kwh) if responses and None not in e_kwh else None,
            "co2e_kg": sum(co2) if responses and None not in co2 else None,
            "energy_method": energy[0].get("method") if energy and energy[0] else None,
            "execution": [r.execution for r in responses],
            "calls": len(responses),
            # Part of tokens_out; None when the provider gives no reasoning (or the model has none).
            "tokens_reasoning": sum(reasoning) if responses and None not in reasoning else None,
            "tokens_in_cached": sum(cached_in) if cached_in else None,
            "cost_usd": sum(costs) if responses and None not in costs else None,
            "cost_usd_per_call": costs,
        }

    def _effort(self, job: Job) -> dict:
        """Sizes of what a person writes: the template once per variant, the task part per task (METRICS.md §6)."""
        texts = task_effort_texts(job.task, job.variant.template.with_context,
                                  "published_input" in placeholders(self.templates[job.variant.template.id].user))
        return text_sizes({"template": self.pfe_texts[job.variant.id], **texts}, self.tokenizers[job.model.id])

    # Manifest -------------------------------------------------------------

    def _model_details(self) -> dict:
        return {m: _describe(c) for m, c in self.callers.items()}

    def write_manifest(self, started: str) -> None:
        cfg = self.cfg
        path = self.out_dir / "run_manifest.json"
        previous = {}
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
        status_counts = Counter()
        results = self.out_dir / "results.csv"
        if results.exists():
            from .results import read_results

            status_counts = Counter(r["status"] for r in read_results(results))
        versions = {k: sorted(v) for k, v in getattr(self, "_model_versions", {}).items()}
        for k, v in (previous.get("model_versions") or {}).items():
            versions[k] = sorted(set(v) | set(versions.get(k, [])))
        manifest = {
            "strategy": STRATEGY,
            "task_set_version": self.task_set_version,
            "tasks_file": {"path": str(cfg.tasks), "sha256": file_sha256(cfg.path(cfg.tasks))},
            "examples_file": {"path": str(cfg.examples), "sha256": file_sha256(cfg.path(cfg.examples)), "version": self.examples_version}
            if self.examples
            else None,
            "templates": {
                tid: {"file": str(t.path), "sha256": t.sha256}
                for tid, t in self.templates.items()
            },
            "variants": [
                {"id": v.id, "template": v.template.id, "n_examples": v.n_examples, "prompt_version": v.template.version}
                for v in self.variants
            ],
            "pfe_tokens": {f"{v}|{m}": n for (v, m), n in self.pfe.items()},
            "tokenizers": {m: t.name for m, t in self.tokenizers.items()},
            "models": [m.public_dict() for m in self.models],
            "model_versions": versions,
            "model_details": self._model_details(),
            "config": cfg.raw,
            "code_version": code_version(),
            "runs": previous.get("runs", []) + [{"start": started, "end": now_iso(), "counts": dict(getattr(self, "_counts", {})), "callers": {k: c.stats for k, c in self.callers.items()}}],
            "status_counts": dict(status_counts),
        }
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _describe(caller) -> dict:
    try:
        return caller.provider.describe()
    except Exception as exc:
        return {"error": str(exc)}


def _fmt(x: float | None) -> str:
    return "" if x is None else repr(x)

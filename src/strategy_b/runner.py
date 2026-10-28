"""Strategy B runner: model × variant × task × run_index, one agent run each (AGENT_SPEC.md §6, §10).

Writes results.csv (same contract as Strategy A, ``strategy = B``), runs.jsonl
(one AgentRun per line) and run_manifest.json. Finished runs are skipped on restart.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from llmcore.cache import DiskCache
from llmcore.caller import Caller
from llmcore.config import ModelConfig
from llmcore.providers import Provider
from llmcore.tokenizers import Tokenizer, get_tokenizer, reasoning_tokens
from llmcore.util import code_version, file_sha256, now_iso
from strategy_a.results import ResultsWriter, drop_statuses, load_done, read_results
from strategy_a.tasks import Task, load_tasks
from strategy_a.templates import pfe_text, task_effort_texts, text_sizes

from . import steps
from .config import StrategyBConfig, VariantSpec
from .graph import AgentContext, build_graph
from .search import CachedSearch, SearchTool, load_search_tool

STRATEGY = "B"


@dataclass
class Job:
    task: Task
    variant: VariantSpec
    model: ModelConfig
    run_index: int

    @property
    def key(self):
        return (self.task.task_id, self.model.id, self.variant.id, self.run_index)


class Runner:
    def __init__(
        self,
        cfg: StrategyBConfig,
        model_ids: list[str] | None = None,
        variant_ids: list[str] | None = None,
        limit: int | None = None,
        providers: dict[str, Provider] | None = None,
        search_tool: SearchTool | None = None,
        read_cache: bool = True,
        read_search_cache: bool = True,
        log: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.models = cfg.select_models(model_ids)
        self.variants = cfg.select_variants(variant_ids)
        self.out_dir = cfg.path(cfg.output_dir)
        self.log = log or (lambda m: print(m, file=sys.stderr))
        self._providers = providers or {}
        self._search_tool = search_tool
        self._limit = limit
        self.read_cache = read_cache
        self.read_search_cache = read_search_cache
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        cfg = self.cfg
        self.tasks = load_tasks(cfg.path(cfg.tasks))[: self._limit or None]
        self.task_set_version = self.tasks[0].task_set_version
        self.plan_tpls = {v.id: steps.load_plan_template(cfg.path(v.plan)) for v in self.variants}
        self.synth_tpls = {v.id: steps.load_synth_template(cfg.path(v.synthesize)) for v in self.variants}
        cache = DiskCache(cfg.cache_path)
        self.callers = {m.id: Caller(m, cache, provider=self._providers.get(m.id), read_cache=self.read_cache)
                        for m in self.models}
        self.tokenizers: dict[str, Tokenizer] = {
            m.id: get_tokenizer(m.tokenizer if m.tokenizer and m.tokenizer != "provider" else cfg.pfe_tokenizer)
            for m in self.models
        }
        tool = self._search_tool or load_search_tool(cfg.search)
        self.search = CachedSearch(tool, cfg.search_cache_path, cfg.max_search_retries, self.read_search_cache,
                                   max_credits=cfg.search_max_credits)
        # PFE (§10.2): the plan and synthesis templates as written, without placeholders.
        self.pfe_texts = {
            v.id: pfe_text(self.plan_tpls[v.id], "") + "\n" + pfe_text(self.synth_tpls[v.id], "") for v in self.variants
        }
        self.pfe = {
            (v.id, m.id): self.tokenizers[m.id].count(pfe_text(self.plan_tpls[v.id], ""))
            + self.tokenizers[m.id].count(pfe_text(self.synth_tpls[v.id], ""))
            for v in self.variants for m in self.models
        }
        self.graphs = {
            (v.id, m.id): build_graph(AgentContext(cfg, v, self.callers[m.id], self.search, self.plan_tpls[v.id],
                                                   self.synth_tpls[v.id], self.tokenizers[m.id].count))
            for v in self.variants for m in self.models
        }
        self._prepared = True

    def jobs(self) -> list[Job]:
        self.prepare()
        return [Job(t, v, m, r) for v in self.variants for t in self.tasks for m in self.models
                for r in range(1, self.cfg.runs_per_task + 1)]

    def pending(self) -> list[Job]:
        self.prepare()
        done = load_done(self.out_dir / "results.csv", self.task_set_version)
        return [j for j in self.jobs() if j.key not in done]

    def run(self, retry_statuses: set[str] | None = None) -> dict:
        self.prepare()
        if retry_statuses:
            n = drop_statuses(self.out_dir / "results.csv", self.out_dir / "runs.jsonl", retry_statuses)
            self.log(f"removed {n} runs with status in {sorted(retry_statuses)}; they will run again")
        pending = self.pending()
        self.log(f"{len(self.jobs())} runs planned, {len(pending)} to do")
        started = now_iso()
        writer = ResultsWriter(self.out_dir)
        self._counts: Counter = Counter()
        self._lock = threading.Lock()
        executors = {m.id: ThreadPoolExecutor(max_workers=m.max_concurrency) for m in self.models}
        # With reuse_retrieval, runs 2-5 (and variants with the same plan prompt) reuse the plan and searches
        # of the first one through the caches; a duplicate that starts while the first is in flight waits for it.
        pending.sort(key=lambda j: (j.run_index > 1, j.task.task_id, j.variant.id, j.run_index))
        try:
            futures = [executors[j.model.id].submit(self._process, j, writer) for j in pending]
            for f in futures:
                f.result()
        finally:
            for ex in executors.values():
                ex.shutdown(wait=True, cancel_futures=True)
            writer.close()
            for c in self.callers.values():
                try:
                    c.close()
                except Exception as exc:
                    self.log(f"closing {c.cfg.id}: {exc}")
            self.write_manifest(started)
        self.log(f"done: {dict(self._counts)}; search: {self.search.stats}")
        return dict(self._counts)

    def _process(self, job: Job, writer: ResultsWriter) -> None:
        graph = self.graphs[(job.variant.id, job.model.id)]
        try:
            state = graph.invoke({"task": job.task, "run_index": job.run_index, "calls": [], "events": []})
        except Exception as exc:  # a bug in one run must not stop the others
            state = {"status": "api_error", "events": [f"unexpected:{type(exc).__name__}: {exc}"], "calls": []}
        status = state.get("status") or "ok"
        predictions = state.get("predictions") or {h["prop_key"]: None for h in job.task.H}
        shared = self.cfg.reuse_retrieval and job.run_index > 1
        usage = self._usage(job, state, shared)
        row = {
            "task_set_version": job.task.task_set_version,
            "task_id": job.task.task_id,
            "strategy": STRATEGY,
            "variant": job.variant.id,
            "model": job.model.id,
            "run_index": job.run_index,
            "status": status,
            "predictions": json.dumps(predictions, ensure_ascii=False),
            "evidence": json.dumps(state.get("evidence_urls") or {}, ensure_ascii=False),
            "pfe_tokens": self.pfe[(job.variant.id, job.model.id)],
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "tokens_reasoning": "" if usage["tokens_reasoning"] is None else usage["tokens_reasoning"],
            "llm_calls": usage["llm_calls"],
            "search_calls": usage["search_calls"],
            "latency_s": _fmt(usage["latency_s"]),
            "energy_kwh": _fmt(usage["energy_kwh"]),
            "co2e_kg": _fmt(usage["co2e_kg"]),
            "cost_usd": _fmt(usage["cost_usd"]),
            "prompt_version": job.variant.version,
            "timestamp": now_iso(),
        }
        record = {
            "task_id": job.task.task_id,
            "model": job.model.id,
            "variant": job.variant.id,
            "run_index": job.run_index,
            "status": status,
            "retrieval_shared_with_run": 1 if shared else None,
            "queries": state.get("search_log") or [],
            "plan_parsed": state.get("plan_parsed"),
            "results": state.get("results") or [],
            "evidence_passed": state.get("evidence") or [],
            "calls": state.get("calls") or [],
            "predictions": predictions,
            "evidence_urls": state.get("evidence_urls") or {},
            "label_map": state.get("labels"),
            "validation_events": state.get("events") or [],
            "usage": usage,
            "effort": self._effort(job),
        }
        writer.write(row, record)
        with self._lock:
            self._counts[status] += 1
            n = sum(self._counts.values())
            if n % 50 == 0:
                self.log(f"{n} runs written: {dict(self._counts)}")

    def _usage(self, job: Job, state: dict, shared: bool) -> dict:
        """Totals for the run. With reuse_retrieval, the plan and the searches belong to run 1 only."""
        tok = self.tokenizers[job.model.id]
        tokens_in = tokens_out = 0
        latency, energy, co2 = 0.0, 0.0, 0.0
        missing_latency = missing_energy = False
        n_calls, reasoning, cached_in, costs = 0, [], [], []
        for call in state.get("calls") or []:
            if shared and call["step"].startswith("plan"):
                continue
            resp = call.get("response")
            if not resp:
                continue
            n_calls += 1
            req = call["request"]
            tin = resp.get("tokens_in")
            tin = tin if tin is not None else tok.count(
                "\n".join([req.get("system") or ""] + [m["content"] for m in req["messages"]]))
            tout = resp.get("tokens_out")
            tout = tout if tout is not None else tok.count(resp.get("text") or "")
            tokens_in += tin
            tokens_out += tout
            raw = resp.get("raw") or {}
            reasoning.append(reasoning_tokens(raw, tok))
            if raw.get("prompt_eval_cached_count") is not None:
                cached_in.append(raw["prompt_eval_cached_count"])
            pricing = job.model.pricing
            costs.append(pricing.cost(tin, tout, resp.get("execution") or "realtime") if pricing else None)
            if resp.get("latency_s") is None:
                missing_latency = True
            else:
                latency += resp["latency_s"]
            e = resp.get("energy") or {}
            if e.get("energy_kwh") is None:
                missing_energy = True
            else:
                energy += e["energy_kwh"]
                co2 += e.get("co2e_kg") or 0.0
        searches = [] if shared else (state.get("search_log") or [])
        search_latency = sum(q.get("latency_s") or 0.0 for q in searches)
        return {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "search_calls": len(searches),
            "search_credits": sum(q.get("credits") or 0.0 for q in searches),
            "search_cache_hits": sum(1 for q in searches if q.get("cached")),
            "evidence_tokens": sum(tok.count(steps.render_evidence([e])) for e in state.get("evidence") or []),
            "latency_s": None if missing_latency else latency + search_latency,
            "llm_latency_s": latency,
            "search_latency_s": search_latency,
            "energy_kwh": None if missing_energy else energy,  # LLM only; the search API's energy is unknown
            "co2e_kg": None if missing_energy else co2,
            "llm_calls": n_calls,
            # Part of tokens_out; None when the provider gives no reasoning count.
            "tokens_reasoning": sum(reasoning) if reasoning and None not in reasoning else None,
            "tokens_in_cached": sum(cached_in) if cached_in else None,
            "cost_usd": sum(costs) if costs and None not in costs else None,  # LLM only; search: search_credits
            "cost_usd_per_call": costs,
        }

    def _effort(self, job: Job) -> dict:
        """Sizes of what a person writes (METRICS.md §6). The agent's queries are not typed by anyone."""
        texts = task_effort_texts(job.task, False, False)
        graph = steps.render_graph_context(job.task, job.variant.with_context)
        texts["context"] = "\n".join(t for t in (texts["context"], graph) if t)
        return text_sizes({"template": self.pfe_texts[job.variant.id], **texts}, self.tokenizers[job.model.id])

    def write_manifest(self, started: str) -> None:
        cfg = self.cfg
        path = self.out_dir / "run_manifest.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        results = self.out_dir / "results.csv"
        status_counts = Counter(r["status"] for r in read_results(results)) if results.exists() else Counter()

        def describe(c):
            try:
                return c.provider.describe()
            except Exception as exc:
                return {"error": str(exc)}

        manifest = {
            "strategy": STRATEGY,
            "task_set_version": self.task_set_version,
            "tasks_file": {"path": str(cfg.tasks), "sha256": file_sha256(cfg.path(cfg.tasks))},
            "variants": [{"id": v.id, "plan": v.plan, "synthesize": v.synthesize, "synthesis": v.synthesis,
                          "with_context": v.with_context, "version": v.version,
                          "plan_sha256": self.plan_tpls[v.id].sha256, "synth_sha256": self.synth_tpls[v.id].sha256}
                         for v in self.variants],
            "search": {k: v for k, v in cfg.search.items() if k != "api_key_env"},
            "blocked_domains": cfg.blocked_domains,
            "reuse_retrieval": cfg.reuse_retrieval,
            "pfe_tokens": {f"{v}|{m}": n for (v, m), n in self.pfe.items()},
            "models": [m.public_dict() for m in self.models],
            "model_details": {m: describe(c) for m, c in self.callers.items()},
            "config": cfg.raw,
            "code_version": code_version(),
            "runs": previous.get("runs", []) + [{"start": started, "end": now_iso(), "counts": dict(self._counts),
                                                 "search": dict(self.search.stats),
                                                 "callers": {k: c.stats for k, c in self.callers.items()}}],
            "status_counts": dict(status_counts),
        }
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _fmt(x):
    return "" if x is None else repr(x)

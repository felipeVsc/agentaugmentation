"""One agent run as a LangGraph graph (AGENT_SPEC.md §5).

    START → plan ──(usable plan)──→ retrieve ──(some results)──→ synthesize → END
              └──(plan_invalid / LLM error)──→ END      └──(retrieval_failed)──→ END

Single-turn by construction: no edge goes back, so each run makes at most one
planning call, one parallel search step and one synthesis call (plus at most one
repair call per LLM step, §8).
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from llmcore.caller import Caller
from llmcore.types import CallFailed, ProviderError, Request, Response
from strategy_a.tasks import Task
from strategy_a.templates import Template

from . import steps
from .config import StrategyBConfig, VariantSpec
from .search import CachedSearch


class AgentState(TypedDict, total=False):
    task: Task
    run_index: int
    labels: dict[str, str]
    queries: list[dict]
    search_log: list[dict]
    results: list[dict]
    evidence: list[dict]
    predictions: dict[str, Any]
    evidence_urls: dict[str, list[str]]
    status: str
    plan_parsed: list[dict]
    calls: Annotated[list[dict], operator.add]  # every LLM call, in order: {step, request, response}
    events: Annotated[list[str], operator.add]  # validation events (§8)


@dataclass
class AgentContext:
    cfg: StrategyBConfig
    variant: VariantSpec
    caller: Caller
    search: CachedSearch
    plan_tpl: Template
    synth_tpl: Template
    count_tokens: Callable[[str], int]


def _call(ctx: AgentContext, step: str, request: Request) -> tuple[Response | None, dict, str | None]:
    """Returns (response, log entry, failure status)."""
    try:
        resp = ctx.caller.call(request)
    except CallFailed as exc:
        return None, {"step": step, "request": request.as_dict(), "error": str(exc)}, exc.status
    return resp, {"step": step, "request": request.as_dict(), "response": resp.as_dict(), "cached": resp.cached}, None


def build_graph(ctx: AgentContext):
    cfg, variant = ctx.cfg, ctx.variant
    params = ctx.caller.cfg.params

    def plan(state: AgentState) -> dict:
        task = state["task"]
        system, user, labels = steps.render_plan(ctx.plan_tpl, task, variant.with_context, cfg.max_subqueries)
        # reuse_retrieval: every run uses the plan of run 1 (same request -> cached response -> same queries).
        run_index = 1 if cfg.reuse_retrieval else state["run_index"]
        request = Request(system, [{"role": "user", "content": user}], params, run_index)
        resp, log, failed = _call(ctx, "plan", request)
        if failed:
            return {"labels": labels, "calls": [log], "status": failed, "events": ["plan_call_failed"]}
        calls = [log]
        queries, events, ok = steps.parse_plan(resp.text, task, labels, cfg.max_subqueries)
        if not ok:
            repair = Request(system, request.messages + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": steps.PLAN_REPAIR.format(max_queries=cfg.max_subqueries, targets=", ".join(labels))},
            ], params, run_index)
            resp2, log2, failed = _call(ctx, "plan_repair", repair)
            calls.append(log2)
            if not failed:
                queries, ev2, ok = steps.parse_plan(resp2.text, task, labels, cfg.max_subqueries)
                events += [f"repair:{e}" for e in ev2]
        if not ok:
            return {"labels": labels, "calls": calls, "events": events, "status": "plan_invalid"}
        return {"labels": labels, "calls": calls, "events": events, "queries": queries, "plan_parsed": queries}

    def retrieve(state: AgentState) -> dict:
        queries = state["queries"]

        def one(q):
            try:
                resp = ctx.search.search(q["query"], cfg.results_per_query, cfg.blocked_domains)
                return q, resp, None
            except ProviderError as exc:
                return q, None, f"{exc.kind}: {exc}"

        with ThreadPoolExecutor(max_workers=min(cfg.max_parallel_queries, len(queries))) as pool:
            done = list(pool.map(one, queries))
        search_log, results, events = [], {}, []
        for q, resp, err in done:
            search_log.append({**q, "error": err, "n_results": len(resp.results) if resp else 0,
                               "latency_s": resp.latency_s if resp else None, "cached": resp.cached if resp else False,
                               "credits": resp.credits if resp else 0.0})
            if err:
                events.append(f"search_failed:{q['id']}")
            else:
                results[q["id"]] = resp.results
        if not results:
            return {"search_log": search_log, "results": [], "evidence": [], "events": events, "status": "retrieval_failed"}
        evidence, logged, ev = steps.build_evidence(results, cfg.blocked_domains, cfg.max_evidence_tokens,
                                                    ctx.count_tokens, cfg.max_chars_per_result)
        return {"search_log": search_log, "results": logged, "evidence": evidence, "events": events + ev}

    def synthesize(state: AgentState) -> dict:
        task, labels, evidence = state["task"], state["labels"], state["evidence"]
        system, user = steps.render_synthesis(ctx.synth_tpl, task, variant.with_context, evidence)
        request = Request(system, [{"role": "user", "content": user}], params, state["run_index"])
        resp, log, failed = _call(ctx, "synthesize", request)
        if failed:
            return {"calls": [log], "status": failed, "events": ["synthesis_call_failed"]}
        calls = [log]
        preds, cites, events, ok = steps.parse_synthesis(resp.text, task, labels, evidence, variant.synthesis)
        last = resp
        if not ok:
            repair = Request(system, request.messages + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": steps.SYNTH_REPAIR.replace("{output_format}", steps.synth_output_format(labels))},
            ], params, state["run_index"])
            resp2, log2, failed = _call(ctx, "synthesize_repair", repair)
            calls.append(log2)
            if not failed:
                preds, cites, ev2, ok = steps.parse_synthesis(resp2.text, task, labels, evidence, variant.synthesis)
                events += [f"repair:{e}" for e in ev2]
                last = resp2
        if not ok:
            return {"calls": calls, "events": events, "status": "synthesis_invalid"}
        status = "truncated" if last.finish_reason == "length" else "ok"
        return {"calls": calls, "events": events, "predictions": preds, "evidence_urls": cites, "status": status}

    def after(next_node: str):
        return lambda state: END if state.get("status") else next_node

    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after("retrieve"), ["retrieve", END])
    g.add_conditional_edges("retrieve", after("synthesize"), ["synthesize", END])
    g.add_edge("synthesize", END)
    return g.compile()

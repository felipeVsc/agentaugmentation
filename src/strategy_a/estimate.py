"""Cost estimate without calling any model.

Input tokens are counted on the rendered prompts with each model's tokenizer
(the PFE tokenizer when the model has none, or when it is ``provider``).
Output tokens use ``expected_output_tokens`` of each template. Prices come
from ``pricing`` in each model entry.
"""

from __future__ import annotations

from collections import defaultdict

from llmcore.tokenizers import get_tokenizer

from .runner import Runner


def estimate(runner: Runner, pending_only: bool = False) -> list[dict]:
    runner.prepare()
    jobs = runner.pending() if pending_only else runner.jobs()
    tok = {}
    for m in runner.models:
        spec = m.tokenizer if m.tokenizer and m.tokenizer != "provider" else runner.cfg.pfe_tokenizer
        tok[m.id] = get_tokenizer(spec)
    prompt_tokens: dict[tuple[str, str, str], int] = {}
    agg = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0})
    for j in jobs:
        k = (j.model.id, j.variant.id, j.task.task_id)
        if k not in prompt_tokens:
            prompt_tokens[k] = tok[j.model.id].count((j.rendered.system or "") + "\n" + j.rendered.user)
        a = agg[(j.model.id, j.variant.id)]
        a["calls"] += 1
        a["tokens_in"] += prompt_tokens[k]
        a["tokens_out"] += j.variant.template.expected_output_tokens
    rows = []
    for m in runner.models:
        for v in runner.variants:
            a = agg.get((m.id, v.id))
            if not a:
                continue
            p = m.pricing
            realtime = (
                a["tokens_in"] / 1e6 * p.input_per_mtok + a["tokens_out"] / 1e6 * p.output_per_mtok if p else None
            )
            rows.append(
                {
                    "model": m.id,
                    "variant": v.id,
                    **a,
                    "cost_realtime_usd": realtime,
                    "cost_batch_usd": realtime * (1 - p.batch_discount) if p else None,
                }
            )
    return rows


def print_estimate(rows: list[dict]) -> None:
    def money(x):
        return "      n/a" if x is None else f"{x:9.2f}"

    print(f"{'model':28s} {'variant':22s} {'calls':>7s} {'tok_in':>12s} {'tok_out':>10s} {'$ realtime':>10s} {'$ batch':>9s}")
    totals = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    for r in rows:
        print(
            f"{r['model']:28s} {r['variant']:22s} {r['calls']:7d} {r['tokens_in']:12,d} {r['tokens_out']:10,d} "
            f"{money(r['cost_realtime_usd']):>10s} {money(r['cost_batch_usd'])}"
        )
        t = totals[r["model"]]
        t[0] += r["calls"]
        t[1] += r["tokens_in"]
        t[2] += r["tokens_out"]
        t[3] += r["cost_realtime_usd"] or 0
        t[4] += r["cost_batch_usd"] or 0
    print("-" * 104)
    grand = [0, 0, 0, 0.0, 0.0]
    for model, t in totals.items():
        print(f"{model:28s} {'TOTAL':22s} {t[0]:7d} {t[1]:12,d} {t[2]:10,d} {money(t[3]):>10s} {money(t[4])}")
        grand = [g + x for g, x in zip(grand, t)]
    print(f"{'ALL MODELS':28s} {'':22s} {grand[0]:7d} {grand[1]:12,d} {grand[2]:10,d} {money(grand[3]):>10s} {money(grand[4])}")
    print("Input tokens are counted on the rendered prompts; output tokens are expected_output_tokens per template.")

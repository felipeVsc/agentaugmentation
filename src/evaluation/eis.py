"""Environmental impact score (EIS) of a run: the agent energy formula of the draft (METRICS.md §4.2).

    E_agent = (N_calls + N_tools · w_tool) · E_single

``E_single`` is the mean estimated energy of one LLM call of the run (EcoLogits or
CodeCarbon), so ``N_calls · E_single`` is the run's measured LLM energy. Strategy A has
``N_tools = 0`` and its EIS is its LLM energy. ``w_tool`` has no default: it must be
chosen for the paper.
"""

from __future__ import annotations


def agent_energy(energy_kwh: float | None, llm_calls: int, tool_calls: int, w_tool: float) -> float | None:
    """EIS in kWh for one run; None when the run's LLM energy is unknown or it made no LLM call."""
    if w_tool is None:
        raise ValueError("w_tool is not set: choose the tool weight before computing the EIS")
    if energy_kwh is None or not llm_calls:
        return None
    e_single = energy_kwh / llm_calls
    return (llm_calls + tool_calls * w_tool) * e_single


def run_eis(row: dict, w_tool: float, count_cached_searches: bool = True) -> float | None:
    """EIS of one ``results.csv`` row (strings as read by csv).

    ``count_cached_searches``: a search answered by the search buffer costs nothing in this
    run, but a reader reproducing the run without the buffer pays for it. True gives that
    reproducible figure; False needs ``search_cache_hits`` from ``runs.jsonl`` and is the
    energy actually spent.
    """
    def num(key):
        value = row.get(key)
        return None if value in (None, "") else float(value)

    calls = int(num("llm_calls") or 0)
    tools = int(num("search_calls") or 0)
    if not count_cached_searches:
        tools -= int(num("search_cache_hits") or 0)
    return agent_energy(num("energy_kwh"), calls, max(tools, 0), w_tool)

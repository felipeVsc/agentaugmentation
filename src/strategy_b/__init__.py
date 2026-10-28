"""Strategy B — Single-Turn Agentic Pathway (AGENT_SPEC.md, Algorithm 7 of the draft).

One try per run: the model plans web searches (1 LLM call), the queries run in
parallel (1 retrieval step), and the model answers from the retrieved evidence
(1 LLM call). No loop, no reflection. LangGraph orchestrates the three steps;
the LLM calls go through ``llmcore`` exactly as in Strategy A (cache, retries,
seeds, usage, energy), so both strategies are comparable.
"""

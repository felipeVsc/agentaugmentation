"""Strategy A — Static Parametric Pathway (LLM_SPEC.md, Algorithm 2 of the draft).

The model answers from its own knowledge, with no tools. Each task is run
``runs_per_task`` times per model and variant; every run is one row of results.csv.
"""

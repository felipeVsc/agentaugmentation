# Agent augmentation benchmark

Pipeline for the paper *Data Augmentation in Property Graphs using agentic AI*. Specs are in `docs/` (start with `docs/SPEC.md`).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/python -m pytest          # offline tests, no Neo4j needed
```

## Task definition (`docs/TASKDEFINITION_SPEC.md`)

1. Copy `.env.example` to `.env` and set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (and `NEO4J_DATABASE` if not the default).
2. Adapt `configs/airports.yaml`: `dataset.node_type`, properties to exclude, leakage pairs.
3. Run:

```bash
venv/bin/tasks check-connection --config configs/airports.yaml   # lists labels and relationship types
venv/bin/tasks build --config configs/airports.yaml              # snapshot + tasks
```

Outputs go to `output_dir` (e.g. `tasks/airports/v1/`): `tasks.csv`, `ground_truth.csv`, `manifest.json`, and `intermediate/` (snapshot, centrality, popularity, context). A task set is never overwritten; use a new `output_dir` for a new version. `tasks snapshot` exports only the snapshot, and the next `build` reuses it.

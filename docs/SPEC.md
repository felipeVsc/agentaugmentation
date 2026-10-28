# SPEC: Agentic AI Data Augmentation for Property Graphs

> Top-level implementation spec for the paper *"Data Augmentation in Property Graphs using agentic AI: measuring human effort, task difficulty, environmental footprint and accuracy"* (Vasconcelos, Aguiar, Chanson, Halfeld-Ferrari, Marcel, Peralta — draft, `docs/paper2.pdf`).
>
> This file describes the research design, the shared conventions, and how the components connect. Each component has its own spec, which is the single source of truth for its details:
>
> | Component | Spec | Paper |
> |---|---|---|
> | Task definition | `TASKDEFINITION_SPEC.md` | Phase 1 of Algorithm 1, Algorithm 2 (October 2026 draft; Algorithm 4 before) |
> | Strategy A (LLM) | `LLM_SPEC.md` | Algorithm 2 |
> | Strategy B (search agent) | `AGENT_SPEC.md` | Algorithm 7 |
> | Evaluation | `EVALUATION_SPEC.md` | Scoring in Algorithms 1 and 5 |
> | Stored metrics (all components) | `METRICS.md` | §3.2 difficulty, §3.4 effort and agent energy (October 2026 draft) |
>
> If this file and a component spec disagree, the component spec wins and this file must be updated.
>
> Where the draft is ambiguous, the v1 choice is marked **[ASSUMPTION]** and listed in the open questions (§11 here for cross-cutting questions, and in each component spec).

---

## 1. Problem

A property graph has nodes whose properties are missing values. The user wants to fill them with **real values found in the world**, not with predicted or synthetic ones. Today this means searching the web by hand for each missing value.

Example: in an airport graph, the node for GRU has no value for `runways`. The user has to search the web to find how many runways GRU has.

The project asks whether LLMs and LLM-driven search agents can do this task, how well, at what cost, and how that depends on the difficulty of the task. Two strategies are compared:

- **Strategy A — Static Parametric Pathway:** the LLM answers from its own knowledge, given the graph context in the prompt. No tools.
- **Strategy B — Single-Turn Agentic Pathway:** the same LLM plans web search queries, a search tool (Tavily) runs them in one step, and the LLM writes the answer from the retrieved evidence, citing its sources.

Both are evaluated on the same tasks along four dimensions: **accuracy**, **human effort**, **sustainability** (environmental footprint), and **task difficulty**.

## 2. Research questions

1. **Accuracy.** How accurately does each strategy recover hidden property values, and how often does it abstain or answer wrongly?
2. **Search vs. memory.** Does access to web search (B) reduce abstention compared with parametric knowledge only (A), without increasing wrong answers?
3. **Difficulty.** Does the difficulty metric of the paper (node centrality, property popularity, sizes of H and G) predict how hard a task is for each strategy?
4. **Effort.** How much human effort does each strategy require (currently: prompt formulation effort, §7.2)?
5. **Sustainability.** What is the energy and token cost of each strategy, and how much accuracy does each unit of cost buy?

## 3. Goals

1. A reproducible pipeline that, given a property graph, a node type, and a set of models, produces result tables comparing Strategy A and Strategy B.
2. A **fixed task set** per dataset, generated once and shared by every strategy and model, so all of them are evaluated on exactly the same tasks.
3. Tasks stratified by node centrality, property popularity, and the sizes of H and G, each with its difficulty score.
4. At least three property graph datasets.
5. Full traceability: raw prompts, raw outputs, search queries and results, token counts, latency, and energy are stored, so every metric can be recomputed without calling a model or the search tool again.
6. As much information as is useful for the paper at each step: which tasks, tokens, cost, latency, energy, failures.

## 4. Out of scope (v1)

- Predicting missing edges (link prediction). Only node property values.
- Writing values back into the source database. The output is an evaluation, not an updated graph.
- Multi-turn agents (ReAct-style loops). Strategy B is single-turn by design.
- Fine-tuning models.
- A user interface. CLI and config files only.
- Domain-level difficulty (raised in the draft; not measured in v1).

## 5. Glossary

| Term | Meaning |
|---|---|
| `PG` | Property graph: nodes and edges, both carrying properties. |
| `t` | Node type under evaluation (e.g. `Airport`). |
| `N` | Set of nodes of type `t`. |
| element `e` | A node or an edge. Properties come from elements. |
| `cent(e)` | Normalized centrality of element `e`, in `(0, 1]`. |
| `context(n)` | Properties of `n`, its direct neighbors, the edges to them, and elements on `n:1` paths from `n` (e.g. Airport → City → Country). |
| `pop(p)` | Popularity of property `p`: fraction of elements of its type with a non-null value for `p`. |
| `cent_p(e)` | Centrality of the element that property `p` comes from. |
| `prop_key` | Identity of a property by its path from `n`, e.g. `Airport.elevation`, `Airport-[IN]->City.name`. |
| task | `(n, H, G)`: node `n`, set `H` of properties of `n` whose values are hidden, set `G` of `(property, value)` pairs given to the model. |
| variant | One distinct configuration of a strategy: a template id for A (e.g. `base_few_shot_3`), an agent config id for B (e.g. `waves1`). |
| configuration | `(task_id, model, strategy, variant)`. Has exactly 5 runs. |
| run | One execution of a configuration, identified by `run_index` 1..5. |
| representative | The answer chosen from the 5 runs of a configuration by the evaluation. |
| abstention | A `null` prediction: the strategy declines to answer. Not an error. |

## 6. Architecture

### 6.1 Data flow

```
                     ┌─────────────────────────┐
  graph (Neo4j) ───► │ Task definition         │ ──► tasks.csv ─────────────┬──────────────┐
                     │ TASKDEFINITION_SPEC.md  │ ──► ground_truth.csv ──┐   │              │
                     └─────────────────────────┘     manifest.json      │   ▼              ▼
                                                                         │ ┌──────────────┐ ┌──────────────┐
                                                                         │ │ Strategy A   │ │ Strategy B   │
                                                                         │ │ LLM_SPEC.md  │ │ AGENT_SPEC.md│
                                                                         │ └──────┬───────┘ └──────┬───────┘
                                                                         │        │ results.csv    │ results.csv
                                                                         │        │ runs.jsonl     │ runs.jsonl
                                                                         ▼        ▼                ▼
                                                                     ┌──────────────────────────────────┐
                                                                     │ Evaluation  EVALUATION_SPEC.md   │
                                                                     └──────────────────────────────────┘
                                                                        task_scores.csv, summary.csv,
                                                                        comparison.csv, ...
```

Each component is a separate command that reads files and writes files. Any component can be re-run without re-running the ones before it; in particular, re-scoring never calls a model.

### 6.2 File contracts

| File | Written by | Read by | Defined in |
|---|---|---|---|
| `tasks.csv` | Task definition | Strategy A, Strategy B, Evaluation | `TASKDEFINITION_SPEC.md` §7.1 |
| `ground_truth.csv` | Task definition | **Evaluation only** | `TASKDEFINITION_SPEC.md` §7.2 |
| `manifest.json` (task set) | Task definition | Evaluation, paper | `TASKDEFINITION_SPEC.md` §7.3 |
| `results.csv` | Strategy A, Strategy B | Evaluation | `EVALUATION_SPEC.md` §3.1 |
| `runs.jsonl` | Strategy A, Strategy B | Evaluation (Strategy B extras), audits | `LLM_SPEC.md` §8.2, `AGENT_SPEC.md` §10 |
| `run_manifest.json` | Strategy A, Strategy B | Audits | `LLM_SPEC.md` §8.3 |
| Evaluation outputs | Evaluation | Paper | `EVALUATION_SPEC.md` §7 |

### 6.3 Directory layout

```
configs/                                   one config per component and dataset
tasks/<dataset>/<task_set_version>/        tasks.csv, ground_truth.csv, manifest.json, intermediate/
results/<dataset>/<task_set_version>/strategy_a/   results.csv, runs.jsonl, run_manifest.json
results/<dataset>/<task_set_version>/strategy_b/   results.csv, runs.jsonl, run_manifest.json
eval/<dataset>/<task_set_version>/<eval_id>/       evaluation outputs
templates/, prompts/, examples/            versioned prompt files and few-shot examples
```

### 6.4 Commands

| Command | Component |
|---|---|
| `tasks build --config configs/<dataset>.yaml` | Task definition |
| `strategy-a run --config configs/strategy_a_<dataset>.yaml` | Strategy A |
| `strategy-b run --config configs/strategy_b_<dataset>.yaml` | Strategy B |
| `eval run --config configs/eval_<dataset>.yaml` | Evaluation |

## 7. Experimental protocol

Mirrors **Algorithm 1 (Unified Evaluation Protocol)** of the paper, split into the components above.

### 7.1 Phase 1 — Task definition (`TASKDEFINITION_SPEC.md`)
1. Snapshot the graph from Neo4j, so the task set does not depend on later changes to the database.
2. Compute centrality (PageRank, normalized per element type to `(0, 1]`), context, and popularity.
3. Sample tasks with three independent stratifications:
   - nodes by `cent(n)`;
   - H properties by `pop(p)` only (all H properties belong to `n`, so adding `cent(n)` would repeat the node factor);
   - G properties by `pop(p) × cent_p(e)`.
4. Compute the difficulty of every task and store its three factors separately:
   ```
   Difficulty(T) = ∛( |H| / (|H|+|G|) × −log₂(cent(n)) × Σ_{p ∈ H ∪ G} −log₂(pop(p) × cent_p(e)) )
   ```
5. Write `tasks.csv` (no hidden values), `ground_truth.csv` (hidden values, separate file), and the manifest. A task set is immutable once written; any change creates a new `task_set_version`.

### 7.2 Phase 2 — Strategies (`LLM_SPEC.md`, `AGENT_SPEC.md`)
For every model `m ∈ M`, every variant, and every task, each strategy runs 5 times and writes one `results.csv` row per run.

| | Strategy A | Strategy B |
|---|---|---|
| Model calls per run | 1 (plus at most one repair call) | Plan + synthesis (plus one fill call when `waves: 2`, plus repair calls) |
| Tools | None | Web search (Tavily), one retrieval step per wave |
| Variants | 6 templates (`{base, compact} × {zero-shot, few-shot, CoT}`); few-shot with 1, 3 and 5 examples | Agent configs (e.g. `waves1`, `waves2`) |
| Answer source | Model's own knowledge | Retrieved evidence only; every non-null value must cite a source |
| Leakage control | Few-shot example nodes must not appear in the task set | Domain blocklist for the source dataset, in the request and as a local filter |

Rules shared by both strategies:
- Same inputs: the same row of `tasks.csv`, the same model, the same generation parameters.
- **Neither strategy reads `ground_truth.csv`.**
- 5 runs per configuration, temperature above 0 (default 0.7), so the runs measure variability.
- No aggregation: every run is written; selecting a representative is done by the evaluation.
- Failures are recorded as rows with a failure `status` and all-null predictions. Nothing is silently dropped.
- Calls are cached with `run_index` in the key; interrupted runs resume without duplicating rows.

### 7.3 Phase 3 — Evaluation (`EVALUATION_SPEC.md`)
1. Validate inputs: same `task_set_version`, exactly 5 runs per configuration, exactly the H keys.
2. Normalize predictions and ground truth by value type (strings, numbers with units, booleans, dates).
3. Select a representative run per configuration without using the ground truth. Default: the medoid run (most consensual); alternative: per-property majority vote (3 of 5).
4. Score each property as correct, wrong, or abstained, in strict and tolerant matching.
5. Aggregate by model, strategy, variant, strata, sizes, difficulty bins, and property; compare A and B on paired tasks.

## 8. Metrics

Each metric is defined in one place; this table only says where.

| Dimension | Metrics | Defined in |
|---|---|---|
| Accuracy | Precision, recall, F1 (micro and macro), exact match, abstention rate, error rate; strict and tolerant | `EVALUATION_SPEC.md` §5.5, §5.6, §6.1 |
| Stability | Agreement rate across the 5 runs, distinct answers per property, gap to the oracle best-of-5 | `EVALUATION_SPEC.md` §5.4, §6.3 |
| Effort | Prompt formulation effort (`pfe_tokens`), prompt tokens actually sent | `LLM_SPEC.md` §5.6, `AGENT_SPEC.md` §10.2, `EVALUATION_SPEC.md` §6.2 |
| Sustainability | Tokens in/out, search calls, latency, energy and CO2e over the 5 runs; efficiency ratios (correct properties per 1k tokens, per kWh, per search call); composite score (formula pending) | `EVALUATION_SPEC.md` §6.4 |
| Difficulty | Difficulty score and its three factors; Spearman correlation with F1 | `TASKDEFINITION_SPEC.md` §6, `EVALUATION_SPEC.md` §6.7 |
| Strategy B only | Citation rate, retrieval coverage, grounded accuracy | `AGENT_SPEC.md` §10.2, `EVALUATION_SPEC.md` §6.6 |

**Energy.** Local models are measured with CodeCarbon. API models are estimated with EcoLogits, using the closest model of the same provider when the exact version is not available; these values are flagged as estimates. Measured and estimated values come from different methods and must be reported as such.

**Effort.** The title of the paper refers to human effort. v1 measures only the effort of writing the prompts (PFE). A measure of the effort a human would need to do the task itself is not yet specified (§11).

## 9. Data sources

- **[ASSUMPTION]** All datasets are loaded from Neo4j. Connection settings (URI, user, database) are read from the dataset config; credentials come from environment variables and are never written to configs, manifests, or logs.
- Each dataset has its own config with: node type, `context_max_hops`, properties excluded or never hidden, `leakage_pairs`, `blocked_domains` for Strategy B, and value-normalization rules (tolerances, aliases) for the evaluation.
- Dataset-specific rules are fixed **before** results are inspected, and recorded in the manifests.
- Adding a dataset requires only a config file and, if needed, its alias and normalization files.

## 10. Non-functional requirements

- **Reproducibility:** one master seed for task sampling; template, prompt, and model versions, file hashes, and the code version are recorded in every manifest.
- **Determinism:** with the same inputs, seed, cached model responses, and recorded search results, every component produces identical outputs.
- **Caching:** model calls and search calls are cached on disk; `--no-cache` forces fresh calls. Cached responses keep their original usage, latency, and energy values.
- **Resumability:** each runner skips `(task_id, model, variant, run_index)` combinations already written.
- **Rate limits and failures:** per-provider concurrency limits, exponential backoff, and a recorded failure status after the last retry.
- **Isolation of hidden values:** `ground_truth.csv` is read only by the evaluation, and tests assert that the strategies never open it.
- **Testability:** each component runs offline in CI with mocks or recorded responses (see each spec's testing section).

## 11. Open questions (cross-cutting)

Questions that affect more than one component. Component-specific questions are listed in each spec.

1. **Node identity.** If G does not identify the node (name, code), neither strategy can tell which entity it is about. Should identifying properties always be given, outside G and outside the difficulty? The answer must be the same for A and B. (`TASKDEFINITION_SPEC.md` §11.1, `LLM_SPEC.md` §11.5)
2. **Human effort.** How to measure the effort a human would need to do the task itself, as opposed to the effort of writing the prompt? Not specified in v1.
3. **PFE definition and tokenizer.** `LLM_SPEC.md` §5.6 counts tokens with each model's tokenizer, but its config sets `tokenizer_for_pfe: cl100k_base`. Which prompts count for the agent? Do few-shot examples and the system prompt count? (`LLM_SPEC.md` §11.4, `AGENT_SPEC.md` §12.5)
4. **Search tool.** `AGENT_SPEC.md` §5.2 allows a model's native search tool instead of Tavily. Using different search tools for different models mixes search-engine differences into model differences.
5. **Evidence-only synthesis.** Strategy B may not use its own knowledge. This makes B a test of retrieval and reading rather than of "LLM plus tools". (`AGENT_SPEC.md` §12.2)
6. **Choice of the A variant for the A vs. B comparison.** Choosing the best A variant by macro F1 on the same tasks favors A. Options: each variant, a fixed variant, or a variant chosen on a held-out split. (`EVALUATION_SPEC.md` §11.5)
7. **Representative selection:** medoid (v1) or per-property majority vote? (`EVALUATION_SPEC.md` §11.1)
8. **Sustainability composite formula**, and how to account for the search API's energy. (`EVALUATION_SPEC.md` §11.7, `AGENT_SPEC.md` §12.7)
9. **Reasoning models:** whether hidden reasoning tokens count in `tokens_out` and sustainability, and how the 5-run design applies when temperature cannot be set. (`LLM_SPEC.md` §11.6)
10. **Datasets:** which three graphs, and whether all are in Neo4j.
11. **Models in M:** which models, local vs. API, and with which parameters.
12. **Draft inconsistencies** carried over:
    - Algorithm 5 scores outside the task loop; treated as a typo, scoring is per task.
    - Algorithm 2 says zero-shot, while the paper lists zero-shot, few-shot, and CoT; all are supported.
    - Algorithm 6 returns "the best results and the template used", while Algorithm 2 returns one result per template; all variants are stored, and the comparison is defined in the evaluation.
    - Algorithm 4 (previous draft) used `cent_p(n)` on line 15 and `cent_p(e)` on line 17; the October 2026 draft settles it on the origin element's centrality, as implemented (`TASKDEFINITION_SPEC.md` §11.3).
    - The difficulty sum term grows with `|H| + |G|`, so a larger G makes a task harder (`TASKDEFINITION_SPEC.md` §6).

## 12. Milestones

1. Graph snapshot, centrality, context, popularity, with unit tests on a toy graph.
2. Task generation and difficulty, reproducing Table 1 of the draft; first airports task set.
3. Provider abstraction, caching, raw run logging, energy measurement.
4. Strategy A with all variants.
5. Evaluation: validation, normalization, selection, scoring, aggregation.
6. Strategy B with Tavily, leakage blocklist, and citation checks.
7. Full run on the airports dataset; result tables and plots (F1 vs. difficulty per strategy and model).
8. Second and third datasets.

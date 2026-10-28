# Spec: Agentic AI Data Augmentation for Property Graphs

Implementation spec for the paper "Data Augmentation in Property Graphs using agentic AI: measuring human effort, task difficulty, environmental footprint and accuracy" (Vasconcelos, Aguiar, Chanson, Halfeld-Ferrari, Marcel, Peralta — draft).



## Problem

A user has a graph database (a property graph) and he has some node properties/attributes that do not have values. So he needs to complete these attributes, by using data augmentation techniques. He wants real data, so he does not want to predict new data. Thus, he must search the web for the values that are missing on the nodes. 

For instance, take an Airport graph database, where nodes are airports that contains properties such as name, location and number of runways. The GRU airport node is missing the runways value, so the user must search the web to find the number of runways for GRU. 

This project tries to use AI agents to do web search (deep search) to find these missing values, testing if AI agents are good for data augmentation on property graphs. The idea is to understand if agents can search these informations better and faster than humans. 

We want to measure how well LLMs can fill in missing node attributes, comparing two strategies:

Strategy A — Static Parametric Pathway: the LLM answers from its internal (parametric) knowledge, guided by graph context in the prompt.
Strategy B — Single-Turn Agentic Pathway: an LLM-driven search agent that plans queries, calls an external web search tool (e.g. Tavily) in a single turn, and synthesizes the evidence into attribute values.

Both are evaluated on the same benchmark tasks along four dimensions: accuracy, prompt formulation effort, sustainability (environmental footprint), and task difficulty.

This is a research paper that is trying to validate this methodology. So we are going to take scientific methodology steps. The paper is the file /docs/paper2.pdf. The paper is still a draft. Where the draft is ambiguous or inconsistent, this spec records the assumption used for implementation and lists the item under Open questions. Assumptions are marked [ASSUMPTION].


## 2. Goals

1. Build a reproducible pipeline that, given a property graph, a node type, and a set of models, produces a results table comparing Strategy A and Strategy B.
2. Generate as much information about the pipeline as you have. So, information on each step, regardings which tasks, how many tokens were spent, how much money did it cost, the latency it took, everything you find that may be useful for a research paper.
3. Generate benchmark tasks automatically, stratified by difficulty (node centrality × property popularity × sizes of H and G).
3. Compute the difficulty metric defined in the paper for every task.
4. Support at least three property graph datasets.
5. Make every run traceable: raw prompts, raw model outputs, tool calls, token counts, and energy measurements are stored, so metrics can be recomputed without re-querying models.


TODO: A leakage guard for the search agent, so it can't just look up the source dataset the graph came from.

TODO: trocar para que os dados sjeam tais, informar as conexoes com o banco e etc e falar para ele usar sempre as mesmas tasks em diferentes modelos e etc

## 3. Out of scope (v1)

- Predicting missing edges (link prediction). Only node attribute values.
- Writing completed values back into the source database. The output is an evaluation table, not an updated graph.
- Multi-turn / conversational agents. Strategy B is single-turn by design.
- Fine-tuning models.
- A user interface. CLI + config files only.
- Domain-level difficulty (raised as an open question in the draft; not measured in v1).

## 4. Glossary

| Term | Meaning |
|---|---|
| `PG` | Property graph (nodes and edges, both carrying properties). |
| `t` | Node type under evaluation (e.g. `Airport`). |
| `N` | Set of nodes of type `t`. |
| element `e` | A node or an edge. Properties come from elements. |
| `cent(e)` | Normalized centrality of element `e` (in `(0, 1]`). |
| `context(n)` | Properties of `n`, its direct neighbors, the edges to them, and properties along paths of `n:1` relationships starting at `n` (e.g. Airport → City → Country). Formal definition in ref. [1] of the paper. |
| `pop(p)` | Popularity of property `p`: probability that a randomly drawn element of the same type has a non-null value for `p`. |
| `cent_p(e)` | Centrality of the element `e` that property `p` comes from. |
| task | Tuple `(n, H, G)`: node `n`, set `H` of property names to predict (values hidden), set `G` of `(property, value)` pairs given to the model. |
| template | A prompt template (compact/base × zero-shot/few-shot/CoT). |
| `r` | Number of repeated runs per prompt: 5 for both strategies. |

## 5. Inputs and configuration

All runs are driven by a single config file (YAML). Required fields:

```yaml
dataset:
  name: airports            # identifier
  source: neo4j             # [ASSUMPTION] graphs are loaded from Neo4j; see §13
  node_type: Airport
  context_max_hops: ...     # bound for n:1 path traversal
centrality:
  measure: pagerank         # paper example; must be normalized to (0, 1]
tasks:
  max_n: 5                  # nodes drawn per centrality stratum
  max_H: 3                  # max |H|
  max_G: 3                  # max |G|
  seed: 42
runs:
  r: 5
models:                     # set M
  - id: ...
    provider: ...           # e.g. openai | anthropic | ollama | ...
    params: {temperature: ..., max_tokens: ...}
strategy_a:
  templates: [base_zero_shot, base_few_shot, base_cot,
              compact_zero_shot, compact_few_shot, compact_cot]
strategy_b:
  search_tool: tavily
  max_subqueries: ...
  results_per_query: ...
output_dir: runs/<timestamp>/
```

## 6. Pipeline overview

Mirrors **Algorithm 1 (Unified Evaluation Protocol)** of the paper.

```
Phase 1 — Structural analysis & task definition
  1. Load PG, select nodes N of type t
  2. Compute cent(e) for nodes and edges
  3. Compute context(n) for each n ∈ N
  4. Compute pop(p) for every property in contexts
  5. TASKS ← taskDefinition(N)            (Algorithm 4)
  6. Compute Difficulty(T) for every task

Phase 2 — Dual-pathway execution & scoring
  for each model m ∈ M:
    for each task (n, G, H) ∈ TASKS:
      R_A ← SimpleLLMRoutine(m, n, G, H, T, r)   (Strategy A, Algorithm 2)
      R_B ← AgenticAIRoutine(m, n, G, H, r)      (Strategy B, Algorithm 7)
      for σ ∈ {A, B}:
        record (task, m, σ, template?, Acc, Effort, Sust, Difficulty)
```

Each phase writes its output to disk so later phases can be re-run independently (e.g. re-score without re-prompting).

## 7. Phase 1 — Structural analysis

### 7.1 Centrality
- Compute `cent(e)` for nodes of type `t` **and** for every element that can contribute properties to a context (neighbor nodes, edges).
- Default measure: PageRank. Must be normalized into `(0, 1]` because the difficulty uses `-log(cent)`.
- **[ASSUMPTION]** Normalization: divide by the max value per element type. A centrality of exactly 0 is replaced by a small epsilon (config) to keep `-log` finite.
- Edge centrality: the draft allows the alternative "probability of the node being source or target of the relationship". **[ASSUMPTION]** v1 implements edge centrality as the mean of the normalized centralities of its two endpoints; make the measure pluggable.

### 7.2 Context
- `context(n)` = properties from: `n` itself; direct neighbors; edges to direct neighbors; and elements reachable along paths where every relationship has `n:1` cardinality from the side of `n`.
- Cardinality `n:1` is detected from the data: relationship type `R` from type `A` to type `B` is `n:1` if every `A` node has at most one outgoing `R`.
- Each context property is stored with its **origin element** (to look up `cent_p(e)`), the path to it, and its value.

### 7.3 Popularity
- `pop(p)` is computed per element type: `#elements of type with non-null p / #elements of type`.
- Properties are identified by `(element_type, property_name)` so that `name` on City and `name` on Country are different properties.

**Acceptance:** for a toy graph with hand-computed values, centralities, contexts, and popularities match exactly (unit tests).

## 8. Task definition (Algorithm 4)

### 8.1 Procedure
1. Score H candidates by `pop(p)` only, and G candidate occurrences by `pop(p) × cent_p(e)`. H properties all belong to `n`, so including `cent(n)` would repeat the node factor and make the H stratum depend on the node stratum (see `TASKDEFINITION_SPEC.md` §5.1).
2. Compute quartiles `PH0..PH4` of the H score, `PG0..PG4` of the G score, and `Q0..Q4` of `cent()` over `N`.
3. Three **node strata**: `[Q0,Q1]` (low), `[Q1,Q3]` (middle, `Q2 ∪ Q3`), `[Q3,Q4]` (high). Draw `max_n` nodes from each.
4. For each drawn node, each `size_H ∈ [1, max_H]`, each `size_G ∈ [1, max_G]`:
   - for each property stratum `P_i ∈ {low, middle, high}` for H:
     - `H` ← draw `size_H` properties **of `n` itself** in that stratum
     - for each property stratum for G:
       - `G` ← draw `size_G` properties from `(n ∪ context(n)) − H` in that stratum, with their values
       - emit task `(n, H, G)`
5. Expected size: `27 × max_n × max_H × max_G` tasks (3 node strata × 3 H strata × 3 G strata).

### 8.2 Rules
- **H only contains properties with a non-null ground-truth value on `n`.** Otherwise the task cannot be scored.
- **No leakage:** G must not contain a property in H, and must not contain a context property that trivially reveals an H value (e.g. a duplicated or derived attribute). **[ASSUMPTION]** v1 enforces only exact-name exclusion on the same element; a configurable deny-list handles known duplicates per dataset.
- Sampling is seeded and deterministic.
- When a stratum has fewer candidates than requested, the combination is skipped and counted in the report (`TASKDEFINITION_SPEC.md` §5.3).
- Each task gets a stable `task_id` (hash of `n`, H, G).

**Acceptance:** with a fixed seed, task generation is byte-for-byte reproducible; no task violates the H/G rules; a report shows task counts per stratum.

## 9. Difficulty metric

For task `T = (n, H, G)`:

```
Difficulty(T) = ∛( |H| / (|H|+|G|)
                  × −log(cent(n))
                  × Σ_{p ∈ H ∪ G} −log(pop(p) × cent_p(e)) )
```

- **Log base 2.** Verified against Table 1 of the draft (e.g. task `(e1, {b}, {a,d})`: 0.33 × 0.51 × 6.14 → 1.01). This example must be a unit test.
- `cent_p(e)` is the centrality of the element the property comes from. For H, that is `n` itself.
- Store each of the three factors separately in the output, not only the final value, so alternative formulas can be compared later (the draft notes an earlier variant with the sum inside the log and no cube root).

**Acceptance:** all four rows of Table 1 are reproduced to two decimals.

## 10. Strategies

### 10.1 Common contract

Both strategies implement the same interface:

```
run(model, task, config) -> StrategyResult
StrategyResult:                           # one per run; written as one row of results.csv
  predictions: {prop_key: value | null}   # null = abstention
  variant: str                            # template id (A) or agent config id (B)
  usage: tokens_in, tokens_out, search_calls, latency, energy
```

The `results.csv` contract is defined in `EVALUATION_SPEC.md` §3.1.

Every `RawRun` stores the exact prompt sent, the raw response, parsed output, and parse errors.

### 10.2 Output format required from the model
- Models must return JSON: `{"<property>": "<value>" | null, ...}` with exactly the keys in H.
- An explicit `null` (or "unknown") is an **abstention**, not an error.
- Unparseable output: one retry with a format-repair instruction; if still invalid, that run counts as all-abstained and the failure is logged.

### 10.3 Strategy A — SimpleLLMRoutine (Algorithm 2)
1. For each template `t ∈ T`: instantiate with `(n, G, H)`.
2. Prompt the model `r` times, no external tools.
3. Write every run, **per template**. No aggregation (§10.5).

Details in `LLM_SPEC.md`.

Templates: `{base, compact} × {zero-shot, few-shot, CoT}` = 6 templates. Templates live as versioned text files; few-shot examples are drawn from nodes **not** in the task set.

### 10.4 Strategy B — AgenticAIRoutine (Algorithm 7)
Single-turn search agent:
1. **Plan:** given `(n, G, H)`, the model produces a list of search sub-queries (parallel where independent, ordered where one depends on another).
2. **Retrieve:** execute the sub-queries with the search tool (Tavily) in one tool-invocation step.
3. **Synthesize:** the model reads the retrieved evidence and outputs the JSON predictions for H, optionally with the source URL per value.
4. Repeat `r` times and write every run. No aggregation (§10.5).

Details in `AGENT_SPEC.md`.

Constraints:
- No multi-turn loop: one planning call, one retrieval step, one synthesis call.
- Store all queries, retrieved snippets, and URLs per run.
- **Leakage guard:** a configurable domain blocklist excludes the dataset's own source (e.g. the site the graph was built from), otherwise the agent may just look up the ground truth table.

### 10.5 Aggregation
- The strategies never aggregate. The evaluation selects one representative per configuration from the `r` runs (`EVALUATION_SPEC.md` §5.4).
- **[ASSUMPTION]** The draft asks "Majority voting??". v1 default is the medoid run; per-property majority vote (3 of 5, otherwise `null`) is the alternative method.

## 11. Metrics

### 11.1 Accuracy (F1)
- Predictions and ground truth are sets of `(property, value)` pairs.
- Precision = correct / non-abstained predictions; Recall = correct / |H|; F1 as usual. Report all three, plus **abstention rate**, since the paper's motivation for Strategy B is reducing abstention.
- **Value matching** uses normalization per property data type:
  - strings: case-folding, whitespace and accent normalization; optional alias table (e.g. country names);
  - numbers: relative or absolute tolerance configurable per property (e.g. elevation, runway length);
  - coordinates: distance tolerance;
  - codes (IATA/ICAO, etc.): exact match after normalization.
- Store both the strict-match and the tolerant-match results.

### 11.2 Prompt Formulation Effort (PFE)
- Proxy: number of tokens in the prompt template (before task instantiation) as written by a human.
- Use one fixed tokenizer for all models so values are comparable. **[ASSUMPTION]** `tiktoken` `cl100k_base`.
- Strategy B: PFE = tokens of the agent's planning + synthesis instructions.

TODO mudar isso aqui para a nova proposta de effort


### 11.3 Sustainability
- The draft defines a composite score but not its formula yet. v1 **records the raw components**, and the composite is computed afterwards:
  - energy (kWh) and CO2e: measured with CodeCarbon for local models, estimated with EcoLogits for API models (`LLM_SPEC.md` §6.5);
  - token counts in/out, number of search calls;
  - wall-clock latency;
  - model size (parameter count) when known.
- EcoLogits values are estimates and are flagged as such; if a model version is not in EcoLogits, the closest model of the same provider is used and recorded.

### 11.4 Results table (one row per task × model × strategy × template)

| Column | Notes |
|---|---|
| `task_id`, `node_id`, `node_stratum`, `H_stratum`, `G_stratum`, `size_H`, `size_G` | Task description |
| `difficulty`, `diff_ratio`, `diff_node_surprise`, `diff_prop_surprise` | §9 |
| `model`, `strategy`, `template_id` | `template_id` null for B |
| `precision`, `recall`, `f1`, `abstention_rate` | strict and tolerant variants |
| `pfe_tokens` | §11.2 |
| `tokens_in`, `tokens_out`, `search_calls`, `latency_s`, `energy_kwh`, `co2e_kg` | §11.3 |

Output as Parquet (plus CSV export).

## 12. Non-functional requirements

- **Reproducibility:** seeds for sampling; model parameters and template versions recorded in each run's manifest.
- **Caching:** identical `(model, params, prompt)` requests are cached to disk, so interrupted runs resume and re-scoring is free. Cache is bypassed for the `r` repetitions (each repetition has its own run index in the cache key).
- **Resumability:** the runner skips tasks whose results already exist in `output_dir`.
- **Rate limits / failures:** exponential backoff; failed calls are logged, never silently dropped.
- **Separation of phases:** structural analysis, task generation, execution, and scoring are separate commands with on-disk outputs.
- **Datasets:** adding a new graph requires only a config file and, if needed, a dataset-specific deny-list and value-normalization rules.

## 13. Open questions

Items where the draft is ambiguous or inconsistent. Each has a v1 assumption above; confirm with co-authors.

1. **Graph source.** Are all three datasets in Neo4j? Which are the other two besides the airport graph?
2. **Algorithm 5 scoring is outside the task loop** (lines 14–16 run once per model, after the loop). Treated as a typo; the spec follows Algorithm 1, scoring per task.
3. **Strategy A returns one result per template** (Algorithm 2) but Algorithm 6 says it returns "the best results ... and the template used". Is the comparison with B done per template, or with the best template per task? v1 stores all templates; "best" can be selected at analysis time.
4. **Algorithm 2 says zero-shot**, while §3.1 lists zero-shot, few-shot, and CoT templates. v1 supports all six.
5. **Aggregation** of `r` runs: medoid run (v1 default) or per-property majority vote (§10.5, `EVALUATION_SPEC.md` §11.1)?
6. **Strategy B "single-turn" vs. "sequential dependency resolution".** If a sub-query depends on another's answer, that needs a second retrieval step. Is one dependent hop allowed?
7. **PFE for Strategy B:** what counts as the "prompt" for an agent?
8. **Sustainability composite formula** and which components it weights.
9. **Edge centrality** definition (§7.1).
10. **Algorithm 4, line 15** uses `cent_p(n)` while line 17 uses `cent_p(e)`. v1 uses the origin element's centrality for both (for H that's `n` anyway).
11. **Difficulty formula:** the sum term grows with `|H|+|G|` while the ratio term only depends on proportions. Intended? Also the draft's own question on whether the H/G swap symmetry is desirable.
12. **Value-matching tolerances** per property type, and whether near-misses count as correct.
13. **Search leakage:** which domains to block for each dataset in Strategy B.
14. **Models in M:** which models, local vs. API, and with which parameters (temperature affects the meaning of `r`).

## 14. Milestones

1. Graph loading, centrality, context, popularity — with unit tests on a toy graph.
2. Task generation (Algorithm 4) + difficulty metric reproducing Table 1.
3. Model provider abstraction, caching, raw run logging.
4. Strategy A with the six templates.
5. Scoring: F1 with value normalization, PFE, sustainability components.
6. Strategy B (search agent with Tavily).
7. Full run on the first dataset; results table and basic plots (F1 vs. difficulty per strategy and model).
8. Add the second and third datasets.
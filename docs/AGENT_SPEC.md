# SPEC: Search Agent (Strategy B — Single-Turn Agentic Pathway)

> Component spec for the web search agent described in the paper *"Data Augmentation in Property Graphs using agentic AI"* (draft). It covers only `AgenticAIRoutine` (Algorithm 7). Task generation, difficulty, Strategy A, and the global protocol are in `SPEC.md`.
>
> Related specs: `TASKDEFINITION_SPEC.md` (produces `tasks.csv`), `LLM_SPEC.md` (Strategy A), `EVALUATION_SPEC.md` (reads `results.csv` and defines its format).
>
> Where the draft is ambiguous, the v1 choice is marked **[ASSUMPTION]** and listed in [Open questions](#12-open-questions).

---

## 1. Purpose

You'll be given an CSV file that was generated using the following procedure: 
Given a node `n`, a set `G` of known `(property, value)` pairs, and a set `H` of property names whose values are hidden.

Then, the task is that the agent finds values for `H` by searching the web and returns them as structured predictions with supporting evidence.

The agent exists to reduce the **abstention** of Strategy A: a model that doesn't know a value from memory may still find it online.

## 2. Design principles

1. **Single-turn.** One planning call, one retrieval step, one synthesis call. No open-ended agent loop. This keeps cost, latency, and energy bounded and comparable across models.
2. **Same model, same inputs as Strategy A.** The only difference from Strategy A is access to search. The model `m` under evaluation does the planning and synthesis.
3. **Evidence-grounded.** Every non-null value must cite at least one retrieved source. The agent must abstain rather than guess.
4. **Fully logged.** Every query, result, and model call is stored so runs can be audited and re-scored without re-querying.

## 3. Out of scope (v1)

- Multi-turn reasoning loops (ReAct-style), reflection, or self-correction rounds.
- Opening and reading full web pages beyond what the search API returns (no crawler).
- Tools other than web search (no SQL, no code execution, no graph queries).
- Writing results back to the graph.

## 4. Interface

```
AgenticAIRoutine(model, task, config) -> StrategyResult

task:                                            # one row of tasks.csv
  task_id: str
  node_id, node_type: str
  G: [{prop_key, property_name, element_type, element_id, path, value}]   # known context
  H: [{prop_key, property_name, element_type, path}]                       # to predict

StrategyResult:                                  # one per run, never aggregated
  predictions: {prop_key: value | null}
  evidence:    {prop_key: [source_url]}           # for non-null predictions
  usage: {tokens_in, tokens_out, search_calls, latency_s, energy_kwh, co2e_kg}
```

Field names follow `tasks.csv` (`TASKDEFINITION_SPEC.md` §7.1). Each run is written as one row of `results.csv` with the same contract as Strategy A (§10.3), so the evaluation doesn't need to know which strategy produced it.

**Hard rule:** this component never reads `ground_truth.csv`.

## 5. Execution flow (one run)

```
          ┌──────────┐     ┌────────────┐     ┌─────────────┐
 task ──► │ 1. PLAN  │ ──► │ 2. RETRIEVE│ ──► │ 3. SYNTHESIZE│ ──► predictions + evidence
          └──────────┘     └────────────┘     └─────────────┘
           LLM call         search API          LLM call
                            (parallel)
```

### 5.1 Step 1 — Plan

The model receives the node, `G`, and `H`, and returns a search plan as JSON:

```json
{
  "queries": [
    {"id": "q1", "query": "Guarulhos airport elevation feet", "targets": ["elevation"]},
    {"id": "q2", "query": "GRU airport number of runways", "targets": ["runways"]}
  ]
}
```

Rules:
- Each query lists the `H` properties it targets. Every property in `H` must be targeted by at least one query.
- Queries should use identifying values from `G` (name, code, city, country) to disambiguate the entity.
- Number of queries is capped by `max_subqueries` (config). Excess queries are truncated and logged.
- Planning must not include any value for an `H` property (the model can't "answer in the plan"). The validator rejects plans whose queries contain a predicted value for a target property only when that can be detected; see §8.

**Dependencies [ASSUMPTION].** The draft mentions "sequential dependency resolution" but also "single-turn tool invocation". v1 resolves this with a `waves` setting:
- `waves: 1` (default, strictly single-turn): all queries are independent and run in parallel. Queries may not have `depends_on`.
- `waves: 2` (optional): queries may declare `"depends_on": ["q1"]` and use a placeholder `{q1}`; after wave 1 returns, one extra short LLM call fills the placeholder from the wave-1 results, then wave 2 runs. This adds exactly one LLM call and one retrieval step, never more.

Results for `waves: 1` and `waves: 2` are reported as separate configurations, never mixed.

### 5.2 Step 2 — Retrieve

- Execute all queries of a wave **in parallel** against the search tool. If the model has an web search tool by default and the configuration for it is active, then use the search tool. Otherwise, use Tavily by default.
- Per query: request `results_per_query` results; keep title, URL, snippet/content, and the search provider's score.
- Apply the **leakage blocklist** (§7) through the search API's domain-exclusion option, and filter again locally after results return.
- Deduplicate results by URL across queries.
- Truncate the evidence passed to synthesis to a token budget (`max_evidence_tokens`), prioritizing results by provider score while keeping at least one result per query when possible.

Failure handling:
- A failed query is retried with backoff (`max_retries`). If it still fails, synthesis proceeds without it, and the failure is recorded.
- If every query fails, the run returns all-null predictions with `status: retrieval_failed`.

### 5.3 Step 3 — Synthesize

The model receives the node, `G`, `H`, and the numbered evidence list, and returns:

```json
{
  "predictions": {
    "elevation": {"value": "2459", "unit": "ft", "sources": [3, 5]},
    "runways":   {"value": null, "sources": []}
  }
}
```

Rules given to the model:
- Use only the evidence provided. Do not use outside knowledge to fill a value. This isolates the effect of search from parametric memory (Strategy A).
- Every non-null value must cite one or more evidence ids. A value with no valid citation is converted to `null` by the validator.
- If sources conflict, prefer the source that most clearly refers to the same entity (matching identifiers from `G`); if still unresolved, return `null`.
- Return exactly the keys in `H`.

**Flattening into `results.csv` [ASSUMPTION].** `predictions[prop_key]` is the value with its unit appended when a unit is given (`"2459 ft"`), and the raw value otherwise. The evaluation normalizer parses numbers and units from this string (`EVALUATION_SPEC.md` §5.2). `evidence[prop_key]` holds the URLs of the cited evidence ids.

## 6. Repetitions

- The full run (plan → retrieve → synthesize) is repeated 5 times (`run_index` 1..5), as in Strategy A.
- Each repetition re-plans and re-searches. Search results may vary between calls; that variance is part of what is measured. A config flag `reuse_retrieval: true` allows running synthesis 5 times on a single retrieval, to separate model variance from search variance.
- The agent only writes the 5 runs. It does not aggregate them or select a representative; that is done by the evaluation (`EVALUATION_SPEC.md` §5.4).


## 7. Leakage guard

The web may contain the exact dataset the graph was built from. If the agent finds that table, it measures lookup rather than augmentation.

- Each dataset config declares a `blocked_domains` list (e.g. the original dataset's website, mirrors, GitHub repos hosting the raw files).
- Blocking is applied both in the search request and by a local post-filter.
- Every retained source URL is logged, so leakage can be audited after the fact. An analysis script reports the most frequent domains per dataset to help find new leaks.

## 8. Validation layer

Applied after each LLM call; violations are logged with the run.

| Check | Action on failure |
|---|---|
| Plan is valid JSON with the expected schema | One repair retry, then run fails with `status: plan_invalid` |
| Every `H` property targeted by a query | Add a default query `"<node label> <property>"` for missing ones |
| Number of queries ≤ `max_subqueries` | Truncate, log |
| Synthesis is valid JSON with exactly the `H` keys | One repair retry, then all-null with `status: synthesis_invalid` |
| Non-null value cites a valid evidence id | Set value to `null`, log `uncited_value` |
| Cited source is from a blocked domain | Set value to `null`, log `leak_blocked` |

## 9. Configuration

```yaml
strategy_b:
  variant: waves1          # written to `variant` in results.csv; use a distinct id for every distinct config
  search_tool: tavily
  search_params:
    depth: basic            # basic | advanced
    results_per_query: 5
  max_subqueries: 6
  waves: 1                  # 1 = strictly single-turn; 2 = one dependent hop
  max_evidence_tokens: 4000
  reuse_retrieval: false
  max_retries: 2
  timeout_s: 30
  prompts:
    plan: prompts/agent_plan_v1.txt
    synthesize: prompts/agent_synth_v1.txt
    fill_dependency: prompts/agent_fill_v1.txt   # used only when waves: 2
datasets:
  airports:
    blocked_domains: [...]
```

Prompt files are versioned; the version is recorded in every run.

## 10. Logging and metrics

### 10.1 Per run (`AgentRun`)

| Field | Content |
|---|---|
| `run_index`, `status` | `ok`, `retrieval_failed`, `plan_invalid`, `synthesis_invalid` |
| `plan_prompt`, `plan_raw`, `plan_parsed` | Planning call |
| `queries` | Each query, its targets, latency, error if any |
| `results` | URL, title, snippet, score, `blocked: bool` |
| `evidence_passed` | The numbered evidence list as sent to synthesis |
| `synth_prompt`, `synth_raw`, `synth_parsed` | Synthesis call |
| `validation_events` | List of §8 events |
| `usage` | Tokens in/out per call, search calls, latency, energy |

### 10.2 Agent-specific metrics

The agent only logs what these metrics need; they are computed by the evaluation (`EVALUATION_SPEC.md` §6.6), since some of them use the ground truth, which the agent never reads.

- **Abstention rate** (main comparison point with Strategy A).
- **Retrieval coverage:** fraction of `H` properties for which at least one retrieved result contains the ground-truth value (after normalization). This separates *search failures* from *synthesis failures*.
- **Grounded accuracy:** fraction of correct predictions whose cited source actually contains the value.
- **Search calls per task** and **evidence tokens per task**, for the sustainability analysis.
- **PFE for the agent [ASSUMPTION]:** token count of the plan + synthesis prompt templates (plus the fill template when `waves: 2`).

### 10.3 `results.csv` and `runs.jsonl`

- `results.csv`: one row per run, following `EVALUATION_SPEC.md` §3.1, with `strategy = B`, `variant` = agent config id (§9), `search_calls` = number of queries executed, `evidence` as in §5.3.
- `runs.jsonl`: one `AgentRun` (§10.1) per line, keyed by `task_id`, model, `variant`, `run_index`.

## 11. Testing

- **Unit tests** with a mocked search tool returning fixed results: plan validation, dependency filling, evidence truncation, citation enforcement, blocklist filtering, flattening into `results.csv`.
- **Golden tests**: a handful of real tasks with recorded search responses (cassettes), so the full flow runs offline and deterministically in CI.
- **Leak test**: a mock result from a blocked domain must never appear in the evidence or be cited.
- **Smoke test** against the live search API, run manually, with a small task budget.

**Acceptance criteria**
- With mocked search, a run produces a valid `StrategyResult` for any task, including when all queries fail.
- No non-null prediction exists without a valid, non-blocked citation.
- For the same seed, cached LLM responses, and recorded search results, the output is identical.

## 12. Open questions

1. **Single-turn vs. sequential dependencies:** is one dependent hop (`waves: 2`) acceptable in the paper's definition of "single-turn"?
2. **Evidence-only vs. evidence + memory:** should the synthesis step be allowed to use the model's own knowledge when evidence is missing? Evidence-only gives a cleaner comparison with Strategy A; allowing memory gives a stronger agent but mixes the two effects.
3. **Repetitions:** re-search on every repetition, or search once and repeat only synthesis?
4. **Search depth and result count:** fixed across all models, or tuned?
5. **PFE for the agent:** which prompts count as human effort?
6. **Blocked domains** per dataset.
7. **Search tool cost and energy:** how to include the search API (which runs on third-party infrastructure) in the sustainability score.
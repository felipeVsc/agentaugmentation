# SPEC: Metrics stored for the paper

> What every step of the pipeline must record, so that the analyses of the paper *"Data Augmentation in Property Graphs using agentic AI"* (October 2026 draft, `docs/infosysnewversion.pdf`) can be made, and so can analyses nobody has planned yet, **without running a model or a search again**.
>
> Related specs: `TASKDEFINITION_SPEC.md` (tasks), `LLM_SPEC.md` (Strategy A), `AGENT_SPEC.md` (Strategy B), `EVALUATION_SPEC.md` (scoring). This spec doesn't redefine the formulas those specs own; it lists what has to be stored, where, and whether the code stores it today.
>
> **Status** column: **✓** stored today · **D** derivable from what is stored (no rerun needed) · **✗** missing: must be added before the paper runs.

---

## 1. Principles

1. **Store raw, derive later.** Log counts, timings, and texts; compute rates, ratios, and composite scores in the evaluation. A formula that changes (the difficulty, the energy multiplier, the effort metric are all still being discussed in the draft) must only need a new evaluation, never a new run.
2. **One grain per file, joined by ids.** Each metric lives at the finest grain it exists at (§2) and carries the ids of the grains above it, so any grouping can be built by a join.
3. **Never lose a failure.** A failed call, a retry, a refused search, an empty answer, each leaves a record with its reason. Failure and abstention rates are results.
4. **Separate measured from estimated.** Every energy, token, and cost value says how it was obtained (`measured`, `estimated`, `counted_with=<tokenizer>`), since the paper must report them as different methods.
5. **Record the world's state.** Web results change over time, and model tags on Ollama Cloud can be updated. Keep the dates, versions, and the retrieved content.
6. **No secrets.** API keys, Neo4j credentials, and the user's email are never part of any record.

## 2. Grains and files

| Grain | Key | File | Producer |
|---|---|---|---|
| Task set | `task_set_version` | `tasks/<ds>/<v>/manifest.json` | Task definition |
| Task | `task_id` | `tasks.csv` | Task definition |
| Task property | `task_id, prop_key, role (H\|G)` | `tasks.csv` (JSON columns H, G) + `ground_truth.csv` | Task definition |
| Node context | `node_id, prop_key, element_id` | `intermediate/context.parquet` | Task definition |
| Session | `strategy, output_dir` | `run_manifest.json` | Strategy A / B |
| Run | `task_id, strategy, variant, model, run_index` | `results.csv` (summary) + `runs.jsonl` (full record) | Strategy A / B |
| LLM call | run key + `step`, `call_index` | `runs.jsonl` → `calls[]` | Strategy A / B |
| Search query | run key + `query_id` | `runs.jsonl` → `queries[]`, `results[]` | Strategy B |
| Scored property | configuration + `prop_key` | `property_scores.csv` | Evaluation |
| Configuration | `task_id, strategy, variant, model` | `task_scores.csv`, `representatives.csv` | Evaluation |

A **configuration** is one (task, strategy, variant, model) with its `r` runs. A **variant** is a template (A, with or without context) or an agent configuration (B).

## 3. Task metrics (task definition)

Needed for: the difficulty analysis (§3.2 of the draft), the accuracy-vs-difficulty figure (Fig. 3), and the strata tables.

### 3.1 Per task

| Metric | Definition | Status |
|---|---|---|
| `node_id`, `node_type`, `dataset` | The task's node | ✓ |
| `node_centrality`, `node_centrality_raw` | `cent(n)` normalized, and the raw PageRank / degree | ✓ |
| `node_stratum`, `H_stratum`, `G_stratum` | Strata used to draw the task (`low/mid/high`) | ✓ |
| `size_H`, `size_G` | `|H|`, `|G|` | ✓ |
| `difficulty` | Draft §3.2 | ✓ |
| `diff_ratio`, `diff_node_surprise`, `diff_prop_surprise` | The three factors, so other combinations (product, mean, sum split into H and G) can be tested | ✓ |
| `diff_prop_surprise_H`, `diff_prop_surprise_G` | The sum term split between H and G. The draft notes the sum dominates the product, and the H↔G swap gives equal difficulty; testing that needs the split | ✓ (`diff_prop_surprise_H`, `diff_prop_surprise_G`) |
| `n_G_own`, `n_G_context` | G properties of `n` itself vs. from its context | ✓ |
| `G_has_own_property` | At least one G property is of `n`. The draft flags tasks where G is all context as much harder than their score says | ✓ |
| `G_identifying` | G contains an identifying property (name, code, IATA/ICAO, city) per a per-dataset list. In the smoke test this decided whether search could work at all | ✗ (needs `identifying_properties` in the dataset config) |
| `G_max_hops` | Longest path length among G properties (0 = own, 1 = neighbor, …); also `hops` on each G entry | ✓ |
| `H_value_types` | Value types of the H properties (string, number, coordinate, date) | D (`ground_truth.csv`) |
| `G_candidates` | Number of eligible G properties of `n ∪ context(n)` (of `n` only without context), i.e. how much the sampler could choose from | ✓ |
| `context_elements` | Number of distinct elements in the context | D (`context.parquet`) |
| `skips_for_node` | Combinations skipped for this node and why | ✓ (manifest report) |

### 3.2 Per task property (H and G)

| Metric | Status |
|---|---|
| `prop_key`, `property_name`, `element_type`, `element_id`, `path` | ✓ |
| `pop(p)`, `cent_p(e)`, `m_p = pop × cent` | ✓ (`popularity`, `centrality`, `score` on every H and G entry; no H values) |
| `value` (G) / ground truth (H), `value_type` | ✓ |
| `in_web_scope` | Whether the hidden value is a fact about the world (altitude, IATA) or an artefact of the dataset (mean runway length, `source`, `dst`, `distCity`). The smoke test showed these set a ceiling for any search strategy | ✗ (per-dataset list in the evaluation config; report separately) |

### 3.3 Per task set (manifest)

Quartile boundaries and interval convention, strata membership, `n:1` relationships, graph statistics, snapshot hash, seed, code version, skipped/duplicate counts. ✓ (`TASKDEFINITION_SPEC.md` §7.3).

## 4. Run and call metrics (Strategy A and B)

Needed for: effort (§3.4.1 of the draft), sustainability, the agent energy formula, cost, and latency.

### 4.1 Tokens

Stored **per call**, summed **per run**.

| Metric | Definition | Status |
|---|---|---|
| `tokens_in` | Prompt tokens (system + user + evidence), as reported by the provider, or counted with the configured tokenizer when the provider undercounts because of its prompt cache | ✓ |
| `tokens_in_counted_with` | `provider` or the tokenizer name | ✓ (`runs.jsonl`) |
| `tokens_in_cached` | Prompt tokens served from the provider's prompt cache (Ollama `prompt_eval_cached_count`; OpenAI `cached_tokens`; Anthropic `cache_read_input_tokens`). Affects cost, energy, and latency | ✓ Ollama (`usage.tokens_in_cached` in `runs.jsonl`); other providers only inside `raw` |
| `tokens_out` | All generated tokens, **including reasoning** | ✓ |
| `tokens_reasoning` | Reasoning ("thinking") tokens only, a part of `tokens_out`. gpt-oss cannot turn reasoning off and spent ~160 output tokens per call against ~25 for gemma4, so this must be separate. Sources: OpenAI `completion_tokens_details.reasoning_tokens`; Google `thoughts_token_count`; Ollama returns the `thinking` text but no count, so it is counted with the run's tokenizer (an estimate; 0 when the model did not think). Empty when the provider gives no count | ✓ (`results.csv`; `llmcore.tokenizers.reasoning_tokens`) |
| `tokens_answer` | `tokens_out − tokens_reasoning` | D once `tokens_reasoning` exists |
| `tokens_prompt_template` | Tokens of the instruction part only (template without the task) = PFE | ✓ (`pfe_tokens`) |
| `tokens_task_input` | Tokens of the entity's known values (the "input:" part) | ✓ (`effort.input` in `runs.jsonl`, §6) |
| `tokens_context` | Tokens of the context part of the prompt (with-context variants; 0 otherwise). Lets the paper separate the cost of context from the cost of the template | ✓ (`effort.context`, §6) |
| `tokens_evidence` | Strategy B: tokens of the search evidence passed to synthesis | ✓ (`evidence_tokens`) |
| `finish_reason` | `stop`, `length`, … `length` with an empty answer is how reasoning models fail | ✓ (per call) |

Check: `tokens_in ≈ tokens_prompt_template + tokens_task_input + tokens_context + tokens_evidence + formatting overhead`. Logging the parts makes the overhead visible.

### 4.2 Calls and tools

| Metric | Definition | Status |
|---|---|---|
| `n_llm_calls` (`N_calls`) | LLM invocations in the run (A: 1; B: plan + synthesis, plus any repair call). B runs that reuse run 1's plan do not count it | ✓ (`llm_calls` in `results.csv`) |
| `n_tool_calls` (`N_tools`) | Tool invocations that actually ran (search requests sent to the API) | ✓ (`search_calls`) |
| `n_tool_calls_cached` | Searches answered by the search buffer, with no API call | ✓ (`search_cache_hits`) |
| `n_llm_calls_cached` | Calls answered by the LLM cache (shared plan, rerun) | ✓ (`cached` per call) |
| `step` | Which step a call belongs to (`answer`, `plan`, `synthesize`, `repair`) | ✓ (B); A has a single step |
| `n_retries`, `retry_reasons` | Retries per call and why (429, timeout, 5xx). Retries cost time and, on some providers, tokens | ✗ (only session totals in `Caller.stats`) |
| `shared_with_run` | B with `reuse_retrieval`: the run whose plan and searches were reused. Needed to count energy and credits once, not five times | ✓ (`retrieval_shared_with_run`) |

**EIS = the agent energy formula (draft §3.4; decided).** The environmental impact score of the paper is

```
E_agent = (N_calls + N_tools · w_tool) · E_single
```

with `N_calls = llm_calls`, `N_tools = search_calls`, and `E_single` = the run's LLM energy ÷ `llm_calls` (mean EcoLogits or CodeCarbon energy of one call). For Strategy A, `N_tools = 0`, so its EIS is its LLM energy. It is computed in the evaluation (`evaluation/eis.py`: `agent_energy`, `run_eis`), never stored, so `w_tool` can be set or changed without rerunning. **`w_tool` is not chosen yet**: the function refuses to run without it.

Searches answered by the search buffer are counted by default (the figure a reader would reproduce without the buffer); `count_cached_searches=False` gives the energy actually spent. The DOLAP Composite EIS is not used.

### 4.3 Energy and emissions

Stored **per call** (the energy of an LLM call depends on its own tokens).

| Metric | Status |
|---|---|
| `energy_kwh`, `co2e_kg` | ✓ |
| `energy_kwh_min`, `energy_kwh_max`, `co2e_kg_min`, `co2e_kg_max` (EcoLogits ranges) | ✓ (`runs.jsonl`) |
| `energy_method` (`codecarbon`, `ecologits`, `none`), `estimated`, `gpu_measured` | ✓ |
| `ecologits_model` (the model used for the estimate; may be a stand-in) | ✓ |
| `energy_country` / electricity mix | ✓ (config) → ✗ in the run record |
| Search API energy | ✗, and not estimable from the outside. Report as excluded; `w_tool` is the paper's proxy |
| EcoLogits other impacts (`adpe`, `pe`, water where available) | ✗. Not needed: the EIS is the agent formula over energy (§4.2) |

Models without an EcoLogits entry (Nemotron-3-Super, GLM-5.3-flash) have `energy_method: none`. They must still have `tokens_*` and `n_*`, so an estimate can be added later from a stand-in model without rerunning.

### 4.4 Time

| Metric | Definition | Status |
|---|---|---|
| `latency_s` | Wall clock of the run | ✓ |
| `llm_latency_s`, `search_latency_s` | Time in LLM calls and in searches (B) | ✓ (`runs.jsonl`) |
| `wait_s` | Time spent in retry backoff and concurrency queues. Without it, latency on a rate-limited free tier measures the provider's limits, not the strategy | ✗ |
| `provider_duration_s` | Server-side time reported by the provider (Ollama `total_duration`) | ✓ (inside `raw`) |
| `timestamp` (start, ISO 8601, UTC) | ✓ |

### 4.5 Cost

| Metric | Definition | Status |
|---|---|---|
| `cost_usd` | LLM cost of the run in USD: Σ over calls of `tokens_in × input price + tokens_out × output price` (reasoning tokens are output tokens). Empty when the model has no price. Search is not included: see `search_credits` | ✓ (`results.csv`; per call in `usage.cost_usd_per_call`) |
| Price table | `pricing: {input_per_mtok, output_per_mtok, as_of}` on each model in the strategy configs, copied into the run manifest (table below) | ✓ |
| `search_credits` | Tavily credits spent (0 for cache hits) | ✓ |
| `search_depth` | `basic` / `advanced` | ✓ (config, manifest) |
| `execution` | `realtime` / `batch` (batch is cheaper) | ✓ |

Prices (Ollama Cloud, USD per 1M tokens, as of 2026-10-27):

| Model | Size | Input | Output |
|---|---|---|---|
| `gpt-oss:20b` | 20B | 0.07 | 0.30 |
| `gpt-oss:120b` | 120B | 0.15 | 0.60 |
| `nemotron-3-super` | 120B | 0.01 | 0.60 |
| `glm-5.3-flash` | 321B / 18B active | 0.15 | 0.50 |
| `mistral-large-3:675b` | 675B | 0.50 | 1.50 |

A price change goes into the config with a new `as_of`; runs keep the cost computed at the time, and the evaluation can recompute any cost from the stored tokens.

### 4.6 Output quality signals (before scoring)

| Metric | Definition | Status |
|---|---|---|
| `status` | `ok`, `parse_error`, `api_error`, `timeout`, `retrieval_failed`, `budget`, … | ✓ |
| `format_compliant` | The raw answer followed the requested format (the published scoring counted only compliant answers) | ✓ (`status` + `parse_events`) |
| `repaired` | The answer was accepted only after a repair | ✓ (A) |
| `empty_answer` | The model returned no answer text (seen with gpt-oss, where reasoning used the budget) | D (from `response.text`) → ✗ as a flag |
| `n_extra_properties` | Properties answered that were not in H. The published F1 counted them | D (`parsed` vs. `label_map`) |
| `n_echoed_values` | Answers copying a value from G. They inflate the published F1 (seen with qwen3-4B) | D |
| `raw text, parsed answer, predictions` | ✓ (`runs.jsonl`) |

## 5. Strategy B retrieval metrics

Needed for: explaining *why* search helps or fails (the smoke test's finding that hard-hard tasks can't identify the node), grounded accuracy, and the cost of the agent.

### 5.1 Per run

| Metric | Definition | Status |
|---|---|---|
| `n_queries` | Queries planned | D (`queries[]`) |
| `n_queries_fallback` | Queries made by the fallback, because the plan failed to parse | D (`queries[].source`) |
| `plan_parse_ok` | The plan was valid JSON with valid targets | D |
| `targets_coverage` | Share of H properties targeted by at least one query | D |
| `query_has_identifier` | Queries contain a G value that identifies the node (§3.1 `G_identifying`) | D once the identifying list exists |
| `query_leaks_truth` | A query contains a hidden value (the model guessed it). Must be 0 by prompt rule; checked, not assumed | D (evaluation, using ground truth) |
| `n_results`, `n_results_blocked` | Results returned and dropped by the blocklist | ✓ / D |
| `n_unique_domains` | Distinct domains among the results | D |
| `evidence_passed` | Results actually given to synthesis, after truncation | ✓ |
| `evidence_truncated` | Evidence was cut to fit the budget | ✗ |
| `cited_urls` | Per property | ✓ (`evidence_urls`) |
| `search_date` | Date of the actual API request (not the cache read) | ✓ (cache entry) → ✗ in the run record |

### 5.2 Computed by the evaluation (ground truth needed)

| Metric | Definition |
|---|---|
| `retrieval_coverage` | Share of H properties whose true value appears in the retrieved text (tolerant match). It separates *search failed* from *synthesis failed* |
| `grounded_correct` | Correct answer whose value appears in a cited result |
| `ungrounded_correct` | Correct answer not supported by its citation (from the model's memory; expected in `search_memory`) |
| `citation_rate` | Share of non-null answers with at least one source |
| `source_is_blocked_domain` | Must be 0; checks the blocklist held |

These need `results[].content` to be stored in full, which is the case today. Do not drop it to save space.

## 6. Effort metrics

**Decision: store everything; choose between PFE, KLM, and token amplification later.** Every run record in `runs.jsonl` has an `effort` block with the size of each text a person would write, in characters, words, and tokens (the run's tokenizer):

| Part | Text | Written |
|---|---|---|
| `template` | The variant's instructions and few-shot examples, without placeholders (the PFE text). B: the plan and synthesis prompts | Once per variant |
| `input` | The entity's known values (the published `input:` line, or G's own properties) | Per task |
| `context` | The graph context given with it (empty without context) | Per task |
| `targets` | The names of the properties to find | Per task |

The agent's search queries are not in it: the agent writes them, not a person.

| Metric | Computed from | Status |
|---|---|---|
| **PFE** (DOLAP): tokens of the template | `pfe_tokens` (`results.csv`) = `effort.template.tokens` | ✓ |
| **KLM**: `E = N_K·K + N_P·P + N_H·H + N_M·M` per strategy | `N_K` = characters typed (`effort.*.chars`, choosing which parts the user types); `N_P`, `N_H`, `N_M` from a per-strategy operator table in the evaluation; operator times from Card et al. | ✓ inputs stored; the operator table is chosen with the metric |
| **Token amplification** (Patrick): AI-generated ÷ human-written tokens | `tokens_out` (with or without `tokens_reasoning`) ÷ (`effort.template.tokens` + `effort.input.tokens` + `effort.context.tokens` + `effort.targets.tokens`) | ✓ inputs stored |

Whether the template counts per task or once per variant (amortized) is a choice made at analysis time; both are possible from these fields.

## 7. Session metadata (run manifest)

| Item | Status |
|---|---|
| Strategy, variants, templates and their versions, `pfe_tokens` | ✓ |
| Model id, provider, tag, digest / version, quantization, parameter count (total and active), context length | ✓ partly (`model_details`); ✗ total/active parameters (needed for size vs. accuracy plots; take from the config comment into a field) |
| Generation parameters (temperature, max_tokens, seed, `think`) | ✓ (config) |
| Search tool, depth, `max_results`, blocklist, `reuse_retrieval`, credit cap | ✓ |
| Price table and its date (§4.5) | ✗ |
| Concurrency, retries, timeout | ✓ (config) |
| Code version, start and end time, host (local GPU model or cloud) | ✓ code version; ✗ host / hardware |
| Status counts, cache hits, credits spent | ✓ |

## 8. Accuracy metrics (evaluation)

Owned by `EVALUATION_SPEC.md`; listed here so the stored inputs are checked against them.

- **Per property:** outcome `correct / wrong / abstained` under each scoring mode: `strict`, `tolerant`, `standard` (unit-aware, `evaluation/dolap.py`) and `published` (DOLAP rules). Keep all modes: the published mode reproduces the DOLAP numbers, the others are the paper's.
- **Per configuration:** precision, recall, F1, exact match, abstention rate, error rate; agreement rate across runs; the representative run (or majority vote) and the oracle best-of-r.
- **Groupings:** model × strategy × variant, × each stratum and size, × difficulty bins, × `G_has_own_property`, × `G_identifying`, × `in_web_scope`, × property, × dataset.
- **Paired A vs. B:** the same (task, model), with bootstrap CIs and Wilcoxon test.
- **Efficiency:** correct properties per 1k tokens, per Wh, per USD, per search credit (all from §4).

## 9. Gaps to close before the paper runs

Done: reasoning tokens, cached input tokens (Ollama), LLM calls, cost in USD, and the effort texts are now stored by both runners; the EIS formula is in the evaluation. Old `results.csv` files get the new columns (empty) when a runner reopens them.

Still missing, in order of how much a rerun would cost if they are missed:

1. **`n_retries`, `wait_s`** per call (§4.2, §4.4). Free-tier 429s otherwise pollute latency.
2. **Search date** in the run record (§5.1).
3. **Model size** (total / active parameters) and **host** in the run manifest (§7).
4. **Dataset lists** `identifying_properties` and `in_web_scope` (§3), per dataset, in config.

Items 1–2 change what the runners log; 3–4 are config and can be added later. The task columns of §3.1 are written by `tasks build`.

## 10. Open questions

1. **`w_tool`** for the EIS (§4.2): to be given. Until then `agent_energy` refuses to compute.
2. **Effort metric** (§6): PFE, KLM, or token amplification as the headline; all inputs are stored. Patrick notes effort, difficulty, and sustainability may correlate and suggests one global metric.
3. **Properties that only exist in the dataset** (`in_web_scope = false`, §3.2). Some hidden properties are not facts about the world but values the dataset authors computed or chose: `runwaysLength` is the *mean* length of all runways (no web page states it), `distCity` is a distance the ETL computed, `source` and `dst` are codes of the source dataset. No model and no search can find them, except by luck. If they stay in H, they lower every strategy's recall by the same amount and hide the difference between A and B. Options: (a) keep them and report as is; (b) report every metric twice, with and without them; (c) exclude them from H when building tasks. (b) needs only a per-dataset list of these properties and costs no rerun.

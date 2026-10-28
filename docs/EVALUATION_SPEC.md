# SPEC: Evaluation

> Component spec for the evaluation step of the paper *"Data Augmentation in Property Graphs using agentic AI"* (draft).
> It reads the `results.csv` files produced by Strategy A and Strategy B, selects one representative answer out of the 5 runs of each configuration, compares it with the ground truth, and computes accuracy, F1, effort, and sustainability metrics.
>
> Related specs: `TASKDEFINITION_SPEC.md` (produces `tasks.csv` and `ground_truth.csv`), `LLM_SPEC.md` (Strategy A), `AGENT_SPEC.md` (Strategy B).
>
> Where the draft is ambiguous, the v1 choice is marked **[ASSUMPTION]** and listed in [Open questions](#11-open-questions).

---

## 1. Purpose

For every task, model, strategy, and strategy variant:

1. Take the 5 runs and choose the **representative** answer.
2. Compare it with the ground truth, property by property.
3. Aggregate the scores across tasks and report them by model, strategy, variant, and difficulty.

The evaluation never calls a model or the search tool. It works only on files, so it can be re-run any number of times at no cost.

## 2. Scope

**In scope:** validating inputs, value normalization and matching, representative selection, per-property and per-task scoring, aggregated metrics, A vs. B comparison, output tables.

**Out of scope:** running strategies, generating tasks, defining the sustainability composite formula (raw components are aggregated here; the formula is plugged in later, see §6.4).

## 3. Inputs

| File | Produced by | Used for |
|---|---|---|
| `tasks.csv` | Task definition | Task metadata: H, strata, difficulty |
| `ground_truth.csv` | Task definition | Hidden values |
| `results.csv` (one or more) | Strategy A, Strategy B | Model predictions and usage |

All `results.csv` files must reference the same `task_set_version` as `tasks.csv`. The evaluation fails otherwise.

### 3.1 `results.csv` contract (both strategies must write this)

One row per **run** (so 5 rows per task × model × variant).

| Column | Type | Description |
|---|---|---|
| `task_set_version` | str | Must match `tasks.csv` |
| `task_id` | str | |
| `strategy` | str | `A` or `B` |
| `variant` | str | Identifies one distinct configuration. Strategy A: template id, including the number of examples for few-shot templates (e.g. `base_zero_shot`, `base_few_shot_3`). Strategy B: agent config id (e.g. `waves1`, `AGENT_SPEC.md` §9) |
| `model` | str | Model identifier |
| `run_index` | int | 1..5 |
| `status` | str | `ok`, or a failure status (`parse_error`, `retrieval_failed`, `timeout`, …) |
| `predictions` | JSON | `{prop_key: value or null}` with exactly the keys of H. `null` = abstention |
| `evidence` | JSON | Strategy B only: `{prop_key: [url, ...]}`. Empty for A |
| `pfe_tokens` | int | Prompt formulation effort of the template(s) used |
| `tokens_in`, `tokens_out` | int | Totals for the run (all LLM calls) |
| `tokens_reasoning` | int | Reasoning tokens, a part of `tokens_out`; empty when unknown (`METRICS.md` §4.1) |
| `llm_calls` | int | LLM calls in the run (`N_calls` of the EIS, `METRICS.md` §4.2) |
| `search_calls` | int | 0 for Strategy A |
| `latency_s` | float | Wall-clock time of the run |
| `energy_kwh`, `co2e_kg` | float | Measured with CodeCarbon (local models) or estimated with EcoLogits (API models), see `LLM_SPEC.md` §6.5 |
| `cost_usd` | float | LLM cost from the model's `pricing`; empty without a price (`METRICS.md` §4.5) |
| `prompt_version` | str | Version of the prompt files |
| `timestamp` | str | ISO 8601 |

A configuration is identified by `(task_id, strategy, variant, model)` and must have exactly `r = 5` rows. A failed run still has a row: its `predictions` are all `null` and its `status` explains why.

## 4. Pipeline

```
results.csv ─┐
tasks.csv ───┼─► 1. validate ─► 2. normalize ─► 3. select representative ─► 4. score
ground_truth ┘                                                                 │
                                       6. write outputs ◄── 5. aggregate ◄─────┘
```

## 5. Steps

### 5.1 Validate
- Schemas and `task_set_version` match.
- Every configuration has exactly 5 runs with `run_index` 1..5 and no duplicates.
- Every `predictions` object has exactly the H keys of its task. Missing keys are treated as `null` and logged; extra keys are dropped and logged.
- Configurations with missing runs are reported and **excluded** from metrics (not silently scored on fewer runs). **[ASSUMPTION]** To keep the A vs. B comparison paired (§6.7), an excluded `(task_id, model)` is excluded for both strategies in that comparison.

### 5.2 Normalize values
Both predictions and ground truth go through the same normalizer, chosen by `value_type` from `ground_truth.csv`:

| Type | Normalization |
|---|---|
| `string` | Unicode NFKC, case-fold, trim, collapse whitespace, strip accents and surrounding punctuation; optional alias table per property (e.g. `"USA"` → `"united states"`) |
| `integer` / `float` | Parse numbers from text (`"2,459 ft"` → `2459`); optional unit conversion per property |
| `boolean` | Map `yes/no/true/false/1/0` |
| `date` | Parse to ISO date; compare at the granularity of the ground truth |

Values that can't be parsed into the expected type are kept as-is and will simply not match.

Abstention markers (`null`, `""`, `"unknown"`, `"n/a"`, `"not found"`, …; configurable list) are normalized to `null`.

### 5.3 Matching
`match(pred, truth, prop_key)` returns `true` / `false`, in two modes. Both are always computed and reported.

- **Strict:** normalized values are equal (numbers: exactly equal after parsing).
- **Tolerant:**
  - strings: normalized edit similarity ≥ `string_threshold` (config, e.g. 0.9), or one contains the other when the property is in `containment_props` (config);
  - numbers: within `abs_tol` or `rel_tol` set per property (e.g. elevation ±1%, coordinates ±0.01°);
  - dates: same rules as strict.

Tolerances are defined per `prop_key` (or per property name) in the evaluation config and recorded in the manifest.

### 5.4 Select the representative run

From the 5 runs of a configuration, choose **one** run as the representative. Its predictions are what gets compared to the ground truth.

**Rule: the ground truth is never used to choose.** The choice must be something a user could reproduce without knowing the answers. (An oracle "best of 5" is reported separately as an upper bound, §6.3.)

**Default method: medoid (most consensual run).**
1. For each pair of runs `(i, j)`, compute agreement: the number of H properties where both runs give matching values (using the **tolerant** matching of §5.3), or both abstain.
2. Each run's consensus score = sum of its agreement with the other 4 runs.
3. The representative is the run with the highest consensus score.
4. Ties are broken by: fewer abstentions → status `ok` over failures → lower `run_index`.

**Alternative method (config `selection: majority_vote`):** build a synthetic answer property by property, taking the value that appears in at least 3 of the 5 runs, or `null` if no value reaches that. This is the per-property majority vote mentioned in the draft. It doesn't correspond to a single real run, so its usage metrics are the totals of all 5 runs. For Strategy B, `evidence[p]` is the union of the sources cited by the runs that voted for the winning value.

Both methods are implemented; `medoid` is the default **[ASSUMPTION]**, and the method used is recorded in every output.

**Stability.** Alongside the selection, compute for each configuration:
- `agreement_rate` = mean pairwise agreement across the 10 run pairs ÷ |H|;
- `distinct_answers` per property.

This shows how consistent a model is across runs, independent of correctness.

### 5.5 Score (per property)
For each H property of the representative:

| Outcome | Condition |
|---|---|
| `correct` | prediction not null and matches ground truth |
| `wrong` | prediction not null and does not match |
| `abstained` | prediction null |

### 5.6 Score (per task)
For a configuration with representative predictions over H:

- `n_correct`, `n_wrong`, `n_abstained`
- **Precision** = `n_correct / (n_correct + n_wrong)` — undefined (empty) if the model abstained on everything
- **Recall** = `n_correct / |H|`
- **F1** = harmonic mean of precision and recall (0 if both are 0; empty if precision is undefined and recall is 0 → reported as 0, see §11)
- **Accuracy** = `n_correct / |H|` (proportion of hidden properties recovered; equal to recall in this setting, kept for readability)
- **Exact match** = 1 if every property in H is correct, else 0
- **Abstention rate** = `n_abstained / |H|`
- **Error rate** = `n_wrong / |H|` (confidently wrong answers, i.e. hallucinations)

Each metric is computed in strict and tolerant mode.

## 6. Aggregated metrics

### 6.1 Accuracy family
Reported for every grouping in §6.5:
- **Micro** averages: sum `n_correct`, `n_wrong`, `n_abstained` over all properties in the group, then compute precision, recall, F1. This weights tasks by |H|.
- **Macro** averages: mean of per-task metrics. This weights every task equally.
- Exact match rate, abstention rate, error rate.
- 95% confidence intervals by bootstrap over tasks (`n_bootstrap`, config; seeded).

### 6.2 Effort
- `pfe_tokens` of the variant (same for all tasks of a variant).
- Mean prompt tokens actually sent per task (template + instantiated task + evidence for B).

### 6.3 Upper bound and stability
- **Oracle best-of-5:** for each configuration, the run with the highest F1 against the ground truth. Reported only as an upper bound, never as the main result. The gap between the representative and the oracle shows how much is lost to run variance.
- **Agreement rate** (mean of §5.4 over configurations).

### 6.4 Sustainability
The cost of an answer includes **all 5 runs**, since all 5 are needed to select the representative.

Per configuration: total `tokens_in`, `tokens_out`, `search_calls`, `latency_s`, `energy_kwh`, `co2e_kg` over the 5 runs.

Per group: mean per task, and **efficiency ratios**: correct properties per 1k tokens, per kWh, and per search call.

The composite sustainability score is computed by a pluggable function `sustainability_score(components) -> float`. v1 ships a placeholder that returns empty until the formula is agreed. **[ASSUMPTION]**

### 6.5 Groupings
Every metric is reported for:
- `model × strategy × variant` (main table);
- the above × `node_stratum`, × `H_stratum`, × `G_stratum`, × `size_H`, × `size_G`;
- the above × **difficulty bins** (quantile bins of `difficulty`; number of bins in config, default 4);
- per `prop_key` (which properties are hardest to recover).

### 6.6 Strategy B extras
When `evidence` is present:
- **Citation rate:** fraction of non-null predictions with at least one source.
- **Grounded accuracy** and **retrieval coverage** need the retrieved text and are computed from the agent's run logs (`runs.jsonl`, see `AGENT_SPEC.md` §10). The agent always logs the retrieved results, so these metrics are required for every Strategy B configuration.

### 6.7 Comparison A vs. B
Comparisons are **paired**: the same task and model under the two strategies.
- For each model: Strategy B vs. Strategy A's best variant (best by macro F1 over all tasks; the choice is recorded) and vs. each A variant.
- Difference in macro F1, recall, abstention rate, error rate, with bootstrap CIs.
- Wilcoxon signed-rank test on per-task F1 differences.
- Spearman correlation between difficulty and F1, per strategy and model. This tests whether the difficulty metric predicts how hard tasks are.

## 7. Outputs

Written to `<output_dir>/` (e.g. `eval/airports/v1/<eval_id>/`).

| File | Grain | Content |
|---|---|---|
| `representatives.csv` | configuration | Selected `run_index` (or `majority_vote`), consensus scores, agreement rate, selection method |
| `property_scores.csv` | configuration × prop_key | Normalized prediction, normalized truth, outcome (strict and tolerant) |
| `task_scores.csv` | configuration | All §5.6 metrics, task metadata (strata, sizes, difficulty), summed usage over 5 runs, oracle F1 |
| `summary.csv` | group | All §6 metrics with CIs, one row per group in §6.5, `grouping` column says which |
| `comparison.csv` | model × A variant | Paired A vs. B differences, CIs, p-values |
| `eval_manifest.json` | — | Config, tolerances, selection method, input file hashes, code version, validation report (excluded configurations, dropped keys, parse failures) |

Optional (flag `--plots`): F1 vs. difficulty bins per strategy and model; abstention vs. error rate per model; representative vs. oracle F1.

## 8. Configuration

```yaml
inputs:
  tasks: tasks/airports/v1/tasks.csv
  ground_truth: tasks/airports/v1/ground_truth.csv
  results:
    - results/airports/v1/strategy_a/results.csv
    - results/airports/v1/strategy_b/results.csv
runs_per_config: 5
selection: medoid            # medoid | majority_vote
matching:
  string_threshold: 0.9
  abstention_markers: ["unknown", "n/a", "not found", "none", ""]
  per_property:
    Airport.elevation: {abs_tol: 10}
    Airport.latitude:  {abs_tol: 0.01}
    Airport.longitude: {abs_tol: 0.01}
  aliases: aliases/airports.yaml
aggregation:
  difficulty_bins: 4
  n_bootstrap: 1000
  seed: 42
output_dir: eval/airports/v1/
```

CLI: `eval run --config configs/eval_airports.yaml`

## 9. Testing and acceptance

- **Normalizer tests:** a table of `(raw, value_type, expected_normalized)` cases per type, including units, accents, abstention markers.
- **Selection tests:** hand-built sets of 5 runs with a known medoid, including ties and all-abstain cases; majority vote with and without a 3-of-5 majority.
- **Scoring tests:** tasks with known correct/wrong/abstained counts → expected precision, recall, F1, exact match.
- **No-oracle test:** changing the ground truth must never change which run is selected.
- **Determinism:** same inputs and seed → byte-identical outputs.

Acceptance: the evaluation runs on the airports task set with both strategies, every configuration is either scored or listed as excluded with a reason, and `summary.csv` has a row for every model × strategy × variant.

## 10. Notes on interpretation

- **Recall vs. precision** is where the strategies are expected to differ: Strategy A tends to abstain (high precision, low recall); Strategy B should answer more often. Always report abstention and error rates next to F1 so a higher F1 can be explained.
- **Strict vs. tolerant** gaps show how much of the error is formatting (units, spelling) rather than wrong knowledge. Large gaps suggest improving the normalizer or the output format instructions, not the strategy.

## 11. Open questions

1. **Selection method:** medoid run (v1 default) or per-property majority vote as in the draft's `Aggregate`? They give different results when runs disagree on some properties but not others.
2. **Should abstention count as agreement** in the medoid computation? v1 says yes; that favors a run that abstains along with the majority.
3. **F1 when the model abstains on everything:** report 0 (v1) or exclude the task from the precision average?
4. **Tolerances** per property type, and whether tolerant or strict matching is the headline metric in the paper.
5. **Strategy A vs. B comparison:** compare B against the best A template, against each template, or against a fixed template?
6. **Incomplete configurations** (fewer than 5 valid runs): exclude (v1) or score on the available runs?
7. **Sustainability composite formula** (§6.4).
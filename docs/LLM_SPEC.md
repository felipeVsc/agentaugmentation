# SPEC: LLM Strategy (Strategy A — Static Parametric Pathway)

> Component spec for `SimpleLLMRoutine` (Algorithm 2) of the paper *"Data Augmentation in Property Graphs using agentic AI"* (draft).
>
> The LLM receives a task as a prompt and answers from its own knowledge, with no external tools. Each task is run 5 times per model and template, and the runs are written to `results.csv`.
>
> Related specs: `TASKDEFINITION_SPEC.md` (produces `tasks.csv`), `AGENT_SPEC.md` (Strategy B), `EVALUATION_SPEC.md` (reads `results.csv` and defines its format).
>
> Where the draft is ambiguous, the v1 choice is marked **[ASSUMPTION]** and listed in [Open questions](#11-open-questions).

---

## 1. Purpose

For every task in `tasks.csv`, every model, and every prompt template:

1. Build the prompt from the template and the task.
2. Send it to the model 5 times.
3. Parse each response into predictions for the hidden properties H.
4. Write one row per run to `results.csv`.

This component does **not** select a representative run or score anything. That is done by the evaluation.

## 2. Scope

**In scope:** reading tasks, rendering prompts from templates, calling models (API and local), parsing responses, retries, caching, usage and energy measurement, writing `results.csv` and raw logs.

**Out of scope:**
- The wording of the prompts. Templates are written separately; this spec defines only the contract a template must follow (§5).
- Web search or any other tool (that's Strategy B).
- Aggregating runs and scoring (evaluation).

**Hard rule:** this component never reads `ground_truth.csv`.

## 3. Inputs

| Input | Description |
|---|---|
| `tasks.csv` | From the task definition step. Uses `task_id`, `task_set_version`, `node_type`, `H`, `G` |
| Templates | Versioned prompt files (§5) |
| Few-shot examples | Optional file used by few-shot templates (§5.4) |
| Config | Models, templates, runs, generation parameters (§9) |

## 4. Execution flow

```
for model m in M:
  for template t in T:
    for task in tasks.csv:
      prompt ← render(t, task)                       # §5
      for run_index in 1..5:
        response ← call(m, prompt, run_index)         # §6
        predictions, status ← parse(response, task)   # §7
        write row to results.csv                      # §8
```

The loops are only the logical order. Calls run concurrently (§6.3), and the output order in the file doesn't matter.

## 5. Prompt templates

### 5.1 Template variants
The draft defines two template styles combined with three prompting techniques, giving six templates:

| | zero-shot | few-shot | chain-of-thought |
|---|---|---|---|
| **base** | `base_zero_shot` | `base_few_shot` | `base_cot` |
| **compact** | `compact_zero_shot` | `compact_few_shot` | `compact_cot` |

The template id is written to the `variant` column of `results.csv`. New templates can be added through the config without code changes.

### 5.2 Template format
Each template is a text file with an optional system part and a user part, separated by a marker line:

```
### SYSTEM
<system instructions>
### USER
<user message with placeholders>
```

File names carry a version (e.g. `templates/base_zero_shot.v1.txt`). The version is written to `prompt_version` in `results.csv`. A template file is never edited after it is used in a run; changes create a new version.

### 5.3 Placeholders
A template may use these placeholders, filled by the renderer:

| Placeholder | Filled with |
|---|---|
| `{node_type}` | e.g. `Airport` |
| `{known_properties}` | G, rendered as described below |
| `{target_properties}` | H, as the list of property labels to predict |
| `{output_format}` | The required JSON shape with exactly the H labels as keys (§5.5) |
| `{examples}` | Few-shot examples (§5.4); empty for other templates |

The renderer fails if the template contains an unknown placeholder or if a required one (`{known_properties}`, `{target_properties}`, `{output_format}`) is missing.

**Rendering G (`{known_properties}`).** Each G entry is written as one line with a human-readable label built from its path, then its value:

```
- Airport name: Guarulhos International
- City (the Airport is IN this City) name: São Paulo
- Country (the City is IN this Country) name: Brazil
```

Labels are generated from `path` and `property_name` in `tasks.csv`. The mapping `label → prop_key` is kept in memory so the response can be mapped back. **[ASSUMPTION]** One line per entry, in the order they appear in the task; the format (`list`, `json`) can be chosen per template in the config.

**Rendering H (`{target_properties}`).** Each H property is given a label (e.g. `elevation`). If two H properties have the same name, the label includes the path to keep them distinct.

### 5.4 Few-shot examples
- Examples are complete tasks **with answers**, stored in a separate file (`examples/<dataset>.v1.json`).
- Example nodes must **not** appear in `tasks.csv` (neither as task node nor as a G element). The runner checks this at startup and fails if they do.
- You will test the following few-shots: 1, 3 and 5-shots.

### 5.5 Required output
The model must answer with a JSON object whose keys are exactly the H labels, and whose values are the predicted value or `null` when it doesn't know:

```json
{"elevation": "2459 ft", "runways": null}
```

- `null` means abstention, and templates must state that abstaining is allowed and preferred to guessing.
- For chain-of-thought templates, the model writes its reasoning first and then the JSON inside a fenced block ```` ```json … ``` ````. Only the last JSON block is parsed.

### 5.6 Prompt Formulation Effort (PFE)
- `pfe_tokens` = token count of the template text as written by a human, **with placeholders removed** (system + user parts, plus the few-shot examples text for few-shot templates).
- Use the tokenizer for each model. The idea is to compare the models, and how much tokens they use also.
- Computed once per template and written on every row.
- Count the number of tokens from the few-shots examples also. 

## 6. Model calls

### 6.1 Provider abstraction
All models are called through one interface:

```
generate(model_config, system, user, run_index) -> Response
Response: text, tokens_in, tokens_out, latency_s, finish_reason, raw
```

Implementations for API providers (e.g. OpenAI, Anthropic, Google) and local models (e.g. Ollama or vLLM). Token counts come from the provider when available, otherwise from the model's tokenizer.

### 6.2 Generation parameters
- Set per model in the config: `temperature`, `top_p`, `max_tokens`.
- **Temperature must be above 0** (default 0.7 **[ASSUMPTION]**). With temperature 0, the 5 runs would be (nearly) identical and the repetition would measure nothing.
- No JSON/structured-output mode by default, so that all models are tested under the same conditions.

### 6.3 Concurrency and rate limits
- Calls run concurrently with a per-provider limit (`max_concurrency`) and requests-per-minute limit.
- Retries with exponential backoff on rate-limit, timeout, and server errors (`max_retries`). After the last retry, the run is recorded with a failure status (§7.2); it is never silently dropped.

### 6.4 Caching and resuming
- Each call is cached on disk with the key `(model, generation params, system, user, run_index)`. Including `run_index` keeps the 5 runs separate.
- On restart, the runner skips `(task_id, model, template, run_index)` combinations already in `results.csv`, so an interrupted batch resumes where it stopped.
- `--no-cache` forces fresh calls.

### 6.5 Energy measurement
- Local models: CodeCarbon tracks each call; `energy_kwh` and `co2e_kg` are written per run.
- API models: Use EcoLogits to track the energy comsuption of each model. If the model version is not available at EcoLogits, use the closest one of the same provider. 

## 7. Parsing

### 7.1 Procedure
1. Extract the JSON object: the last fenced `json` block if present, otherwise the last `{…}` object in the text.
2. Parse it. Accept common slips: trailing commas, single quotes.
3. Map labels back to `prop_key` using the mapping from rendering.
4. Keys of H that are missing → `null`. Keys not in H → dropped. Both are logged.
5. Values are kept **as returned** (as strings or numbers). Normalization and unit handling are done by the evaluation, not here.

### 7.2 Failures
| Case | Action | `status` |
|---|---|---|
| No JSON found or invalid JSON | One repair call: resend with the response and a short instruction to return only valid JSON in the required format. If it still fails, all predictions `null` | `parse_error` |
| Response cut off by `max_tokens` | Parse what is there; if it fails, as above | `truncated` |
| Call failed after retries | All predictions `null` | `api_error` / `timeout` |
| Valid | — | `ok` |

The repair call's tokens are added to the run's usage. Whether a repair was needed is recorded in the run log.

## 8. Outputs

### 8.1 `results.csv`
Follows the contract defined in `EVALUATION_SPEC.md` §3.1, with one row per run. Strategy A-specific values:

| Column | Value |
|---|---|
| `strategy` | `A` |
| `variant` | Template id, e.g. `base_cot`. Few-shot templates include the number of examples (§5.4), e.g. `base_few_shot_3` |
| `run_index` | 1..5 |
| `predictions` | `{prop_key: value or null}`, exactly the keys of H |
| `evidence` | `{}` |
| `search_calls` | `0` |
| `pfe_tokens` | §5.6 |
| `prompt_version` | Template file version |

### 8.2 `runs.jsonl` (raw log)
One JSON line per run: `task_id`, model, template, `run_index`, the rendered system and user prompts, raw response text, parsed JSON, the `label → prop_key` mapping, parse events, repair call (if any), usage, status, timestamp.

This is what makes the results auditable: any number in `results.csv` can be traced back to the exact prompt and response.

### 8.3 `run_manifest.json`
Config, model versions as reported by the providers, template files and their hashes, `task_set_version`, code version (git commit), start/end time, counts by status.

## 9. Configuration

```yaml
inputs:
  tasks: tasks/airports/v1/tasks.csv
  examples: examples/airports.v1.json
runs_per_task: 5
templates:
  - id: base_zero_shot
    file: templates/base_zero_shot.v1.txt
    context_format: list
  - id: base_few_shot
    file: templates/base_few_shot.v1.txt
    n_examples: 3
  # ...
models:
  - id: gpt-x
    provider: openai
    params: {temperature: 0.7, max_tokens: 1024}
    max_concurrency: 8
  - id: llama-x
    provider: ollama
    params: {temperature: 0.7, max_tokens: 1024}
    max_concurrency: 1
    measure_energy: true
tokenizer_for_pfe: cl100k_base
max_retries: 3
timeout_s: 60
output_dir: results/airports/v1/strategy_a/
```

CLI: `strategy-a run --config configs/strategy_a_airports.yaml` (options: `--models`, `--templates`, `--limit N` for a quick trial, `--no-cache`).

## 10. Testing and acceptance

- **Renderer tests:** each template renders for a sample task; unknown/missing placeholders fail; labels map back to the right `prop_key`, including duplicated property names.
- **Parser tests:** a set of real-looking responses (clean JSON, JSON in prose, CoT with fenced block, trailing commas, missing keys, extra keys, no JSON) with expected predictions and status.
- **Mock provider:** returns scripted responses, so the whole flow runs offline in CI.
- **Leakage test:** the runner fails if few-shot example nodes appear in the task set; a test asserts `ground_truth.csv` is never opened.
- **Resume test:** interrupting and restarting produces the same `results.csv` without duplicate rows.

Acceptance: for a small task subset, every `(task, model, template)` has exactly 5 rows in `results.csv`, every row passes the evaluation's input validation, and every row has a matching entry in `runs.jsonl`.

## 11. Open questions

1. **Temperature** for the 5 runs: one value for all models, or tuned per model? This affects how different the runs are.
2. **Few-shot examples:** from the same dataset (other nodes) or from a different domain? Same examples for every task, or chosen per task (e.g. same node stratum)?
3. **Structured output mode:** allowed when a model supports it, or disabled for all to keep conditions equal?
4. **PFE:** should the few-shot examples count as human effort? Should the system prompt count?
5. **Node identity in the prompt:** if G doesn't include the node's name or code, should the template still identify the node? (Same question as in the task definition spec; the answer must be the same for Strategy A and B.)
6. **Reasoning models:** for models with built-in hidden reasoning, are those reasoning tokens counted in `tokens_out` and sustainability?
7. **Repair call:** allowed (v1), or does a parse failure count directly as abstention? The repair adds cost and may help some models more than others.
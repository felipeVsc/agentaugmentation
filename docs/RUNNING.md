# What you need to run this repo (status: 2026-10-23)

This file lists what has to be prepared before each step can run: data, prompts, keys and configs. The design is in the specs (`docs/SPEC.md` and the component specs); this file is only about running.

## Status

| Step | Code | Tested | Can run today? |
|---|---|---|---|
| 1. Task definition (`src/taskdef/`) | Done | 28 offline tests pass (toy graph); not yet run on the real Neo4j graph | Yes, with Neo4j access |
| 2. Strategy A, LLM only (`src/llmcore/`, `src/strategy_a/`) | Done; models run locally through **Ollama** | 25 offline tests pass, plus a real run with `llama3.2:1b` | Yes, once `tasks.csv` exists (few-shot variants also need the examples file, §3.3) |
| 3. Strategy B, search agent (`src/strategy_b/`) | Done: LangGraph, Ollama, Tavily | 11 offline tests; real run with qwen3:4b and a fake search tool | Yes, with `TAVILY_API_KEY` in `.env` (§4) |
| 4. Evaluation (`src/evaluation/dolap.py`) | Published and standard scoring for the DOLAP replication | Reproduces Table 3 | Through `notebooks/dolap2026_ollama.ipynb` |

---

## 1. Environment

- Python 3.11 or newer.
- The project venv:
  ```bash
  python3 -m venv venv
  venv/bin/pip install -e ".[dev]"
  ```
- For Strategy A with local models:
  ```bash
  venv/bin/pip install -e ".[dev,local]"   # codecarbon, tiktoken, tokenizers; registers `strategy-a`
  ```
  plus [Ollama](https://ollama.com) (`ollama serve`, or the Ollama app) and the models: `ollama pull qwen3:4b-instruct`.
  No API keys are needed. `.[api]` installs the SDKs for API models if you need them again.
- A `.env` file in the repo root (git-ignored) with the Neo4j settings for step 1:
  ```bash
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=...
  # NEO4J_DATABASE=neo4j
  # OLLAMA_HOST=http://localhost:11434    # only if Ollama runs elsewhere
  ```
  Credentials are read only from the environment or `.env`; they are never written to configs, manifests or logs.

---

## 2. Step 1 — Task definition

### What you need
1. **The graph in Neo4j**, reachable with the credentials in `.env`. Run this once per dataset; you'll need 3 datasets.
2. **A dataset config**, copied from `configs/airports.yaml`:

   | Field | What to set |
   |---|---|
   | `dataset.name`, `dataset.node_type` | Dataset name and the Neo4j label of the nodes whose properties are hidden (e.g. `Airport`) |
   | `dataset.node_id_property` | Optional: a domain id such as `iata`, so task ids and logs are readable |
   | `properties.exclude_properties` | Technical properties that must never appear in a task (`id`, embeddings, timestamps, ...) |
   | `properties.never_hide` | Properties that may be given (G) but never asked (H) |
   | `properties.leakage_pairs` | Properties that reveal each other, e.g. `[[Airport.iata, Airport.icao]]`, `[[Airport.tz, Airport.timezone]]`, `[[Airport.latitude, Airport.location]]` (the leaks found in `test_results.csv`) |
   | `sampling.max_n`, `max_H`, `max_G`, `seed` | Task-set size; see the open decision in §6 |
   | `task_set_version`, `output_dir` | One new version and directory per task set; a task set is never overwritten |

3. **Decide the leakage pairs and exclusions before looking at any results.** They are recorded in the manifest.

### Run
```bash
venv/bin/tasks check-connection --config configs/airports.yaml   # lists labels and relationship types
venv/bin/tasks build --config configs/airports.yaml
```

### Output, in `output_dir` (e.g. `tasks/airports/v1/`)
| File | Read by |
|---|---|
| `tasks.csv` | Strategies A and B, and the evaluation |
| `ground_truth.csv` | **The evaluation only.** The strategies refuse to open it |
| `manifest.json`, `intermediate/` | Audits and the paper |

---

## 3. Step 2 — Strategy A (LLM only)

### 3.1 Inputs you have to provide

| # | Input | Status | Where |
|---|---|---|---|
| 1 | `tasks.csv` from step 1 | Needs step 1 | `tasks/<dataset>/<version>/tasks.csv` |
| 2 | **Prompt templates**: 6 files (§3.2) | Drafts written (`*.v1.txt`); revise the wording before the real run, as a `.v2` | `templates/` |
| 3 | **Few-shot examples**: at least 5 solved tasks per dataset (§3.3) | **Missing**; needs real data | `examples/<dataset>.v1.json` |
| 4 | **Strategy A config** with templates and models (§3.4) | Done for airports, Ollama models | `configs/strategy_a_airports.yaml` |
| 5 | Ollama running, with the models pulled | Yours | `ollama serve`, `ollama pull <model>` |

### 3.2 Prompt templates

There is one text file per template. The version goes in the file name (`base_zero_shot.v1.txt` → `prompt_version = v1`). **Never edit a template after it has been used in a run**; create `.v2` instead.

```text
### SYSTEM
You complete missing property values of entities in a property graph, using only your own knowledge.
### USER
Known properties of a {node_type} and of related elements:
{known_properties}

Give the real-world values of these properties of the same {node_type}:
{target_properties}

If you don't know a value, use null; abstaining is better than guessing.
Answer with only this JSON object:
{output_format}
### EXAMPLE
(optional: how one few-shot example is written; uses {index}, {node_type}, {known_properties}, {target_properties}, {answer})
```

| Placeholder | Required | Filled with |
|---|---|---|
| `{known_properties}` | Yes | G, one line per entry: `- City (the Airport is IN this City) name: São Paulo` (or JSON with `context_format: json`) |
| `{target_properties}` | Yes | H labels, one per line (`- elevation`) |
| `{output_format}` | Yes | `{"elevation": "<value or null>", "runways": "<value or null>"}`, with exactly the H labels |
| `{node_type}` | No | e.g. `Airport` |
| `{examples}` | Few-shot templates only | The rendered examples |

Rules:
- Unknown or missing placeholders make the run fail at startup.
- Write `{{` and `}}` for literal braces.
- For chain-of-thought templates, tell the model to reason first and then put the JSON in a ```` ```json ```` block. The parser takes the last JSON block.

**Templates to write.** These follow `LLM_SPEC.md` §5.1. The published paper used 9 variants.

| Template id | File | Few-shot sizes |
|---|---|---|
| `base_zero_shot` | `templates/base_zero_shot.v1.txt` | — |
| `base_few_shot` | `templates/base_few_shot.v1.txt` | 1, 3, 5 → variants `base_few_shot_1`, `_3`, `_5` |
| `base_cot` | `templates/base_cot.v1.txt` | — (add a few-shot CoT template if you want the published CoT + 1/3/5-shot variants) |
| `compact_zero_shot` | `templates/compact_zero_shot.v1.txt` | — |
| `compact_few_shot` | `templates/compact_few_shot.v1.txt` | 1, 3, 5 |
| `compact_cot` | `templates/compact_cot.v1.txt` | — |

This gives 10 variants. Every variant multiplies the cost, so check with `strategy-a estimate` (§3.5) before running.

**PFE** is counted for each template × model. It covers the template text without placeholders (system and user) plus the rendered few-shot examples, counted with the model's `tokenizer`, or with `pfe_tokenizer` when the model has none.

### 3.3 Few-shot examples

```json
{
  "version": "v1",
  "examples": [
    {
      "node_id": "SBGR",
      "node_type": "Airport",
      "G": [
        {"prop_key": "Airport.name", "property_name": "name", "element_type": "Airport",
         "element_id": "SBGR", "path": "Airport", "value": "Guarulhos International"},
        {"prop_key": "Airport-[IN]->City.name", "property_name": "name", "element_type": "City",
         "element_id": "c_45", "path": "Airport-[IN]->City", "value": "São Paulo"}
      ],
      "H": [{"prop_key": "Airport.elevation", "property_name": "elevation",
             "element_type": "Airport", "path": "Airport"}],
      "answers": {"Airport.elevation": 2459}
    }
  ]
}
```

- `G` and `H` have the same shape as in `tasks.csv`. The easiest way to build examples is to copy a few rows from a task build and fill `answers` from the graph.
- **The example nodes must not appear in `tasks.csv`**, either as a task node or as a G element. The run fails if they do. Pick nodes that the sampler didn't select, or build the examples from a separate task set.
- 5-shot needs at least 5 examples. The first N examples are used for every task.
- Each dataset needs its own file.

### 3.4 Strategy A config (Ollama)

`configs/strategy_a_airports.yaml` is ready. It holds the task file, the templates (few-shot commented out until the examples exist), defaults shared by all models, and the models:

```yaml
model_defaults:
  provider: ollama
  params: {temperature: 0.7, max_tokens: 1024}
  max_concurrency: 1                 # one call at a time (required by codecarbon)
  timeout_s: 600                     # the first call loads the model
  energy: {method: codecarbon, country_iso_code: BRA}
  options: {num_ctx: 8192, keep_alive: 30m, seed: 0}

models:
  - id: qwen3-4b-instruct            # written to results.csv
    model: qwen3:4b-instruct         # Ollama tag (`ollama list`)
    tokenizer: hf:Qwen/Qwen3-4B-Instruct-2507
  - id: qwen3-14b
    model: qwen3:14b
    options: {think: false}          # merged with model_defaults.options
    enabled: false                   # skipped unless named with --models
```

Ollama options (all optional, under `options:`):

| Option | Default | What it does |
|---|---|---|
| `num_ctx` | 8192 | Context window. **Ollama silently cuts the start of a longer prompt**, so `check` warns when a prompt may not fit. Keep it fixed per model: changing it reloads the model |
| `think` | not sent | `false` turns thinking off on thinking models (qwen3, deepseek-r1, gpt-oss) |
| `seed`, `seed_per_run` | 0, true | Run r uses seed `seed + r`: the 5 runs differ (temperature > 0), and each can be reproduced exactly |
| `keep_alive` | `30m` | How long the model stays loaded between calls |
| `base_url` | `$OLLAMA_HOST` or `http://localhost:11434` | Another machine, e.g. a GPU server |
| `tokens_in_from` | `tokenizer` if the model sets `tokenizer`, else `reported` | Where `tokens_in` comes from: Ollama's `prompt_eval_count` (always kept in `runs.jsonl`) or the model's tokenizer |

Other Ollama generation options (`top_k`, `repeat_penalty`, `min_p`, `format`) go under `params.extra`.

Things to know:
- **Model version.** Each run records the model tag, its digest and its quantization (e.g. `llama3.2:1b@baf6a787fdff (Q8_0)`). The manifest also records the parameter count, the context length and the Ollama version. Always report the quantization in the paper: a Q4 and an FP16 model are different models.
- **Tokenizer.** For exact PFE and `tokens_in` counts, set `tokenizer: hf:<repo>` to the original model's tokenizer. It downloads from Hugging Face; Llama and Mistral repos are gated and need a Hugging Face login. Without it, `cl100k_base` gives an approximation.
- **Energy.** `codecarbon` measures the whole machine, one call at a time. **On this Mac (Apple M4) CodeCarbon finds no GPU and only estimates CPU and RAM power from load**, so the values are flagged `estimated: true, gpu_measured: false` and undercount: Ollama runs on the Apple GPU. For paper numbers, run on a Linux machine with an NVIDIA GPU, where CodeCarbon reads the GPU power. Alternatively, `energy: {method: ecologits, ecologits_provider: huggingface_hub, ecologits_model: <closest HF model>}` gives a datacenter-style estimate comparable to the published paper's.
- **No batch API.** `batch` commands report that Ollama has none; use `run`.
- API providers (OpenAI, Anthropic, Gemini) are still supported if needed: `provider: openai | anthropic | google`, with keys in `.env`.

### 3.5 Commands

```bash
ollama serve                                                             # if the Ollama app isn't running
strategy-a check    --config configs/strategy_a_airports.yaml            # server, models pulled, digest, quantization, prompt fits num_ctx, PFE
strategy-a check    --config ... --ping                                  # plus one tiny real call per model
strategy-a render   --config ... --variant base_cot                      # print one rendered prompt
strategy-a estimate --config ...                                         # number of calls and tokens

strategy-a run --config ... --limit 2 --models llama3.2-1b              # small trial first
strategy-a run --config ...
```

Useful flags:
- `--models a,b` and `--templates t1,t2` select models and templates or variants.
- `--retry-failed` reruns rows with status `api_error` or `timeout`.
- `--no-cache` ignores cached answers.

An interrupted run resumes where it stopped.

### 3.6 Output, in `output_dir`
| File | Content |
|---|---|
| `results.csv` | One row per run, in the `EVALUATION_SPEC.md` §3.1 format |
| `runs.jsonl` | For each run: the exact prompts, the raw answer, the parse events, the repair call if any, and usage |
| `run_manifest.json` | Config, template hashes, PFE per variant and model, model versions reported by the providers, code version, status counts |
| `cache/` | Every model answer. Rerunning from the cache costs nothing |
| `batches/` | Batch state and request files |

---

## 4. Step 3 — Strategy B (single-turn search agent)

One try per run, as in `AGENT_SPEC.md`:

```
START → plan ──→ retrieve ──→ synthesize → END
          └─(plan invalid)→ END   └─(all searches failed)→ END
```

1. **Plan** (1 LLM call): the model writes up to `max_subqueries` search queries, each targeting hidden properties. A default query is added for any property no query targets.
2. **Retrieve** (1 step): the queries run in parallel on Tavily. Blocked domains are excluded in the request and filtered again locally. Results are deduplicated and cut to `max_evidence_tokens`.
3. **Synthesize** (1 LLM call): the model answers from the numbered evidence. In `evidence_only`, a value without a valid citation becomes null.

Each LLM step gets at most one repair call. There is no loop.

LangGraph only wires the steps. The LLM calls go through the same `llmcore` caller as Strategy A, so caching, seeds, token counts and energy are identical, and `results.csv` has the same columns (`strategy = B`, plus `evidence` URLs and `search_calls`).

### What you need
- `pip install -e ".[agent]"` (LangGraph), Ollama with the models, and **`TAVILY_API_KEY`** in `.env`.
- A task set: the published 36 tasks (`tasks/airports/dolap2026/`) or any `tasks.csv`.
- `configs/strategy_b_dolap2026.yaml`:
  - **search:** `tool`, `depth` (basic = 1 credit, advanced = 2), `results_per_query`, `max_subqueries`, `max_evidence_tokens`;
  - **`blocked_domains`:** the dataset's own sources, so the agent can't simply look up the table. The airports config blocks OurAirports, OpenFlights, their GitHub repos, Kaggle and the published repo. Check this list before the real run;
  - **`reuse_retrieval: true`:** plan and search once per task and model, then synthesize 5 times on that evidence. This keeps the credits at about 110 per model for the 36 tasks, instead of about 540;
  - **variants:** `search_evidence` (evidence only, isolates the effect of search) and `search_memory` (evidence plus the model's own knowledge). Both use the same plan, so the second variant reuses the first one's searches at no cost.
- Prompts: `prompts/agent/plan.v1.txt`, `synth.v1.txt`, `synth_memory.v1.txt`.

### Commands
```bash
strategy-b check  --config configs/strategy_b_dolap2026.yaml          # models ready, PFE, credit estimate
strategy-b render --config configs/strategy_b_dolap2026.yaml          # the plan prompt of one task
strategy-b run    --config configs/strategy_b_dolap2026.yaml --models qwen3-4b-instruct --limit 2
strategy-b run    --config configs/strategy_b_dolap2026.yaml
```

Every search is cached in `search_cache_dir`, shared by all models and variants, so a repeated query costs nothing and a run can be replayed offline.

The search buffer also:
- treats queries that differ only in case, spaces or trailing punctuation as the same query;
- searches an identical query once when several are in flight (the LLM cache does the same, so the two variants share one plan);
- stops searching at `search_max_credits` (a cap per `strategy-b run`).

Flags:
- `--runs N` overrides `runs_per_task` (both CLIs), for example `--runs 1` for a test.
- `--no-cache` skips cached LLM answers only.
- `--no-search-cache` searches again, which spends credits.
- `--retry-failed` reruns runs with status `api_error`, `timeout` or `retrieval_failed`.

For a cheap test, use `configs/strategy_b_dolap2026_smoke.yaml`: cloud gemma4, 1 run, basic depth, a 150-credit cap and 2 concurrent calls. The free Ollama Cloud tier returns HTTP 429 above 2 concurrent calls. Its results are in `notebooks/cloud_smoke_gemma4.ipynb`. `runs.jsonl` holds the full record of each run: queries, every result (with `blocked`), the evidence sent, both LLM calls, and the validation events.

### Not built (on purpose)
- **`waves: 2`** (one dependent search hop): the spec allows it as an option, but it is outside the one-try design.
- **Fetching full pages:** only the search snippets are used.
- **Scoring Strategy B in the notebook:** `evaluation.dolap` scores Strategy A's CSV answers; B's JSON predictions need a small loader before both can be compared.

## 5. Still to do before the real Strategy A run
1. Build `tasks.csv` (step 1).
2. Revise the draft templates (§3.2), saving changes as `.v2`.
3. Build the few-shot example file for each dataset (§3.3), then enable the few-shot templates in the config.
4. Decide where to run: this Mac is fine for trials, but for energy figures use a machine whose GPU CodeCarbon can measure.
5. Pull the models for the paper and fix their tags and quantizations (`ollama pull qwen3:14b`, ...).

## 6. Open decisions that affect a run
| Decision | Affects | Current default |
|---|---|---|
| Task design: published 2×2 (36 tasks) or Algorithm 2 of the new draft (about 1,215) | Cost × 30, `sampling` in the dataset config | The code follows Algorithm 2 of the October 2026 draft (`TASKDEFINITION_SPEC.md`) |
| With-context and without-context conditions from the published paper | Doubles the cost; needs a leakage-filtered context file per task from step 1 | Not built; G already carries the neighbourhood |
| PFE tokenizer: per model, or one shared tokenizer | `pfe_tokens` | Per model when `tokenizer` is set, otherwise `cl100k_base` |
| Few-shot examples: same for all tasks, or chosen per task | The examples file | The first N examples, for every task |
| Node identity: whether G always includes the node's name or code | Templates and step 1 | Not enforced |
| Repair call allowed | Cost and fairness across models | Allowed, one per run |
| Search for Strategy B: shared Tavily evidence or searches per model | Tavily credits | Not built |

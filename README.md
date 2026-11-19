# Data augmentation in property graphs using agentic AI

Code, task sets, runs and analyses for the extended DOLAP paper *Data Augmentation in Property Graphs using agentic
AI* (Information Systems). The question: can an LLM recover a hidden property of a node in a property graph from
some of the node's other properties, and does a web-search agent do better than the LLM alone?

- **Task.** A node `n`, a set **G** of given properties (|G| ∈ {1, 3, 5}, from the node and, with context, from its
  neighbours) and one hidden property **H**. The model must return the value of H, or abstain.
- **Strategy A (LLM alone).** Nine prompts (compact, zero-shot, 1/3/5-shot, each with and without chain of thought;
  Figure 2 of the paper), no tools.
- **Strategy B (search agent).** One call plans up to 3 web queries (Ollama web search), a second call answers from
  the retrieved evidence with citations (`search_evidence`). The `search_memory` variant replays only the second call
  on the same evidence, also allowing the model's own knowledge.
- **Measures.** Accuracy (F1, precision, F1 with abstentions ignored), energy (Wh), effort (token amplification) and
  task difficulty.

## The experiments in the paper

| Experiment | Tasks | Strategies | Results |
|---|---|---|---|
| Airports, context on | 108 tasks of `tasks/airports/v5_ctx` (the ids in `data/new-diff-airports_ressampled.csv`, resampled by difficulty out of 243) | A (9 prompts), B search, B search + memory | `results/airports/v5_ctx/` |
| Airports, context off | the matched tasks of `tasks/airports/v5_noctx` (77 of the 108 have one) | A | `results/airports/v5_noctx/` |
| Football, context on | `tasks/football/v1_ctx_idmix` (239 tasks) | A, B search, B search + memory | `results/football/v1_ctx_idmix/` |
| Football, context off | `tasks/football/v1_noctx_idmix` (233 tasks) | A | `results/football/v1_noctx_idmix/` |
| DOLAP 2026 comparison | the 36 published tasks, `tasks/airports/dolap2026` | A with the published prompts and with the new ones | `results/airports/dolap2026/` |

Models (Ollama Cloud), 3 runs each, answers voted over the runs: `gpt-oss:20b`, `gemma4:31b`, `gpt-oss:120b`,
`mistral-large-3:675b` and `deepseek-v4.1-flash`. Ollama has since retired mistral-large-3, so its runs can be
re-scored but not re-run. The DOLAP comparison uses the other four models, with 5 runs for the published protocol.

## Repository layout

| Path | Content |
|---|---|
| `src/taskdef/` | Task-set construction (Algorithm 2): graph snapshot, centrality (PageRank), context, property popularity, stratified sampling, difficulty, validation. Command `tasks`. |
| `src/llmcore/` | Provider-agnostic LLM calls (Ollama local and cloud, OpenAI, Anthropic, Google), response cache, tokenizers, energy (EcoLogits). |
| `src/strategy_a/` | Strategy A: prompt rendering, runs, answer parsing, `results.csv` / `runs.jsonl`. Command `strategy-a`. |
| `src/strategy_b/` | Strategy B: LangGraph agent (plan → search → synthesize), search tools and caches, replay. Command `strategy-b`. |
| `src/evaluation/` | Scoring: `dolap.py` (value tolerance, voting, and the DOLAP 2026 published scoring for comparison), `taskset.py` (correct / wrong / abstained, metrics, bootstrap), `eis.py` (energy). |
| `configs/` | One YAML per task set (`airports_*.yaml`, `football_*.yaml`) and per run (`strategy_a_*.yaml`, `strategy_b_*.yaml`). |
| `templates/` | Strategy A prompts: `figure2/` (the paper's 9 prompts), `dolap2026/` (the published DOLAP prompts), `figure2_dolap/` (Figure 2 prompts on the DOLAP input format). |
| `prompts/agent/` | Strategy B prompts: `plan.v2.txt`, `synth.v2.txt` (evidence only), `synth_memory.v2.txt` (evidence + own knowledge). |
| `data/` | Few-shot examples (`examples/`), property descriptions (`property_notes/`), the DOLAP 2026 published data (`dolap2026/`), the resampled airport task ids and the football difficulty file. |
| `tasks/` | Task sets: `tasks.csv` (what the strategies read), `ground_truth.csv` (only the evaluation reads it), `manifest.json`, `intermediate/` (snapshot, centrality, context, popularity). |
| `results/` | Raw runs per task set and strategy (`results.csv`, `runs.jsonl`, `runs.csv`, `run_manifest.json`), scored rows (`scored/`), energy and effort per run (`energy_effort/`), caches. |
| `scripts/` | Derived task sets, scored CSVs, per-run extracts, run orchestration (`run_*.sh`). |
| `notebooks/` | Figures, tables and report notebooks (`build_*.py` writes a notebook; `*_discussion.json` holds its text). |
| `docs/` | Specs, analyses and paper sections (see [Documentation](#documentation)). |
| `tests/` | Offline tests (no Neo4j, no API calls). |

## Setup

Python 3.11 or later.

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev,local,agent,reports]"   # add "api" for OpenAI / Anthropic / Google models
venv/bin/python -m pytest                              # offline tests: 107 pass
cp .env.example .env                                   # then fill in the keys you need
```

Extras: `local` (Ollama, tokenizers, CodeCarbon), `agent` (LangGraph and fastembed, for Strategy B), `reports`
(matplotlib and Jupyter, for `notebooks/`), `api` (other providers), `dev` (pytest, ruff).

**Keys.** `.env` holds the Neo4j connection and the Ollama keys. It is git-ignored, configs only name the
variable they read (`api_key_env`), and no key is written to any output. You only need Neo4j to rebuild a task set
from the database, and only need Ollama keys to make new model calls or searches. Re-scoring and every figure work
offline from the files in `results/`. The first `strategy-a` or `strategy-b` command downloads tokenizer files
(tiktoken and Hugging Face) once, so it needs network access.

## Running the pipeline

Every step reads files and writes files, so each one can be re-run alone. Runs are resumable: a run that is already
in `results.csv` is skipped, and the response and search caches mean that re-running a finished configuration makes
no new API calls.

### 1. Task sets (already built; only needed for a new task set)

```bash
venv/bin/tasks check-connection --config configs/airports_v5_ctx.yaml   # reads Neo4j, lists labels and relationships
venv/bin/tasks build --config configs/airports_v5_ctx.yaml              # writes tasks/airports/v5_ctx and v5_noctx
```

- The airport sets reuse the snapshot in `tasks/airports/v2_ctx/intermediate/snapshot` (`dataset.snapshot_dir`).
- Context off is a *matched* set (`no_context_output_dir`): same node, H, |G| and G stratum, with G made of the node's
  own properties. It is not the context-on tasks with the context removed.
- A task set is never overwritten: use a new `output_dir` for a new version.
- Football: `tasks snapshot --config configs/football_raw.yaml` → `scripts/football_lift_stats.py` (season
  statistics become player properties) → `tasks build --config configs/football_v1_ctx.yaml` →
  `scripts/football_identifier_mix.py` (adds the player's name and team to G at random) →
  `scripts/football_identifier_mix_noctx.py`.
- Few-shot examples: `scripts/build_fewshot_examples_v2.py configs/airports_v5_ctx.yaml data/examples/airports_v5.json`.

### 2. Strategy A

```bash
venv/bin/strategy-a check    --config configs/strategy_a_v5_ctx.yaml               # keys, models, templates
venv/bin/strategy-a render   --config configs/strategy_a_v5_ctx.yaml --variant zero  # print one prompt, no call
venv/bin/strategy-a estimate --config configs/strategy_a_v5_ctx.yaml               # calls, tokens, cost
venv/bin/strategy-a run      --config configs/strategy_a_v5_ctx.yaml [--models cloud-gemma4-31b] [--limit 5 --runs 1]
```

Configs: `strategy_a_v5_ctx.yaml`, `strategy_a_v5_noctx.yaml`, `strategy_a_football_idmix.yaml`,
`strategy_a_football_idmix_noctx.yaml`, and for the DOLAP comparison `strategy_a_dolap2026.yaml` (published prompts)
and `strategy_a_dolap2026_newprompts.yaml`. `scripts/run_v5.sh` and `scripts/run_dolap2026_lineup.sh` run whole
lineups, one model at a time.

### 3. Strategy B

```bash
venv/bin/strategy-b check  --config configs/strategy_b_v5_ctx.yaml
venv/bin/strategy-b run    --config configs/strategy_b_v5_ctx.yaml [--models ...] [--retry-failed]
venv/bin/strategy-b replay --config configs/strategy_b_v5_ctx_memory.yaml \
    --source-dir results/airports/v5_ctx/strategy_b --source-variant search_evidence    # search + memory, no new searches
```

Football: `strategy_b_football_idmix.yaml` and `strategy_b_football_idmix_memory.yaml`. Orchestration:
`scripts/run_v5.sh`, `scripts/run_v5_memory.sh`, `scripts/run_football_b.sh`.

**Costs and limits.** Model calls are cheap. One Ollama web search costs about $0.015, about 95% of the cost of a
Strategy B run. Ollama also enforces undocumented limits: about 300 searches per hour, and a block of 2–4 h after
about 700 searches. Keep the configs' pacing at `search.min_interval_s: 30` or more and
`max_rate_limit_wait_s` at 21600 or more. Ollama Pro allows 3 concurrent requests per account, so run one model at a
time per key, and never list two keys in `search.api_key_env`. Searches are cached in
`results/<dataset>/search_cache_ollama/`. The domain `github.com/felipeVsc` (this repository, which holds the ground
truth) is blocked.

### 4. Scoring

```bash
venv/bin/python notebooks/build_metrics_both_notebook.py        # airports v5_ctx: writes results/airports/v5_ctx/scored/
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/metrics_both.ipynb
venv/bin/python scripts/football_idmix_scored.py                 # football ctx and noctx: results/football/*/scored/
venv/bin/python scripts/football_idmix_scored.py airports/v5_noctx
venv/bin/python scripts/energy_effort_extract.py                 # results/energy_effort/runs.csv, accuracy.csv
venv/bin/python scripts/b_search_detail.py                       # Strategy B searches per run
venv/bin/python scripts/a_vs_b_per_task.py [football]            # A against B per task
venv/bin/python scripts/property_popularity.py                   # pop(p) per property
```

`scored_voted.csv` has one row per (strategy, prompt, model, task, hidden property) with the voted answer, the truth
and the outcome. Every figure and table is computed from these files.

### 5. Figures, tables and paper text

```bash
venv/bin/python notebooks/resampled_figures.py          # airports, 108 tasks: notebooks/figures/paper_reports/airports_resampled/
venv/bin/python notebooks/resampled_dolap_replica.py    # the DOLAP 2026 figures redrawn with the new data (same folder, dolap_*)
venv/bin/python notebooks/make_lessons_learned_tex.py   # docs/paper_lessons_learned_resampled.tex (§5.4 of the paper)
venv/bin/python notebooks/coauthor_figures.py           # airports and football A/B figures: figures/paper_reports/coauthor/
venv/bin/python notebooks/build_paper_reports_notebook.py football_a   # or football_b, football_ab, both
```

Each run writes PNG and PDF figures, CSV tables (`tables/`) and a `facts.json` with every number quoted in the
captions and text. Report notebooks are built and executed like this:

```bash
venv/bin/python notebooks/build_<name>_notebook.py [args]
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
venv/bin/jupyter nbconvert --to html notebooks/<name>.ipynb
```

| Notebook | What it covers |
|---|---|
| `dolap2026_ollama.ipynb` | Why the DOLAP 2026 numbers are higher than the new ones: the two scorings side by side |
| `dolap2026_lineup_report.ipynb` | The current models on the 36 DOLAP tasks, against the six proprietary models |
| `paper_report_football_*.ipynb`, `paper_report_both_datasets.ipynb` | Football Strategy A, B and A against B; both datasets |
| `energy_effort_report.ipynb` | Energy and token amplification, both datasets |
| `metrics_both.ipynb` | Airports A and B metrics; writes the airports scored files |

## Scoring and metrics

Each hidden property of a task gets one outcome, decided on the answer voted over the 3 runs:

- **abstained**: no value, null, or a marker such as "unknown";
- **correct**: within the tolerance of `evaluation.dolap.matches_standard`, which is token overlap of 0.75 or more in
  both directions for text, relative similarity of 0.75 or more for numbers, and 10 km for coordinates;
- **wrong**: otherwise.

Voting: text takes the most frequent value, numbers the mean, and the vote is an abstention only if every run
abstains. With c correct, w wrong and a abstained answers:

| Metric | Formula | Abstentions |
|---|---|---|
| Precision | c / (c + w) | not counted |
| Recall | c / (c + w + a) | counted as misses |
| **F1** (the paper's accuracy) | 2c / (2c + 2w + a) | counted as misses, through recall |
| F1 ign | 2c / (2c + w) | not counted |
| Abstention rate, error rate | a / n, w / n | |

The code is in `src/evaluation/taskset.py` (`outcome`, `score_voted`, `metrics`) and, for figures,
`notebooks/paper_report_lib.py` (`metrics`, `boot`). Intervals are 95% bootstrap over tasks.

`evaluation/dolap.py` also reproduces the DOLAP 2026 published scoring, for comparison only. That scoring drops
abstentions and fixes precision at 1. `docs/DOLAP_GAP_ANALYSIS.md` explains the difference.

- **Energy** (Wh per run): EcoLogits energy of the run's LLM calls plus 0.3 Wh per web search. Reported per run and
  per correct answer.
- **Token amplification**: tokens processed (input and output over all calls, reasoning included) divided by the
  tokens the user writes (template, input, context and targets).

## Documentation

| File | Content | Status |
|---|---|---|
| `docs/paper_findings_both_datasets.md` | Findings on both datasets, with the CSV index | results (243 airport tasks; superseded for airports by the 108-task figures) |
| `docs/DOLAP_GAP_ANALYSIS.md`, `docs/DOLAP_SCORING_EXPLAINED.md` | Why the DOLAP 2026 scores are higher; the old scoring explained | analysis |
| `docs/paper_lessons_learned_resampled.tex` | §5.4 Lessons learned, on the 108 airport tasks | paper text |
| `docs/RUNNING.md` | Early run instructions (23 Oct 2026) | historical; this README replaces it |

The design specs (`docs/*SPEC*.md`, `docs/METRICS.md`), the paper drafts (`docs/*.pdf`) and the project handover
notes are kept with the authors and are not in the repository (see `.gitignore`). The code's docstrings refer to the
spec sections by name.

## Data notes

- **Large files.** The raw runs (`runs.jsonl`, `runs.csv`, up to about 190 MB each), the caches and the
  `embeddings.sqlite` files are too large for a normal git push. Everything in the figures can be recomputed from
  `scored/`, `results/energy_effort/` and the small CSVs.
- In the airports scored files, the memory variant is `strategy=B, variant=search_memory`; in the football files it is
  `strategy=B_memory`. `paper_report_lib.load_voted()` normalises this.
- Football context comparisons must use the tasks whose context actually differs (`paper_report_lib.context_pairs()`):
  200 of the 233 context-off football tasks are identical to their context-on version.
- Strategy B on the context-off sets (`results/*/v*_noctx*/strategy_b*`) was started but not completed, and is not
  used.

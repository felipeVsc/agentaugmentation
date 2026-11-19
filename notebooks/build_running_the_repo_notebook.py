"""Builds notebooks/running_the_repo.ipynb: a walk-through of the repository, from the task sets to the paper figures.

    venv/bin/python notebooks/build_running_the_repo_notebook.py
    venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/running_the_repo.ipynb

Everything in the notebook runs offline, with no API key: the strategies are shown with the mock model and the mock
search, and the scoring and figures use the runs already in results/. The cells that would call Ollama Cloud are
guarded by RUN_LIVE = False."""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "running_the_repo.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Running the repository

A walk-through of the pipeline of *Data Augmentation in Property Graphs using agentic AI*:

1. **Task sets**: what a task is and where the task files are;
2. **Strategy A** (the LLM alone) and **3. Strategy B** (the search agent), run here offline with a mock model;
4. **Real runs** on Ollama Cloud (commands; off by default);
5. **Scoring**: from raw answers to F1, precision and recall;
6. **Figures and tables** of the paper.

Every cell runs without an API key and makes no paid call. The demo runs write to a temporary folder, never to
`results/`. Setup and the full command list are in `README.md`.
""")

code("""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)                                    # configs use paths relative to the repository root
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "notebooks")]
BIN = Path(sys.executable).parent                 # venv/bin: the `tasks`, `strategy-a` and `strategy-b` commands
DEMO = Path(tempfile.mkdtemp(prefix="agentaug_demo_"))
pd.set_option("display.max_colwidth", 120)

RUN_LIVE = False   # True runs the cells of section 4, which call Ollama Cloud (needs OLLAMA_API_KEY in .env)

def sh(*args, last=None):
    \"\"\"Run a command of the repository and print its output (only the last lines when `last` is set).\"\"\"
    out = subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=ROOT)
    lines = (out.stdout + out.stderr).strip().splitlines()
    print("\\n".join(lines[-last:] if last else lines))
    if out.returncode:
        print(f"(exit code {out.returncode})")

from dotenv import dotenv_values
env = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
print("repository:", ROOT.name)
print(".env found:", bool(env))
for name in ("OLLAMA_API_KEY", "NEO4J_URI"):          # whether they are set, never their values
    print(f"  {name} set:", bool(env.get(name) or os.environ.get(name)))
""")

md("""
## 1. Task sets

A task is one node of the graph, a set **G** of given properties (from the node, and with context also from its
neighbours) and one hidden property **H** that the model must recover. The task sets are in `tasks/`:

- `tasks.csv`: what the strategies read;
- `ground_truth.csv`: the hidden values, read only by the evaluation (the strategies refuse to open it);
- `manifest.json` and `intermediate/`: how the set was built (snapshot, centrality, context, popularity).

The airports set with context has 243 tasks; the paper evaluates the 108 of them listed in
`data/new-diff-airports_ressampled.csv`, resampled by difficulty.
""")

code("""
from evaluation import taskset as E

tasks = E.load_tasks(ROOT / "tasks/airports/v5_ctx/tasks.csv")
truth = E.load_ground_truth(ROOT / "tasks/airports/v5_ctx/ground_truth.csv")
selected = pd.read_csv(ROOT / "data/new-diff-airports_ressampled.csv").rename(columns={"taskID": "task_id"})
print(f"{len(tasks)} tasks in airports v5_ctx, {len(selected)} in the paper's evaluation, "
      f"{tasks.task_id.isin(selected.task_id).sum()} of them found")
tasks[["task_id", "node_id", "size_G", "node_stratum", "H_stratum", "G_stratum"]].head()
""")

code("""
t = tasks.iloc[0]
print("node:", t.node_id)
print("G (given):")
for g in t.G_list:
    print(f"  {g['path']}.{g['property_name']} = {g['value']!r}")
print("H (hidden):", [h["prop_key"] for h in t.H_list])
print("ground truth:", {h["prop_key"]: truth[(t.task_id, h["prop_key"])] for h in t.H_list})
""")

md("""
A new task set is built from Neo4j (read only) with `tasks build --config configs/airports_v5_ctx.yaml`; it writes
both the context-on set and the matched context-off set (`tasks/airports/v5_noctx`). Task sets are never overwritten.

## 2. Strategy A: the LLM alone

A Strategy A config names the task set, the prompt templates (`templates/figure2/`: compact, zero-shot, few-shot,
with and without chain of thought), the models and the output folder. Here a demo config uses the **mock** model,
which answers every prompt with an empty CSV (an abstention), so the whole run is offline.
""")

code("""
demo_a = {
    "inputs": {"tasks": "tasks/airports/v5_ctx/tasks.csv", "examples": "data/examples/airports_v5.json",
               "property_notes": "data/property_notes/airports.yaml"},
    "runs_per_task": 3,
    "pfe_tokenizer": "simple",                       # the real configs use cl100k_base (downloaded once)
    "output_dir": str(DEMO / "strategy_a"),
    "templates": [{"id": "zero", "file": "templates/figure2/zero.v2.txt", "output": "csv"},
                  {"id": "few", "file": "templates/figure2/few.v2.txt", "output": "csv", "n_examples": [1]}],
    "model_defaults": {"params": {"temperature": 0.7, "max_tokens": 256}, "max_concurrency": 1},
    "models": [{"id": "mock-model", "provider": "mock", "tokenizer": "simple",
                "options": {"text": "property,value\\n"}}],
}
CFG_A = DEMO / "strategy_a.yaml"
CFG_A.write_text(yaml.safe_dump(demo_a, sort_keys=False))
sh(BIN / "strategy-a", "render", "--config", CFG_A, "--limit", "1", "--variant", "zero")   # the prompt, no call
""")

code("""
sh(BIN / "strategy-a", "run", "--config", CFG_A, "--limit", "3")
sh(BIN / "strategy-a", "run", "--config", CFG_A, "--limit", "3")   # resumable: the finished runs are skipped
res_a = pd.read_csv(DEMO / "strategy_a/results.csv")
res_a[["task_id", "variant", "model", "run_index", "status", "predictions", "tokens_in", "tokens_out"]]
""")

md("""
Each run is one row of `results.csv` (the parsed answer per hidden property) and one record of `runs.jsonl` (the
full prompts, raw responses, tokens, energy and timings). The response cache (`cache/`) means a finished
configuration is never paid for twice.

## 3. Strategy B: the search agent

Two LLM calls per run: the **plan** identifies the entity and writes up to 3 web queries; the **synthesis** answers
from the retrieved evidence, citing it. Here the model and the search are both mocks, driven from Python with the
same runner the `strategy-b` command uses.
""")

code("""
from llmcore.providers.mock import MockProvider
from strategy_b.config import load_config as load_b
from strategy_b.runner import Runner as RunnerB
from strategy_b.search import MockSearch

demo_b = {
    "inputs": {"tasks": "tasks/airports/v5_ctx/tasks.csv", "property_notes": "data/property_notes/airports.yaml"},
    "runs_per_task": 1, "max_subqueries": 3, "results_per_query": 3, "pfe_tokenizer": "simple",
    "output_dir": str(DEMO / "strategy_b"),
    "search": {"tool": "mock"},
    "blocked_domains": ["github.com/felipeVsc"],      # the repository that holds the ground truth
    "variants": [{"id": "search_evidence", "plan": "prompts/agent/plan.v2.txt",
                  "synthesize": "prompts/agent/synth.v2.txt"}],
    "models": [{"id": "mock-model", "provider": "mock", "tokenizer": "simple",
                "params": {"temperature": 0.7, "max_tokens": 256}, "max_concurrency": 1}],
}
CFG_B = DEMO / "strategy_b.yaml"
CFG_B.write_text(yaml.safe_dump(demo_b, sort_keys=False))

def mock_llm(request):
    text = request.messages[-1]["content"]
    if "Write at most" in text:                       # the plan call
        return json.dumps({"entity": "unknown", "queries": [{"id": "q1", "query": "airport ICAO code", "targets": []}]})
    # the synthesis call: answer every asked property with null (an abstention), in the requested format
    labels = re.findall(r'"([^"]+)": \\{"value": "<value or null>"', text)
    return json.dumps({"predictions": {label: {"value": None, "unit": None, "sources": []} for label in labels}})

def mock_search(query):
    return [{"url": "https://example.org/airport", "title": "An airport", "content": "Some text about an airport.",
             "score": 0.9}]

cfg_b = load_b(CFG_B)
runner_b = RunnerB(cfg_b, limit=2, providers={"mock-model": MockProvider(cfg_b.models[0], responder=mock_llm)},
                   search_tool=MockSearch({}, responder=mock_search), log=print)
runner_b.run()
pd.read_csv(DEMO / "strategy_b/results.csv")[["task_id", "variant", "status", "predictions", "llm_calls", "search_calls"]]
""")

code("""
run = json.loads((DEMO / "strategy_b/runs.jsonl").read_text().splitlines()[0])
print("fields logged per run:", sorted(run))
""")

md("""
## 4. Real runs on Ollama Cloud

The paper's configs are in `configs/` (`strategy_a_v5_ctx.yaml`, `strategy_b_v5_ctx.yaml`, the football and DOLAP
ones). They read the key from `OLLAMA_API_KEY` in `.env`. Useful steps, from the repository root:

```bash
venv/bin/strategy-a check    --config configs/strategy_a_v5_ctx.yaml                 # keys, models, templates
venv/bin/strategy-a estimate --config configs/strategy_a_v5_ctx.yaml --pending       # calls, tokens and cost left
venv/bin/strategy-a run      --config configs/strategy_a_v5_ctx.yaml --models cloud-gemma4-31b
venv/bin/strategy-b run      --config configs/strategy_b_v5_ctx.yaml --models cloud-gemma4-31b
venv/bin/strategy-b replay   --config configs/strategy_b_v5_ctx_memory.yaml \\
    --source-dir results/airports/v5_ctx/strategy_b --source-variant search_evidence   # B + memory, no new search
```

The paper's runs are complete, so `run` on these configs finds every run already done and makes no call. A new
experiment needs its own `output_dir`. Strategy B costs about $0.015 per web search; keep the configs' pacing
(30 s between searches) and run one model at a time per key.
""")

code("""
if RUN_LIVE:
    sh(BIN / "strategy-a", "check", "--config", "configs/strategy_a_v5_ctx.yaml", "--models", "cloud-gemma4-31b")
else:
    print("RUN_LIVE is False: no call to Ollama Cloud")
""")

md("""
## 5. Scoring

Each hidden property gets one outcome, on the answer voted over the runs (text: most frequent value; numbers: mean;
abstained only if every run abstains):

- **abstained**: no value, null or "unknown";
- **correct**: within the tolerance of `evaluation.dolap.matches_standard` (text overlap ≥ 0.75 both ways, numbers
  within 25%, coordinates within 10 km);
- **wrong**: otherwise.

With c correct, w wrong and a abstained: **precision = c / (c + w)**, **recall = c / (c + w + a)**,
**F1 = 2c / (2c + 2w + a)** (the paper's accuracy) and **F1 ign = 2c / (2c + w)** (abstentions ignored).

Scoring the raw answers of one configuration reproduces the shared scored file exactly:
""")

code("""
raw = E.load_results(ROOT / "results/airports/v5_ctx/strategy_a")
raw = raw[(raw.model == "cloud-gemma4-31b") & (raw.variant == "zero") & (raw.status == "ok")]
scored = E.score_voted(raw, tasks, truth)

shared = pd.read_csv(ROOT / "results/airports/v5_ctx/scored/scored_voted.csv", low_memory=False)
same = scored.merge(shared[(shared.strategy == "A") & (shared.model == "cloud-gemma4-31b") & (shared.variant == "zero")],
                    on=["task_id", "prop_key"], suffixes=("", "_shared"))
print(f"{len(scored)} properties scored; identical outcome in the shared file: {(same.outcome == same.outcome_shared).mean():.0%}")
{k: round(v, 3) if isinstance(v, float) else v for k, v in E.metrics(scored).items()}
""")

md("""
`results/<dataset>/<set>/scored/scored_voted.csv` holds these rows for every strategy, prompt, model and task
(written by `notebooks/build_metrics_both_notebook.py` for airports and `scripts/football_idmix_scored.py` for
football). The paper's numbers on the 108 tasks, pooled over the models:
""")

code("""
v = shared[shared.task_id.isin(selected.task_id)].copy()
v["approach"] = v.variant.map({"search_evidence": "B search", "search_memory": "B search + memory"}).fillna("A, all 9 prompts")
pooled = E.metrics_by(v, ["approach"]).set_index("approach")
pooled["f1_ign"] = pooled.published_f1
pooled[["properties", "correct", "wrong", "abstained", "f1", "precision", "recall", "f1_ign"]].round(3)
""")

code("""
per_model = E.metrics_by(v, ["model", "approach"]).pivot(index="model", columns="approach", values="f1")
per_model.round(3)
""")

md("""
## 6. Figures and tables of the paper

Each script reads the scored files and writes PNG and PDF figures, CSV tables and a `facts.json` with every number
quoted in the text:

| Script | Output |
|---|---|
| `notebooks/resampled_figures.py` | airports, 108 tasks: accuracy, energy, tokens, difficulty (`figures/paper_reports/airports_resampled/`) |
| `notebooks/resampled_dolap_replica.py` | the DOLAP 2026 figures redrawn with the new data (same folder, `dolap_*`) |
| `notebooks/make_lessons_learned_tex.py` | `docs/paper_lessons_learned_resampled.tex`, the Lessons learned section |
| `notebooks/coauthor_figures.py` | airports and football A/B figures (`figures/paper_reports/coauthor/`) |
| `notebooks/build_paper_reports_notebook.py` | the football and both-datasets report notebooks |
""")

code("""
sh(sys.executable, "notebooks/resampled_figures.py", last=1)
out = ROOT / "notebooks/figures/paper_reports/airports_resampled"
print(sorted(p.name for p in out.glob("*.png")))
""")

code("""
from IPython.display import Image
Image(filename=str(out / "f1_precision_abstention.png"), width=900)
""")

code("""
facts = json.loads((out / "facts.json").read_text())
print(json.dumps(facts["best_prompt"], indent=1))
""")

md("""
## Where things are

| What | Where |
|---|---|
| Task sets | `tasks/<dataset>/<set>/` |
| Raw runs | `results/<dataset>/<set>/strategy_{a,b,b_memory}/results.csv` (the full logs `runs.jsonl` are not in git) |
| Scored rows | `results/<dataset>/<set>/scored/scored_voted.csv` |
| Energy and effort per run | `results/energy_effort/runs.csv` |
| Figures and tables | `notebooks/figures/paper_reports/` |
| Code | `src/taskdef`, `src/strategy_a`, `src/strategy_b`, `src/evaluation`, `src/llmcore` |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)

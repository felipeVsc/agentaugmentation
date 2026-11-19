"""What the Strategy B agent searched for and found, per run, on airports-v5-ctx and football-v1-ctx-idmix.

    venv/bin/python scripts/b_search_detail.py

Writes, for each set, results/<dataset>/<set>/scored/:
- b_search_runs.csv: one row per (variant, model, task, run) of strategy_b (search_evidence) and strategy_b_memory
  (search_memory, the same retrieval replayed): the entity the planner named and whether it matches the task's node,
  queries, results, blocked results, evidence passed and its tokens, real searches, whether the run answered and
  whether the answer cites evidence (an answer without a citation comes from the model's memory);
- b_evidence_domains.csv: one row per (variant, model, domain): how often the domain was in the evidence passed to the
  synthesis and how often an answer cited it.
The runs.jsonl files are ~160 MB each; the reports read these instead."""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["cloud-gpt-oss-20b", "cloud-gemma4-31b", "cloud-gpt-oss-120b", "cloud-mistral-large-3", "cloud-deepseek-v4.1-flash"]
SETS = {  # set -> the task nodes' true names (key column, name column, file)
    "airports/v5_ctx": ("icao", "name", "tasks/airports/v5_ctx/task_airports_centrality.csv"),
    "football/v1_ctx_idmix": ("node_id", "name", "tasks/football/v1_ctx_idmix/task_players_centrality.csv"),
}
STOP = {"airport", "international", "regional", "municipal", "airfield", "aerodrome", "field", "the", "de", "of", "and"}


def words(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return {w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP and len(w) > 1}


def matches(planned, true_name):
    """The planner's entity names the task's node: at least half of the node's name words appear in it."""
    t, p = words(true_name), words(planned)
    return bool(t) and len(t & p) / len(t) >= 0.5


def domain(url):
    d = urlparse(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d


for name, (key, col, names_file) in SETS.items():
    tasks = pd.read_csv(ROOT / "tasks" / name / "tasks.csv", dtype=str).set_index("task_id")
    true = pd.read_csv(ROOT / names_file, dtype=str).set_index(key)[col].to_dict()
    runs, dom = [], Counter()
    for sub in ("strategy_b", "strategy_b_memory"):
        with open(ROOT / "results" / name / sub / "runs.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("model") not in MODELS:
                    continue
                u = r.get("usage") or {}
                node = tasks.loc[r["task_id"], "node_id"]
                entity = r.get("plan_entity") or "unknown"
                preds, cites = r.get("predictions") or {}, r.get("evidence_urls") or {}
                answered = [p for p, v in preds.items() if v is not None]
                cited = [p for p in answered if cites.get(p)]
                passed = [domain(e["url"]) for e in r.get("evidence_passed") or [] if e.get("url")]
                cited_d = {domain(x) for p in answered for x in cites.get(p) or []}
                for d in set(passed):
                    dom[(r["variant"], r["model"], d, "passed")] += 1
                for d in cited_d:
                    dom[(r["variant"], r["model"], d, "cited")] += 1
                runs.append({
                    "variant": r["variant"], "model": r["model"], "task_id": r["task_id"], "run_index": r["run_index"],
                    "status": r["status"], "node_id": node, "node_name": true.get(node),
                    "plan_entity": entity, "entity_named": entity.strip().lower() not in ("", "unknown"),
                    "entity_match": matches(entity, true.get(node, "")),
                    "queries": len(r.get("queries") or []),
                    "query_errors": sum(bool(q.get("error")) for q in r.get("queries") or []),
                    "results": len(r.get("results") or []),
                    "results_blocked": sum(bool(x.get("blocked")) for x in r.get("results") or []),
                    "evidence_passed": len(r.get("evidence_passed") or []),
                    "evidence_domains": len(set(passed)), "evidence_tokens": u.get("evidence_tokens", 0),
                    "evidence_truncated": any(str(e).startswith("evidence_truncated") for e in r.get("validation_events") or []),
                    "searches": u.get("search_calls", 0), "searches_real": (u.get("search_calls", 0) or 0) - (u.get("search_cache_hits", 0) or 0),
                    "answered": bool(answered), "answer_cited": bool(cited), "answer_uncited": bool(answered) and not cited,
                    "cited_domains": ";".join(sorted(cited_d)),
                })
    out = ROOT / "results" / name / "scored"
    out.mkdir(exist_ok=True)
    runs = pd.DataFrame(runs)
    # the memory replay reuses the retrieval of the search_evidence run, which alone logs the truncation
    k = ["model", "task_id", "run_index"]
    trunc = runs[runs.variant == "search_evidence"].set_index(k).evidence_truncated
    mem = runs.variant == "search_memory"
    runs.loc[mem, "evidence_truncated"] = runs[mem].set_index(k).index.map(trunc).values
    runs.to_csv(out / "b_search_runs.csv", index=False)
    d = pd.Series(dom).rename("runs").reset_index()
    d.columns = ["variant", "model", "domain", "role", "runs"]
    d = d.pivot_table(index=["variant", "model", "domain"], columns="role", values="runs", fill_value=0).reset_index()
    d.columns.name = None
    d.sort_values(["variant", "model", "passed"], ascending=[True, True, False]).to_csv(out / "b_evidence_domains.csv", index=False)
    print(f"{name}: b_search_runs.csv {len(runs):,} rows; entity named {runs.entity_named.mean():.0%}, "
          f"matches the node {runs.entity_match.mean():.0%}; b_evidence_domains.csv {len(d):,} rows")

"""The no-context version of tasks/football/v1_ctx_idmix (scripts/football_identifier_mix.py): same tasks, but G holds
only properties of the player itself, as tasks/football/v1_noctx for v1_ctx.

    venv/bin/python scripts/football_identifier_mix_noctx.py

- The team, when given through PLAYED_FOR (Player-[PLAYED_FOR]->Team.Team name), becomes the player's own Squad
  (same value), so the coin flips of the identifier design are kept.
- Every other context property (league name and country, team country) is replaced as in the generator
  (TaskBuilder.without_context): by an own property of the player from the task's G stratum, not in H or G and not
  leaking H, drawn in a stable random order; it takes the position of the replaced entry. The name and the Squad are
  never drawn as replacements, so has_name / has_team stay the coin flips' result.
- Tasks without context properties are identical and keep their task_id (as the generator's ids, a hash of the
  task's content), so their runs can be reused from the cache of v1_ctx_idmix. Changed tasks get a new task_id;
  paired_task_id = the v1_ctx_idmix task, source_task_id = the v1_ctx task.

Writes tasks/football/v1_noctx_idmix/: tasks.csv, ground_truth.csv, tasks_readable.csv, manifest.json."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskdef.difficulty import compute_difficulty, surprise  # noqa: E402
from taskdef.leakage import value_leaks  # noqa: E402
from taskdef.sampling import Quartiles, stable_order  # noqa: E402

SRC, OUT, BASE = ROOT / "tasks/football/v1_ctx_idmix", ROOT / "tasks/football/v1_noctx_idmix", ROOT / "tasks/football/v1_ctx"
NAME_KEY, SQUAD_KEY, TEAM_CTX_KEY = "Player.Name", "Player.Squad", "Player-[PLAYED_FOR]->Team.Team name"
VERSION = "football-v1-noctx-idmix"

src_manifest = json.loads((SRC / "manifest.json").read_text())
base = src_manifest["source_manifest"]
seed, min_len = base["config"]["sampling"]["seed"], base["config"]["properties"]["leak_min_substring_len"]
g_q = Quartiles(tuple(base["quartiles"]["G_property_popularity"]))
tasks = pd.read_csv(SRC / "tasks.csv", dtype={"paired_task_id": str})
truth = pd.read_csv(SRC / "ground_truth.csv")
hidden_value = {(r.task_id, r.prop_key): json.loads(r.value) for r in truth.itertuples()}
ctx = pd.read_parquet(BASE / "intermediate/context.parquet")
own = ctx[ctx.node_id.isin(tasks.node_id) & ctx.origin.eq("node") & ctx.eligible]


def entry(r, identifier=False) -> dict:
    """A G entry from a context.parquet row, in the format of tasks.csv."""
    e = {"prop_key": r.prop_key, "property_name": r.property_name, "element_type": r.element_type,
         "element_id": r.element_id, "path": r.path, "origin": r.origin, "value": json.loads(r.value),
         "popularity": r.popularity, "centrality": r.centrality, "score": r.popularity * r.centrality, "hops": int(r.hops)}
    return {**e, "identifier": True} if identifier else e


own_rows = {nid: {r.prop_key: r for r in d.itertuples()} for nid, d in own.groupby("node_id")}
rows, gt_rows, skipped, changes = [], [], [], []
for t in tasks.to_dict("records"):
    H, G = json.loads(t["H"]), json.loads(t["G"])
    if all(e["origin"] == "node" for e in G):
        rows.append({**t, "task_set_version": VERSION, "paired_task_id": t["task_id"], "source_task_id": t["paired_task_id"],
                     "context_replaced": json.dumps([])})
        gt_rows += truth[truth.task_id == t["task_id"]].to_dict("records")
        continue
    cand = own_rows[t["node_id"]]
    h_keys = {h["prop_key"] for h in H}
    leaks = lambda r: any(value_leaks(hidden_value[(t["task_id"], h["prop_key"])], json.loads(r.value), min_len) for h in H)
    pool = [r for k, r in cand.items() if k not in h_keys | {e["prop_key"] for e in G} | {NAME_KEY, SQUAD_KEY}
            and g_q.stratum(r.popularity) == t["G_stratum"] and not leaks(r)]
    pool = iter(stable_order(pool, seed, f"Gnoctx|{t['task_id']}", key=lambda r: r.prop_key))
    newG, replaced, reason = [], [], None
    for e in G:
        if e["origin"] == "node":
            newG.append(e)
            continue
        if e["prop_key"] == TEAM_CTX_KEY and SQUAD_KEY in cand and SQUAD_KEY not in h_keys and not leaks(cand[SQUAD_KEY]):
            r = cand[SQUAD_KEY]
            new = entry(r, identifier=e.get("identifier", False))
        else:
            r = next(pool, None)
            if r is None:
                reason = "G_insufficient_own"
                break
            new = entry(r)
        newG.append(new)
        replaced.append({"from": e["prop_key"], "to": new["prop_key"]})
    if reason:
        skipped.append({"task_id": t["task_id"], "node_id": t["node_id"], "G_stratum": t["G_stratum"], "reason": reason})
        continue
    G = newG
    has_name, has_team = any(e["prop_key"] == NAME_KEY for e in G), any(e["prop_key"] in (SQUAD_KEY, TEAM_CTX_KEY) for e in G)
    d = compute_difficulty(t["node_centrality"], [h["score"] for h in H], [e["score"] for e in G])
    new_id = hashlib.sha256(f"{t['task_id']}|noctx".encode()).hexdigest()[:16]
    rows.append({**t, "task_id": new_id, "task_set_version": VERSION, "G": json.dumps(G, ensure_ascii=False),
                 "difficulty": d.value, "diff_ratio": d.ratio, "diff_node_surprise": d.node_surprise, "diff_prop_surprise": d.prop_surprise,
                 "diff_prop_surprise_G": sum(surprise(e["score"]) for e in G),
                 "n_G_own": len(G), "n_G_context": 0, "G_has_own_property": True,
                 "condition": {(False, False): "none", (True, False): "name", (False, True): "team", (True, True): "both"}[(has_name, has_team)],
                 "has_name": has_name, "has_team": has_team,
                 "paired_task_id": t["task_id"], "source_task_id": t["paired_task_id"],
                 "context_replaced": json.dumps(replaced, ensure_ascii=False)})
    gt_rows += truth[truth.task_id == t["task_id"]].assign(task_id=new_id).to_dict("records")
    changes += replaced

OUT.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(OUT / "tasks.csv", index=False)
pd.DataFrame(gt_rows).to_csv(OUT / "ground_truth.csv", index=False)

# readable version, as tasks/football/v1_ctx_idmix/tasks_readable.csv
readable_src = pd.read_csv(SRC / "tasks_readable.csv").set_index("task_id")
fmt = lambda e: f"{e['prop_key'] if e['origin'] != 'node' else e['property_name']} = {e['value']}"
readable = pd.DataFrame([{
    "task_id": r["task_id"], "paired_task_id": r["paired_task_id"], "source_task_id": r["source_task_id"], "node_id": r["node_id"],
    "player": readable_src.loc[r["paired_task_id"], "player"], "node_stratum": r["node_stratum"], "H_stratum": r["H_stratum"],
    "G_stratum": r["G_stratum"], "size_G": r["size_G"], "condition": r["condition"], "context_replaced": r["context_replaced"],
    "hidden_property": json.loads(r["H"])[0]["property_name"], "true_value": readable_src.loc[r["paired_task_id"], "true_value"],
    "G": "; ".join(fmt(e) for e in json.loads(r["G"])), "difficulty": round(r["difficulty"], 4)} for r in rows])
readable.to_csv(OUT / "tasks_readable.csv", index=False)

n_changed = int((out.context_replaced != "[]").sum())
counts = pd.crosstab(out.size_G, out.condition)
(OUT / "manifest.json").write_text(json.dumps({
    "task_set_version": VERSION, "derived_from": src_manifest["task_set_version"], "source_dir": str(SRC.relative_to(ROOT)),
    "rule": "each context property of G replaced: the PLAYED_FOR team name by the player's Squad (same value), the others by an own property of the G stratum in a stable random order (as TaskBuilder.without_context), never the name or the Squad",
    "tasks": len(rows), "identical": len(rows) - n_changed, "changed": n_changed,
    "replacements": pd.DataFrame(changes).value_counts().reset_index().to_dict("records") if changes else [],
    "condition_by_size_G": {str(k): v for k, v in counts.to_dict("index").items()},
    "skipped": skipped, "source_manifest": src_manifest}, indent=1, ensure_ascii=False, default=int))
print(f"{len(rows)} tasks: {len(rows) - n_changed} identical, {n_changed} with context replaced; skipped {len(skipped)}: {skipped}")
print(pd.DataFrame(changes).value_counts().to_string()); print(counts)

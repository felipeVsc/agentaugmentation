"""Derive a football Player task set where identifiers are frequent but random: for each task of tasks/football/v1_ctx
two independent coin flips decide whether the player's name and the team are added to G, for estimating how much
each identifier helps.

    venv/bin/python scripts/football_identifier_mix.py

- In v1_ctx the name is 1 of about 45 properties of the high popularity stratum (most are season statistics), so it is
  drawn in 3 of 239 tasks; the team in 15. Here each is added with probability P_ADD, independently of the other
  and of the task (seeded per task). With |G| = 1 and both drawn, only one fits: the name or the team, at random.
- Strata are kept: an identifier is only added to a task whose G stratum is the identifier's own (its popularity
  stratum, as the generator draws G; name, Team name and Squad are all high). Tasks of the other G strata are not
  flipped (condition_assigned = "not eligible") and keep the G of v1_ctx.
- name = Player.Name; team = Player-[PLAYED_FOR]->Team.Team name, or Player.Squad (same value) for players without
  a PLAYED_FOR team. Each identifier replaces a randomly chosen sampled entry and goes to a random position; |G|
  stays 1, 3 or 5. G_stratum describes the sampled entries; G_dropped records the ones taken out.
- An identifier is not added when H is that identifier or its value leaks H (as in the generator); the task then
  gets the condition without it (condition_assigned keeps the draw: none / name / team / both). Identifiers already sampled in G are kept, so
  has_name / has_team describe the final G.
- New task_id (hash of the original id and the condition); paired_task_id = the original task. Difficulty is
  recomputed with the final G.

Writes tasks/football/v1_ctx_idmix/: tasks.csv, ground_truth.csv, tasks_readable.csv, manifest.json."""
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskdef.difficulty import compute_difficulty, surprise  # noqa: E402
from taskdef.leakage import value_leaks  # noqa: E402
from taskdef.sampling import Quartiles  # noqa: E402

SRC, OUT = ROOT / "tasks/football/v1_ctx", ROOT / "tasks/football/v1_ctx_idmix"
NAME_KEY = "Player.Name"
TEAM_KEYS = ["Player-[PLAYED_FOR]->Team.Team name", "Player.Squad"]  # first available
VERSION, SEED, P_ADD = "football-v1-ctx-idmix", 42, 0.38  # 0.38 within the high G stratum (was 0.2 in every stratum)

manifest = json.loads((SRC / "manifest.json").read_text())
min_len = manifest["config"]["properties"]["leak_min_substring_len"]
assert manifest["config"]["sampling"]["G_strata_by"] == "popularity"
g_q = Quartiles(tuple(manifest["quartiles"]["G_property_popularity"]))
tasks = pd.read_csv(SRC / "tasks.csv", dtype={"paired_task_id": str})
truth = pd.read_csv(SRC / "ground_truth.csv")
hidden_value = {(r.task_id, r.prop_key): json.loads(r.value) for r in truth.itertuples()}
ctx = pd.read_parquet(SRC / "intermediate/context.parquet")
ctx = ctx[ctx.node_id.isin(tasks.node_id) & ctx.prop_key.isin([NAME_KEY, *TEAM_KEYS])]


def entry(r) -> dict:
    """A G entry from a context.parquet row, in the format of tasks.csv."""
    return {"prop_key": r.prop_key, "property_name": r.property_name, "element_type": r.element_type,
            "element_id": r.element_id, "path": r.path, "origin": r.origin, "value": json.loads(r.value),
            "popularity": r.popularity, "centrality": r.centrality, "score": r.popularity * r.centrality,
            "hops": int(r.hops), "identifier": True}


ident = {}  # node_id -> {"name": entry, "team": entry}
for nid, d in ctx.groupby("node_id"):
    rows = {r.prop_key: r for r in d.itertuples()}
    ident[nid] = {"name": entry(rows[NAME_KEY]) if NAME_KEY in rows else None,
                  "team": next((entry(rows[k]) for k in TEAM_KEYS if k in rows), None)}
is_name = lambda e: e["prop_key"] == NAME_KEY
is_team = lambda e: e["prop_key"] in TEAM_KEYS
ident_strata = {g_q.stratum(e["popularity"]) for v in ident.values() for e in v.values() if e}
assert len(ident_strata) == 1, ident_strata  # name and team share one stratum, so eligibility is one test per task
IDENT_STRATUM = ident_strata.pop()

# two independent coin flips per task of the identifiers' G stratum, seeded by the task id
assigned = {}
for tid, stratum in zip(tasks.task_id, tasks.G_stratum):
    if stratum != IDENT_STRATUM:
        assigned[tid] = "not eligible"
        continue
    crng = random.Random(f"{SEED}|flip|{tid}")
    name, team = crng.random() < P_ADD, crng.random() < P_ADD
    assigned[tid] = {(False, False): "none", (True, False): "name", (False, True): "team", (True, True): "both"}[(name, team)]

rows, gt_rows, notes = [], [], []
for t in tasks.to_dict("records"):
    H, G = json.loads(t["H"]), json.loads(t["G"])
    cond = assigned[t["task_id"]]
    want = {"not eligible": [], "none": [], "name": ["name"], "team": ["team"], "both": ["name", "team"]}[cond]
    add = []
    for kind in want:
        e = ident.get(t["node_id"], {}).get(kind)
        if e is None:
            notes.append({"task_id": t["task_id"], "condition": cond, "reason": f"no {kind} property"})
        elif any(h["prop_key"] == e["prop_key"] or (kind == "team" and h["prop_key"] in TEAM_KEYS) for h in H):
            notes.append({"task_id": t["task_id"], "condition": cond, "reason": f"H is the {kind}"})
        elif leaks := [h["prop_key"] for h in H if value_leaks(hidden_value[(t["task_id"], h["prop_key"])], e["value"], min_len)]:
            notes.append({"task_id": t["task_id"], "condition": cond, "reason": f"{kind} leaks {leaks}"})
        elif not any((is_name if kind == "name" else is_team)(g) for g in G):
            add.append(e)
    if len(add) > len(G):  # |G| = 1 and both drawn: keep one at random
        add = [random.Random(f"{SEED}|one|{t['task_id']}").choice(add)]
    # each added identifier replaces a random sampled (non-identifier) entry and takes a random position
    trng = random.Random(f"{SEED}|{t['task_id']}")
    replaceable = [i for i, g in enumerate(G) if not (is_name(g) or is_team(g))]
    drop = sorted(trng.sample(replaceable, min(len(add), len(replaceable))))
    dropped = [G[i]["prop_key"] for i in drop]
    G = [g for i, g in enumerate(G) if i not in drop]
    for e in add[:len(drop)]:
        G.insert(trng.randint(0, len(G)), e)
    has_name, has_team = any(map(is_name, G)), any(map(is_team, G))
    final = {(False, False): "none", (True, False): "name", (False, True): "team", (True, True): "both"}[(has_name, has_team)]
    d = compute_difficulty(t["node_centrality"], [h["score"] for h in H], [e["score"] for e in G])
    new_id = hashlib.sha256(f"{t['task_id']}|idmix|{cond}".encode()).hexdigest()[:16]
    rows.append({**t, "task_id": new_id, "task_set_version": VERSION, "size_G": len(G), "G": json.dumps(G, ensure_ascii=False),
                 "difficulty": d.value, "diff_ratio": d.ratio, "diff_node_surprise": d.node_surprise, "diff_prop_surprise": d.prop_surprise,
                 "diff_prop_surprise_G": sum(surprise(e["score"]) for e in G),
                 "n_G_own": sum(e["origin"] == "node" for e in G), "n_G_context": sum(e["origin"] != "node" for e in G),
                 "G_has_own_property": any(e["origin"] == "node" for e in G), "paired_task_id": t["task_id"],
                 "condition_assigned": cond, "condition": final, "has_name": has_name, "has_team": has_team,
                 "n_identifiers_added": len(drop), "G_dropped": json.dumps(dropped, ensure_ascii=False)})
    gt_rows += truth[truth.task_id == t["task_id"]].assign(task_id=new_id).to_dict("records")

OUT.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(OUT / "tasks.csv", index=False)
pd.DataFrame(gt_rows).to_csv(OUT / "ground_truth.csv", index=False)

# readable version, as tasks/football/v1_ctx/tasks_readable.csv
names = {nid: v["name"]["value"] if v["name"] else None for nid, v in ident.items()}
fmt = lambda e: f"{e['prop_key'] if e['origin'] != 'node' else e['property_name']} = {e['value']}"
readable = pd.DataFrame([{
    "task_id": r["task_id"], "paired_task_id": r["paired_task_id"], "node_id": r["node_id"], "player": names.get(r["node_id"]),
    "node_stratum": r["node_stratum"], "H_stratum": r["H_stratum"], "G_stratum": r["G_stratum"], "size_G": r["size_G"],
    "condition": r["condition"], "condition_assigned": r["condition_assigned"], "G_dropped": r["G_dropped"],
    "hidden_property": json.loads(r["H"])[0]["property_name"],
    "true_value": json.loads(next(g["value"] for g in gt_rows if g["task_id"] == r["task_id"])),
    "G": "; ".join(fmt(e) for e in json.loads(r["G"])), "n_G_own": r["n_G_own"], "n_G_context": r["n_G_context"],
    "difficulty": round(r["difficulty"], 4)} for r in rows])
readable.to_csv(OUT / "tasks_readable.csv", index=False)

counts = pd.crosstab(out.size_G, out.condition)
(OUT / "manifest.json").write_text(json.dumps({
    "task_set_version": VERSION, "derived_from": manifest["task_set_version"], "source_dir": str(SRC.relative_to(ROOT)),
    "snapshot": manifest["snapshot"], "seed": SEED, "p_add": P_ADD, "identifiers": {"name": NAME_KEY, "team": TEAM_KEYS},
    "rule": "for each task of the source set the name and the team are each added with probability p_add, independently; each added identifier replaces a random sampled entry at a random position (|G| unchanged)",
    "tasks": len(rows), "condition_by_size_G": {str(k): v for k, v in counts.to_dict("index").items()},
    "notes": notes, "source_manifest": manifest}, indent=1, ensure_ascii=False, default=int))
print(len(rows), "tasks"); print(counts); print(pd.crosstab(out.condition_assigned, out.condition))
print("notes:", pd.DataFrame(notes).reason.value_counts().to_dict() if notes else {})

"""Derive the football Player task set with an identifier in G: every task of tasks/football/v1_ctx gets the player's
name (Player.Name) in G, everything else unchanged, for a paired test of how much an identifier helps.

    venv/bin/python scripts/football_add_identifier.py

- The name replaces the last sampled entry of G and comes first: |G| stays 1, 3 or 5 (G = the name only when |G| = 1,
  the name and the first 2 or 4 sampled entries otherwise). G_stratum describes the sampled entries; G_dropped records
  the one taken out.
- Tasks whose H is the name are left out (the name would give the answer away); tasks whose G already has it are
  kept unchanged (name_added = false).
- New task_id (hash of the original id); paired_task_id = the original task. Difficulty is recomputed with the name
  in G.
- Leakage is checked as in the generator: the name must not equal or contain the hidden value.

Writes tasks/football/v1_ctx_id/: tasks.csv, ground_truth.csv, tasks_readable.csv, manifest.json."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskdef.difficulty import compute_difficulty, surprise  # noqa: E402
from taskdef.graph import Graph  # noqa: E402
from taskdef.leakage import value_leaks  # noqa: E402

SRC, OUT = ROOT / "tasks/football/v1_ctx", ROOT / "tasks/football/v1_ctx_id"
ID_KEY, ID_NAME = "Player.Name", "Name"
VERSION = "football-v1-ctx-id"

manifest = json.loads((SRC / "manifest.json").read_text())
graph = Graph.load(ROOT / manifest["config"]["dataset"]["snapshot_dir"])
min_len = manifest["config"]["properties"]["leak_min_substring_len"]
players = [graph.nodes[i] for i in graph.nodes_of_type("Player")]
pop_name = sum(ID_NAME in n.props for n in players) / len(players)

tasks = pd.read_csv(SRC / "tasks.csv", dtype={"paired_task_id": str})
truth = pd.read_csv(SRC / "ground_truth.csv")
hidden_value = {(r.task_id, r.prop_key): json.loads(r.value) for r in truth.itertuples()}
rows, gt_rows, skipped = [], [], []
for t in tasks.to_dict("records"):
    H, G = json.loads(t["H"]), json.loads(t["G"])
    node = graph.nodes[t["node_id"]]
    if any(h["prop_key"] == ID_KEY for h in H):
        skipped.append({"task_id": t["task_id"], "reason": "H is the name"})
        continue
    added = not any(e["prop_key"] == ID_KEY for e in G)
    dropped = None
    if added:
        name = node.props[ID_NAME]
        leaks = [h["prop_key"] for h in H if value_leaks(hidden_value[(t["task_id"], h["prop_key"])], name, min_len)]
        if leaks:
            skipped.append({"task_id": t["task_id"], "reason": f"name leaks {leaks}"})
            continue
        cent = t["node_centrality"]
        dropped = G[-1]["prop_key"]
        G = [{"prop_key": ID_KEY, "property_name": ID_NAME, "element_type": "Player", "element_id": t["node_id"], "path": "Player",
              "origin": "node", "value": name, "popularity": pop_name, "centrality": cent, "score": pop_name * cent,
              "hops": 0, "identifier": True}] + G[:-1]
    d = compute_difficulty(t["node_centrality"], [h["score"] for h in H], [e["score"] for e in G])
    new_id = hashlib.sha256(f"{t['task_id']}|{ID_KEY}".encode()).hexdigest()[:16]
    rows.append({**t, "task_id": new_id, "task_set_version": VERSION, "size_G": len(G), "G": json.dumps(G, ensure_ascii=False),
                 "difficulty": d.value, "diff_ratio": d.ratio, "diff_node_surprise": d.node_surprise, "diff_prop_surprise": d.prop_surprise,
                 "diff_prop_surprise_G": sum(surprise(e["score"]) for e in G),
                 "n_G_own": sum(e["origin"] == "node" for e in G), "n_G_context": sum(e["origin"] != "node" for e in G),
                 "G_has_own_property": True, "paired_task_id": t["task_id"],
                 "name_added": added, "G_dropped": dropped})
    gt_rows += truth[truth.task_id == t["task_id"]].assign(task_id=new_id).to_dict("records")

OUT.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(OUT / "tasks.csv", index=False)
pd.DataFrame(gt_rows).to_csv(OUT / "ground_truth.csv", index=False)

# readable version, as tasks/football/v1_ctx/tasks_readable.csv
fmt = lambda e: f"{e['prop_key'] if e['origin'] != 'node' else e['property_name']} = {e['value']}"
readable = pd.DataFrame([{
    "task_id": r["task_id"], "paired_task_id": r["paired_task_id"], "node_id": r["node_id"],
    "player": graph.nodes[r["node_id"]].props.get(ID_NAME), "position": graph.nodes[r["node_id"]].props.get("Position"),
    "node_stratum": r["node_stratum"], "H_stratum": r["H_stratum"], "G_stratum": r["G_stratum"],
    "size_G": r["size_G"], "name_added": r["name_added"], "G_dropped": r["G_dropped"],
    "hidden_property": json.loads(r["H"])[0]["property_name"],
    "true_value": json.loads(next(g["value"] for g in gt_rows if g["task_id"] == r["task_id"])),
    "G": "; ".join(fmt(e) for e in json.loads(r["G"])), "n_G_own": r["n_G_own"], "n_G_context": r["n_G_context"],
    "difficulty": round(r["difficulty"], 4)} for r in rows])
readable.to_csv(OUT / "tasks_readable.csv", index=False)

(OUT / "manifest.json").write_text(json.dumps({
    "task_set_version": VERSION, "derived_from": manifest["task_set_version"], "source_dir": str(SRC.relative_to(ROOT)),
    "snapshot": manifest["snapshot"], "identifier": ID_KEY, "identifier_popularity": pop_name,
    "rule": "each task of the source set with the node's name as the first entry of G, replacing the last sampled entry (|G| unchanged); tasks hiding the name are left out",
    "tasks": len(rows), "name_added": int(out.name_added.sum()), "name_already_in_G": int((~out.name_added).sum()),
    "skipped": skipped, "source_manifest": manifest}, indent=1, ensure_ascii=False))
print(f"{len(rows)} tasks ({int(out.name_added.sum())} with the name added, {int((~out.name_added).sum())} already had it); "
      f"skipped {len(skipped)}: {skipped}")

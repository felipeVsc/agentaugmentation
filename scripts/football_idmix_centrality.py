"""Centrality of the players of the football idmix task sets, as CSV.

    venv/bin/python scripts/football_idmix_centrality.py

Writes, in tasks/football/v1_ctx_idmix/ and tasks/football/v1_noctx_idmix/:

- players_centrality.csv: every Player of the snapshot, with its centrality (normalized PageRank, as the generator),
  stratum (the generator's quartile rule), rank and percentile, and how many tasks of each set it has;
- task_players_centrality.csv: the same rows for the players that have a task in either set.

taskset_role: "drawn" (the node drawn for its stratum) or "replacement" (drawn when a drawn node had too few
candidates), from the source manifest's report."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskdef.graph import Graph  # noqa: E402
from taskdef.sampling import Quartiles  # noqa: E402

CTX, NOCTX = ROOT / "tasks/football/v1_ctx_idmix", ROOT / "tasks/football/v1_noctx_idmix"
m = json.loads((CTX / "manifest.json").read_text())["source_manifest"]
g = Graph.load(ROOT / m["config"]["dataset"]["snapshot_dir"])
q = Quartiles(tuple(m["quartiles"]["node_centrality"]))
on, off = pd.read_csv(CTX / "tasks.csv"), pd.read_csv(NOCTX / "tasks.csv")

c = pd.read_parquet(ROOT / "tasks/football/v1_ctx/intermediate/centrality.parquet")
c = c[(c.element_kind == "node") & (c.element_type == "Player")].copy()
n = len(c)
c["rank"] = c.centrality.rank(ascending=False, method="min").astype(int)
c["percentile"] = c.centrality.rank(pct=True).round(4)
role = {i: "drawn" for v in m["report"]["nodes_selected"].values() for i in v}
role |= {i: "replacement" for v in m["report"]["nodes_replacing"].values() for i in v}
n_on, n_off = on.node_id.value_counts(), off.node_id.value_counts()

rows = []
for r in c.itertuples():
    p = g.nodes[r.element_id].props
    rows.append({"node_id": r.element_id, "player_id": p.get("Player ID"), "name": p.get("Name"), "squad": p.get("Squad"),
                 "nation": p.get("Nation"), "position": p.get("Position"), "centrality_stratum": q.stratum(r.centrality),
                 "centrality": r.centrality, "raw_centrality": r.raw_centrality, f"rank_of_{n}": r.rank,
                 "percentile": r.percentile, "taskset_role": role.get(r.element_id, ""),
                 "tasks_context": int(n_on.get(r.element_id, 0)), "tasks_no_context": int(n_off.get(r.element_id, 0))})
d = pd.DataFrame(rows)
d["in_taskset"] = (d.tasks_context + d.tasks_no_context) > 0
d["centrality_stratum"] = pd.Categorical(d.centrality_stratum, ["low", "mid", "high"], ordered=True)
d = d.sort_values(["centrality_stratum", "centrality", "name"])[
    ["node_id", "player_id", "name", "squad", "nation", "position", "centrality_stratum", "centrality", "raw_centrality",
     f"rank_of_{n}", "percentile", "in_taskset", "taskset_role", "tasks_context", "tasks_no_context"]]

# the strata must be the ones the tasks were drawn in
task_strata = on.drop_duplicates("node_id").set_index("node_id").node_stratum
t = d[d.in_taskset]
assert (t.set_index("node_id").centrality_stratum.astype(str) == task_strata.reindex(t.node_id)).all()
for out in (CTX, NOCTX):
    d.to_csv(out / "players_centrality.csv", index=False)
    t.to_csv(out / "task_players_centrality.csv", index=False)
print(f"{len(d)} players, {len(t)} in the task sets | strata of task players: {t.centrality_stratum.value_counts().to_dict()}"
      f" | tasks: {t.tasks_context.sum()} with context, {t.tasks_no_context.sum()} without")

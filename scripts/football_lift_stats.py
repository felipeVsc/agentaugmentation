"""Derive a football snapshot for a task set: a player's or a team's season statistics become its own properties.

    venv/bin/python scripts/football_lift_stats.py [player|team]     (default: player)

Reads tasks/football/raw/intermediate/snapshot (export of Neo4j, `tasks snapshot --config configs/football_raw.yaml`).
In the graph, the statistics sit on relationships to a League, not on the nodes:

- player: Player-[PLAYER_STATS_IN_LEAGUE]->League. For every player with exactly one such relationship, its properties
  are moved to the player, without the "Player - " prefix. Players in several leagues are left unchanged: their
  statistics differ per league, so none of them is the player's property. Writes tasks/football/snapshot_lifted.
- team: Team-[TEAM_STATS_IN_LEAGUE]->League. Each team plays one domestic league; 62 also play an international
  competition (League.Country = "Inter", e.g. the Champions League). The domestic league's statistics are moved to the
  team, the international ones stay on their relationship (context). Writes tasks/football/snapshot_lifted_team.

The relationship the statistics came from is kept, without properties. Properties that repeat the node's own (player:
Player = Name, Date of birth, Nation, Country = the country of Nation; team: Squad = Team name, Country) are dropped.
Neo4j is not modified."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taskdef.config import DatasetConfig  # noqa: E402
from taskdef.graph import Graph  # noqa: E402
from taskdef.neo4j_source import _remap_node_ids  # noqa: E402

SRC = ROOT / "tasks/football/raw/intermediate/snapshot"
SPEC = {
    "player": {"node": "Player", "stats": "PLAYER_STATS_IN_LEAGUE", "prefix": "Player - ",
               "duplicates": {"Player", "Date of birth", "Nation", "Country"}, "out": ROOT / "tasks/football/snapshot_lifted"},
    "team": {"node": "Team", "stats": "TEAM_STATS_IN_LEAGUE", "prefix": "",
             "duplicates": {"Squad", "Country"}, "out": ROOT / "tasks/football/snapshot_lifted_team"},
}
INTERNATIONAL = "Inter"   # League.Country of the international competitions

which = sys.argv[1] if len(sys.argv) > 1 else "player"
S = SPEC[which]
g = Graph.load(SRC)
stats_of = {}
for e in g.edges.values():
    if e.type == S["stats"]:
        stats_of.setdefault(e.src, []).append(e)


def chosen(edges):
    # the relationship whose statistics become the node's: the only one, or (team) the only domestic one
    if len(edges) == 1:
        return edges[0]
    if which == "team":
        domestic = [e for e in edges if g.nodes[e.dst].props.get("Country") != INTERNATIONAL]
        return domestic[0] if len(domestic) == 1 else None
    return None


counts, clashes, source_league = Counter(), Counter(), Counter()
for nid, edges in stats_of.items():
    edge = chosen(edges)
    if edge is None:
        counts["left unchanged (several leagues)"] += 1
        continue
    node = g.nodes[nid]
    for name, value in edge.props.items():
        if name in S["duplicates"]:
            continue
        new = name[len(S["prefix"]):] if S["prefix"] and name.startswith(S["prefix"]) else name
        if new in node.props:
            clashes[new] += 1
            continue
        node.props[new] = value
        if name in edge.prop_types:
            node.prop_types[new] = edge.prop_types[name]
    edge.props, edge.prop_types = {}, {}
    counts["lifted"] += 1
    counts["lifted, other competitions kept as context"] += len(edges) > 1
    source_league["international" if g.nodes[edge.dst].props.get("Country") == INTERNATIONAL else "domestic"] += 1

if which == "team":
    # the raw export named Players by their fbref id only; name Teams by theirs ("Team ID"), as dataset.node_id_property
    nodes, edges = _remap_node_ids(list(g.nodes.values()), list(g.edges.values()), DatasetConfig(name="football", node_type="Team", node_id_property="Team ID"))
    g = Graph(nodes, edges)

S["out"].mkdir(parents=True, exist_ok=True)
g.save(S["out"])
n_nodes = len(g.nodes_of_type(S["node"]))
report = {"node_type": S["node"], "source": str(SRC.relative_to(ROOT)), "source_hash": Graph.load(SRC).content_hash(),
          "hash": g.content_hash(), "nodes": n_nodes, "without_stats": n_nodes - len(stats_of), **dict(counts),
          "lifted_from": dict(source_league), "dropped_duplicates": sorted(S["duplicates"]), "name_clashes": dict(clashes),
          "stats": g.stats()}
(S["out"] / "lift_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items() if k != "stats"}, indent=1))

"""Exports the lifted football graph (player statistics as Player properties) for sharing, as files Neo4j can load.

    venv/bin/python scripts/export_football_lifted.py

Source: tasks/football/snapshot_lifted (scripts/football_lift_stats.py), the snapshot every football task set was built
from. Neo4j is not read or modified. Writes exports/football_lifted/:

- csv/                    one file per node label and relationship type, with typed headers for neo4j-admin import
- import.sh               neo4j-admin database import full ... into a new, empty database
- football_lifted.cypher  the same graph as a Cypher script (cypher-shell -f), for any running database
- README.md               what was lifted, counts, and how to load it

Players with statistics in exactly one league have them as their own properties; the 2017 players with statistics in
several leagues keep them on PLAYER_STATS_IN_LEAGUE (see football_lift_stats.py). It is the task snapshot as is."""
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tasks/football/snapshot_lifted"
OUT = ROOT / "exports/football_lifted"
TYPE = {"integer": "long", "float": "double", "string": "string"}
BATCH = 500

nodes = pd.read_parquet(SRC / "nodes.parquet")
edges = pd.read_parquet(SRC / "edges.parquet")
label_of = dict(zip(nodes.element_id, nodes.element_type))
report = json.loads((SRC / "lift_report.json").read_text())


if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "csv").mkdir(parents=True)


MIXED = Counter()


def columns(df):
    """Property names and their CSV type per element type. A property that is an integer on some elements and a decimal
    on others becomes a decimal column (Neo4j allows mixed types per element, a CSV column does not); counted in MIXED."""
    seen = defaultdict(lambda: defaultdict(set))
    for et, pt in zip(df.element_type, df.property_types):
        for k, t in json.loads(pt).items():
            seen[et][k].add(t)
    cols = defaultdict(dict)
    for et, props in seen.items():
        for k, ts in props.items():
            if len(ts) == 1:
                cols[et][k] = ts.pop()
            elif ts == {"integer", "float"}:
                cols[et][k] = "float"
                MIXED[f"{et}.{k}"] += 1
            else:
                cols[et][k] = "string"
                MIXED[f"{et}.{k} (as string)"] += 1
    return cols


def cell(v):
    return "" if v is None else (repr(v) if isinstance(v, float) else str(v))


# ---------------------------------------------------------------- CSV for neo4j-admin import
ncols, ecols = columns(nodes), columns(edges)
for et, g in nodes.groupby("element_type"):
    props = sorted(ncols[et])
    with open(OUT / "csv" / f"nodes_{et}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([":ID", ":LABEL"] + [f"{p}:{TYPE[ncols[et][p]]}" for p in props])
        for r in g.itertuples():
            pr = json.loads(r.properties)
            w.writerow([r.element_id, ";".join(json.loads(r.labels))] + [cell(pr.get(p)) for p in props])
for et, g in edges.groupby("element_type"):
    props = sorted(ecols[et])
    with open(OUT / "csv" / f"rels_{et}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([":START_ID", ":END_ID", ":TYPE"] + [f"{p}:{TYPE[ecols[et][p]]}" for p in props])
        for r in g.itertuples():
            pr = json.loads(r.properties)
            w.writerow([r.src_id, r.dst_id, et] + [cell(pr.get(p)) for p in props])

node_files = " \\\n  ".join(f"--nodes=csv/nodes_{et}.csv" for et in sorted(nodes.element_type.unique()))
rel_files = " \\\n  ".join(f"--relationships=csv/rels_{et}.csv" for et in sorted(edges.element_type.unique()))
(OUT / "import.sh").write_text(f"""#!/bin/sh
# Loads the graph into a NEW, empty database (default name: footballlifted) of a stopped or running DBMS.
#   NEO4J_HOME=/path/to/dbms sh import.sh [database]
# Then, with the DBMS running: CREATE DATABASE footballlifted (Enterprise / Desktop), or set it as the default database.
# The id column is only used to connect the relationships; each node keeps its own properties (Player ID, Team ID, ...).
cd "$(dirname "$0")"
DB=${{1:-footballlifted}}
"$NEO4J_HOME/bin/neo4j-admin" database import full "$DB" --overwrite-destination=false --id-type=string \\
  --array-delimiter=";" \\
  {node_files} \\
  {rel_files}
""")


# ---------------------------------------------------------------- Cypher script
def lit(v):
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)   # a JSON string is a valid Cypher string literal
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v)


def mp(d):
    return "{" + ", ".join(f"`{k}`: {lit(v)}" for k, v in d.items()) + "}"


lines = ["// Lifted football graph: player statistics as Player properties. Load into an EMPTY database:",
         "//   cypher-shell -u <user> -p <password> -d <database> -f football_lifted.cypher",
         "// _eid is a temporary key used to connect the relationships; it is removed at the end.", ""]
for et in sorted(nodes.element_type.unique()):
    lines.append(f"CREATE INDEX export_eid_{et} IF NOT EXISTS FOR (n:`{et}`) ON (n._eid);")
lines.append("CALL db.awaitIndexes();")
lines.append("")
for et, g in nodes.groupby("element_type"):
    rows = [{"_eid": r.element_id, **json.loads(r.properties)} for r in g.itertuples()]
    for i in range(0, len(rows), BATCH):
        lines.append(f"UNWIND [{', '.join(mp(x) for x in rows[i:i + BATCH])}] AS r CREATE (n:`{et}`) SET n = r;")
for et, g in edges.groupby("element_type"):
    g = g.assign(sl=g.src_id.map(label_of), dl=g.dst_id.map(label_of))
    for (sl, dl), gg in g.groupby(["sl", "dl"]):
        rows = [{"s": r.src_id, "d": r.dst_id, "p": json.loads(r.properties)} for r in gg.itertuples()]
        for i in range(0, len(rows), BATCH):
            body = ", ".join(f"{{s: {lit(x['s'])}, d: {lit(x['d'])}, p: {mp(x['p'])}}}" for x in rows[i:i + BATCH])
            lines.append(f"UNWIND [{body}] AS r MATCH (a:`{sl}` {{_eid: r.s}}), (b:`{dl}` {{_eid: r.d}}) "
                         f"CREATE (a)-[x:`{et}`]->(b) SET x = r.p;")
for et in sorted(nodes.element_type.unique()):
    lines.append(f"MATCH (n:`{et}`) REMOVE n._eid;")
    lines.append(f"DROP INDEX export_eid_{et} IF EXISTS;")
(OUT / "football_lifted.cypher").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- README
st = report["stats"]
player_props = Counter(k for p in nodes[nodes.element_type == "Player"].properties for k in json.loads(p))
lifted_props = sorted(k for k, n in player_props.items() if k not in {"Player ID", "Name", "Date of birth", "Nation"})
(OUT / "README.md").write_text(f"""# Football graph, player statistics on the Player nodes

Built from the fbref.com 2024-25 football graph (Neo4j). In the original graph a player's season statistics sit on the
`PLAYER_STATS_IN_LEAGUE` relationship to a League. Here they are properties of the Player:

- **{report['lifted']} players** with statistics in exactly one league: the statistics are moved onto the player (the
  `Player - ` prefix is removed). The `PLAYER_STATS_IN_LEAGUE` relationship is kept, without properties.
- **{report['left unchanged (several leagues)']} players** with statistics in several leagues: unchanged, statistics stay
  on each `PLAYER_STATS_IN_LEAGUE` relationship, since they differ per league.
- Statistics that repeat the player's own properties are not copied: {', '.join(report['dropped_duplicates'])}.
- Teams, leagues, countries and the other relationships are unchanged.

This is exactly the graph every football task set of the paper was generated from (snapshot hash `{report['hash'][:16]}…`).

| Nodes | Count | | Relationships | Count |
|---|---|---|---|---|
""" + "\n".join(f"| {a} | {b} | | {c} | {d} |" for (a, b), (c, d) in
                zip(list(st['nodes_per_type'].items()) + [("", "")] * 2, list(st['edges_per_type'].items()))) + f"""

Player properties ({len(player_props)}): {', '.join(sorted(player_props))}.

## Loading

- **Neo4j dump:** `footballlifted.dump` (made with Neo4j 2026.08.1, block format): `neo4j-admin database load
  footballlifted --from-path=<folder with the dump>`, then `CREATE DATABASE footballlifted` (or set it as the default).
  Needs Neo4j 2026.08 or newer.
- **New database, fastest:** `NEO4J_HOME=/path/to/dbms sh import.sh` (neo4j-admin import of `csv/`), then create or
  select the database `footballlifted`.
- **Any running, empty database:** `cypher-shell -u <user> -p <password> -d <database> -f football_lifted.cypher`.

Types: integers as `long`, decimals as `double`, the rest as `string`. Missing values are absent properties,
as in the original graph (checked on 2026-11-10: no NaN, null or empty-string values in Neo4j). In the CSV
files, properties that are integers on some elements and decimals on others are `double` ({len(MIXED)} properties); the
Cypher script and the dump keep each value's own type, as in the original graph.
""", encoding="utf-8")

print(json.dumps({"out": str(OUT.relative_to(ROOT)), "nodes": len(nodes), "relationships": len(edges),
                  "files": sorted(p.name for p in (OUT / "csv").iterdir()),
                  "mixed_type_columns": len(MIXED), "cypher_mb": round((OUT / "football_lifted.cypher").stat().st_size / 1e6, 1)}, indent=1))

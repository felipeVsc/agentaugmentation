"""Builds the few-shot examples of the Figure 2 prompts (1, 3 and 5 shots) for an Algorithm-2 task set.

    venv/bin/python scripts/build_fewshot_examples_v2.py configs/airports_v5_ctx.yaml data/examples/airports_v5.json
    venv/bin/python scripts/build_fewshot_examples_v2.py configs/football_v1_ctx.yaml data/examples/football_v1.json

Like the published examples (an airport code, sometimes the country, then two values), each example is
an airport with:
- input: one identifying property (iata, icao, name or city) plus 0-2 other own properties;
- output: 2 own properties with their database values.

To avoid overfitting the examples to the tasks:
- the airports appear nowhere in the task sets (neither as task node nor as a G element, with or without context);
- they are drawn at random, rotating over the centrality strata (not only hubs);
- the properties are drawn at random, never the ones that cannot be known outside the dataset
  (commercial_flights, distCity, source, type), which only the property descriptions explain;
- each output property appears in one example only, so no value pattern is repeated.
The examples are nested: the 1-shot prompt uses example 1, the 3-shot prompt examples 1-3.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from taskdef.centrality import compute_centrality
from taskdef.config import load_config
from taskdef.context import detect_n1, n1_keys
from taskdef.leakage import value_leaks
from taskdef.pipeline import load_snapshot
from taskdef.popularity import compute_popularity
from taskdef.sampling import STRATA, TaskBuilder, stable_order

# per dataset (cfg.dataset.name): the identifying properties and the ones that can be known outside the dataset
PROFILES = {
    "airports": (["iata", "icao", "name", "city"],
                 ["name", "city", "country", "iata", "icao", "altitude", "latitude", "longitude", "runways",
                  "tz_database_timezone", "timezone", "dst"]),
    # Football Player: the name or the team (Squad), as input only; public facts and the season statistics fbref publishes.
    "football": (["Name", "Squad"],
                 ["Nation", "Position", "Date of birth", "Matches played", "Starts",
                  "Playing time in minutes", "Goals", "Assists", "Yellow cards", "Red cards"]),
}
STRATUM_ORDER = ["high", "mid", "low", "mid", "high"]
N_EXAMPLES = 5
SEED = 7


def entry(c):
    return {"prop_key": c.prop_key, "property_name": c.property_name, "element_type": c.element_type,
            "element_id": c.element_id, "path": c.path, "origin": "node", "value": c.value}


def used_elements(*task_files):
    used = set()
    for f in task_files:
        t = pd.read_csv(f)
        used |= set(t.node_id) | {g["element_id"] for G in t.G for g in json.loads(G)}
    return used


def main(config_path, out_path):
    cfg = load_config(config_path)
    IDENTIFYING, KNOWABLE = PROFILES[cfg.dataset.name]
    graph, _ = load_snapshot(cfg)
    cent = compute_centrality(graph, cfg.centrality)
    builder = TaskBuilder(graph, cfg, cent, compute_popularity(graph, cfg.properties.empty_string_is_null),
                          n1_keys(detect_n1(graph)))
    files = [Path(cfg.output_dir) / "tasks.csv"]
    if cfg.sampling.no_context_output_dir:
        files.append(Path(cfg.sampling.no_context_output_dir) / "tasks.csv")
    used = used_elements(*files)
    node_q = builder.partitions().node
    pools = {s: stable_order([n for n in builder.N if node_q.stratum(cent[n]) == s and n not in used],
                             SEED, f"example|{s}", key=str) for s in STRATA}
    min_len = cfg.properties.leak_min_substring_len

    examples, taken, outputs = [], set(), set()
    for i, stratum in enumerate(STRATUM_ORDER[:N_EXAMPLES]):
        for node_id in pools[stratum]:
            if node_id in taken:
                continue
            own = {c.property_name: c for c in builder.candidates(node_id) if builder._is_own(c, node_id)}
            ident = [p for p in stable_order(IDENTIFYING, SEED, f"ident|{node_id}", key=str) if p in own]
            if not ident:
                continue
            n_extra = int(stable_order(["0", "1", "2"], SEED, f"extra|{node_id}", key=str)[0])
            rest = [p for p in stable_order(KNOWABLE, SEED, f"in|{node_id}", key=str) if p in own and p != ident[0]]
            given = [ident[0]] + rest[:n_extra]
            out = [p for p in stable_order(KNOWABLE, SEED, f"out|{node_id}", key=str)
                   if p in own and p not in given and p not in outputs
                   and not any(value_leaks(own[p].value, own[g].value, min_len) for g in given)][:2]
            if len(out) < 2:
                continue
            H = [own[p] for p in out]
            examples.append({
                "node_id": node_id, "node_type": cfg.dataset.node_type, "centrality_stratum": stratum,
                "H": [{"prop_key": h.prop_key, "property_name": h.property_name, "element_type": h.element_type,
                       "path": h.path} for h in H],
                "G": [entry(own[p]) for p in given],
                "answers": {h.prop_key: h.value for h in H},
            })
            taken.add(node_id)
            outputs.update(out)
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"version": "v2", "task_set": cfg.version, "seed": SEED,
                                          "examples": examples}, ensure_ascii=False, indent=1))
    print(f"wrote {len(examples)} examples to {out_path}")
    for e in examples:
        print(" ", e["node_id"], e["centrality_stratum"], "| in:", {g["property_name"]: g["value"] for g in e["G"]},
              "| out:", {k.split(".", 1)[1]: v for k, v in e["answers"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])

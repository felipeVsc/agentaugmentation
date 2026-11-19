"""Builds the few-shot examples file for Strategy A on an Algorithm-2 task set (LLM_SPEC.md §5.4).

    venv/bin/python scripts/build_fewshot_examples.py configs/airports_v3_ctx.yaml data/examples/airports_v3.json

Each example is a solved task on an airport that appears nowhere in the task set (neither as task node
nor as a G element), most central first: H = one own property (rotating over HIDDEN), G = the airport's
name plus 2 context properties from its n:1 path (city, country), with the database values as answers.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from taskdef.centrality import compute_centrality
from taskdef.config import load_config
from taskdef.context import detect_n1, hops, n1_keys
from taskdef.leakage import value_leaks
from taskdef.pipeline import load_snapshot
from taskdef.popularity import compute_popularity
from taskdef.sampling import TaskBuilder

HIDDEN = ["iata", "altitude", "city", "timezone", "runways"]
N_EXAMPLES = 5


def g_entry(c, origin):
    return {"prop_key": c.prop_key, "property_name": c.property_name, "element_type": c.element_type,
            "element_id": c.element_id, "path": c.path, "origin": origin, "value": c.value}


def main(config_path, out_path):
    cfg = load_config(config_path)
    graph, _ = load_snapshot(cfg)
    cent = compute_centrality(graph, cfg.centrality)
    builder = TaskBuilder(graph, cfg, cent, compute_popularity(graph, cfg.properties.empty_string_is_null),
                          n1_keys(detect_n1(graph)))
    tasks = pd.read_csv(Path(cfg.output_dir) / "tasks.csv")
    used = set(tasks.node_id) | {g["element_id"] for G in tasks.G for g in json.loads(G)}

    examples = []
    for node_id in sorted(builder.N, key=lambda n: -cent[n]):
        if node_id in used:
            continue
        cands = builder.candidates(node_id)
        own = {c.property_name: c for c in cands if builder._is_own(c, node_id)}
        hidden = HIDDEN[len(examples) % len(HIDDEN)]
        if hidden not in own or "name" not in own:
            continue
        h = own[hidden]
        ctx = [c for c in cands if not builder._is_own(c, node_id) and c.path.startswith(f"{cfg.dataset.node_type}-[")
               and hops(c.path) <= 2 and c.element_kind == "node" and c.property_name == "name"
               and not value_leaks(h.value, c.value, cfg.properties.leak_min_substring_len)]
        if len(ctx) < 2 or value_leaks(h.value, own["name"].value, cfg.properties.leak_min_substring_len):
            continue
        ctx = sorted(ctx, key=lambda c: hops(c.path))[:2]
        examples.append({
            "node_id": node_id, "node_type": cfg.dataset.node_type,
            "H": [{"prop_key": h.prop_key, "property_name": h.property_name, "element_type": h.element_type, "path": h.path}],
            "G": [g_entry(own["name"], "node")] + [g_entry(c, "context") for c in ctx],
            "answers": {h.prop_key: h.value},
        })
        if len(examples) == N_EXAMPLES:
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"version": "v1", "task_set": cfg.task_set_version, "examples": examples},
                                         ensure_ascii=False, indent=1))
    print(f"wrote {len(examples)} examples to {out_path}")
    for e in examples:
        print(" ", e["node_id"], "| G:", "; ".join(f"{g['path']}.{g['property_name']}={g['value']}" for g in e["G"]),
              "| H:", e["answers"])


if __name__ == "__main__":
    main(*sys.argv[1:3])

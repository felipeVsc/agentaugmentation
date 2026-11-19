"""pop(p) of the target node type's properties, with their H stratum and how often each is hidden, for the airports
v5 and football idmix task sets (from the task-set manifests; pop(p) = share of the type's nodes with a non-null value).

    venv/bin/python scripts/property_popularity.py

Writes tasks/<dataset>/<set>/property_popularity.csv: property, popularity, H_stratum (quartiles of pop over the
type's properties), and tasks_hidden (tasks of the set whose H is that property)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SETS = ["airports/v5_ctx", "airports/v5_noctx", "football/v1_ctx_idmix", "football/v1_noctx_idmix"]

for name in SETS:
    d = ROOT / "tasks" / name
    m = json.loads((d / "manifest.json").read_text())
    while "strata" not in m:   # a derived set keeps its source's manifest
        m = m["source_manifest"]
    stratum = {p: s for s, ps in m["strata"]["H_property"].items() for p in ps}
    hidden = pd.read_csv(d / "tasks.csv", dtype=str).H.map(lambda h: json.loads(h)[0]["property_name"]).value_counts()
    t = pd.DataFrame({"property": list(m["strata"]["H_property_popularity"]),
                      "popularity": list(m["strata"]["H_property_popularity"].values())})
    t["H_stratum"] = t.property.map(stratum)
    t["tasks_hidden"] = t.property.map(hidden).fillna(0).astype(int)
    t.sort_values(["popularity", "property"], ascending=[False, True]).to_csv(d / "property_popularity.csv", index=False)
    print(f"{name}: {len(t)} properties, {t.tasks_hidden.sum()} tasks; strata {t.H_stratum.value_counts().to_dict()}")

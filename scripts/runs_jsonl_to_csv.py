"""Flattens a runs.jsonl (one JSON record per run) into a CSV with one row per run, written next to it as runs.csv.

    venv/bin/python scripts/runs_jsonl_to_csv.py [--tasks tasks/airports/v5_ctx/tasks.csv] results/airports/v5_ctx/strategy_a/runs.jsonl [...]

Nested objects become dotted columns (usage.tokens_in, effort.template.tokens, ...). predictions, label_map, parsed and
parse_events stay JSON strings (their keys vary by run). The first LLM call is expanded into call.* columns (its request is the same as the
system/user columns, so it is left out); any further calls are kept as JSON in extra_calls.

--tasks adds the given properties of each run's task, after label_map (runs.jsonl only describes H):
G (the task set's G entries as JSON: prop_key, property_name, element_type = node label or relationship type, path,
origin = node / context, hops, value, popularity, ...), G_props (JSON list of "element_type.property_name"),
size_G, G_stratum, n_G_own and n_G_context.
"""
import csv
import json
import sys
from pathlib import Path

AS_JSON = {"predictions", "label_map", "parse_events", "parsed"}
G_COLUMNS = ["G", "G_props", "size_G", "G_stratum", "n_G_own", "n_G_context"]


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in AS_JSON:
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (dict, list)):
            out[prefix + k] = json.dumps(v, ensure_ascii=False)
        else:
            out[prefix + k] = v
    return out


def row(r):
    calls = r.pop("calls", []) or []
    out = flatten(r)
    if calls:
        first = dict(calls[0])
        first.pop("request", None)
        out.update(flatten(first, "call."))
    out["extra_calls"] = json.dumps(calls[1:], ensure_ascii=False) if len(calls) > 1 else ""
    return out


def load_g(tasks_csv):
    csv.field_size_limit(sys.maxsize)
    g = {}
    with open(tasks_csv, newline="", encoding="utf-8") as fh:
        for t in csv.DictReader(fh):
            entries = json.loads(t["G"])
            g[t["task_id"]] = {"G": t["G"],
                               "G_props": json.dumps([f"{e['element_type']}.{e['property_name']}" for e in entries], ensure_ascii=False),
                               **{k: t[k] for k in G_COLUMNS[2:]}}
    return g


def convert(path, g=None):
    path = Path(path)
    rows, columns = [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = row(json.loads(line))
                if g is not None:
                    extra = g[r["task_id"]]  # KeyError: the run is not from this task set
                    r = {k2: v2 for k, v in r.items() for k2, v2 in ([(k, v)] + (list(extra.items()) if k == "label_map" else []))}
                rows.append(r)
                columns.update(dict.fromkeys(rows[-1]))
    out = path.with_suffix(".csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows, {len(columns)} columns")


if __name__ == "__main__":
    args, g = sys.argv[1:], None
    if args[:1] == ["--tasks"]:
        g, args = load_g(args[1]), args[2:]
    for p in args:
        convert(p, g)

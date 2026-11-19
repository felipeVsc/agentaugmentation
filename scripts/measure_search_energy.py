"""Measures the energy of a local stand-in for one Strategy B web search (Tavily, advanced depth).

    venv/bin/python scripts/measure_search_energy.py results/airports/v2_ctx/strategy_b/runs.jsonl \
        --queries 40 --out results/airports/search_energy/

What a Tavily advanced search does, and what is measured here:

1. query the search index                      -> not measurable (the provider's index); literature value
2. fetch the result pages                      -> measured: the real result URLs of Strategy B's queries
3. extract the text, split it into chunks      -> measured
4. rank the chunks against the query           -> measured: embeddings (BAAI/bge-small-en-v1.5, ONNX, CPU),
   and keep the best chunks per page              cosine similarity, top 3 chunks per page
5. send the pages over the network             -> not measured; estimated from the bytes downloaded

Energy is the process's own CPU energy counter (``proc_pid_rusage`` RUSAGE_INFO_V6 ``ri_energy_nj``, macOS on Apple
Silicon; no sudo needed). It covers every thread of this process, so idle waiting on the network costs nothing and
no baseline is subtracted. The embedding model is loaded once, before the measured part.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
import numpy as np
from bs4 import BeautifulSoup

CHUNK_CHARS = 500
CHUNKS_PER_SOURCE = 3
FETCH_TIMEOUT_S = 10
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15"


class _RusageV6(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint "
        "proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups "
        "child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
        "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility "
        "cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time "
        "serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
        "serviced_energy interval_max_phys_footprint runnable_time flags user_ptime system_ptime pinstructions "
        "pcycles energy_nj penergy_nj secure_time_in_system secure_ptime_in_system neural_footprint "
        "lifetime_max_neural_footprint interval_max_neural_footprint").split()] + [("reserved", ctypes.c_uint64 * 9)]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def energy_j() -> float:
    """Energy used so far by this process (all threads), in joules."""
    ru = _RusageV6()
    if _libproc.proc_pid_rusage(os.getpid(), 6, ctypes.byref(ru)) != 0:
        raise OSError("proc_pid_rusage failed (needs macOS 12+)")
    return ru.energy_nj / 1e9


class Meter:
    """Energy and wall time of a block: ``with meter("fetch"): ...``."""

    def __init__(self):
        self.parts: dict[str, float] = {}
        self.seconds: dict[str, float] = {}

    def __call__(self, name):
        meter = self

        class _Block:
            def __enter__(self):
                self.e, self.t = energy_j(), time.perf_counter()

            def __exit__(self, *exc):
                meter.parts[name] = meter.parts.get(name, 0.0) + energy_j() - self.e
                meter.seconds[name] = meter.seconds.get(name, 0.0) + time.perf_counter() - self.t

        return _Block()


def load_queries(runs_jsonl: Path, n: int, seed: int) -> list[dict]:
    """Distinct Strategy B queries that returned results, with their (not blocked) result URLs."""
    seen, out = set(), []
    for line in open(runs_jsonl, encoding="utf-8"):
        r = json.loads(line)
        urls_by_q: dict[str, list[str]] = {}
        for x in r.get("results", []):
            if not x.get("blocked"):
                urls_by_q.setdefault(x["query_id"], []).append(x["url"])
        for q in r.get("queries", []):
            if q.get("n_results") and q["query"] not in seen and urls_by_q.get(q["id"]):
                seen.add(q["query"])
                out.append({"query": q["query"], "urls": urls_by_q[q["id"]][:5]})
    random.Random(seed).shuffle(out)
    return out[:n]


def fetch(client: httpx.Client, url: str) -> tuple[str, int, str | None]:
    try:
        r = client.get(url)
        return r.text if r.status_code == 200 else "", len(r.content), None if r.status_code == 200 else str(r.status_code)
    except httpx.HTTPError as exc:
        return "", 0, type(exc).__name__


def chunks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)] if text else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_jsonl", type=Path)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/airports/search_energy"))
    args = ap.parse_args()

    from fastembed import TextEmbedding

    e0 = energy_j()
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    list(model.embed(["warm up"]))
    load_j = energy_j() - e0

    queries = load_queries(args.runs_jsonl, args.queries, args.seed)
    rows = []
    with httpx.Client(timeout=FETCH_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client, \
            cf.ThreadPoolExecutor(max_workers=5) as pool:
        for i, q in enumerate(queries, 1):
            m = Meter()
            with m("fetch"):
                pages = list(pool.map(lambda u: fetch(client, u), q["urls"]))
            with m("extract"):
                page_chunks = [chunks(html) for html, _, _ in pages]
            with m("rank"):
                flat = [c for cs in page_chunks for c in cs]
                kept = 0
                if flat:
                    vecs = np.array(list(model.embed(flat)))
                    qv = np.array(list(model.query_embed(q["query"])))[0]
                    sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-9)
                    start = 0
                    for cs in page_chunks:
                        kept += min(CHUNKS_PER_SOURCE, len(cs))
                        np.argsort(-sims[start:start + len(cs)])[:CHUNKS_PER_SOURCE]
                        start += len(cs)
            rows.append({"query": q["query"], "urls": len(q["urls"]), "pages_ok": sum(1 for p in pages if p[2] is None),
                         "errors": [p[2] for p in pages if p[2]], "bytes": sum(p[1] for p in pages),
                         "chunks": len(flat), "chunks_kept": kept,
                         **{f"{k}_j": v for k, v in m.parts.items()}, **{f"{k}_s": v for k, v in m.seconds.items()}})
            print(f"{i}/{len(queries)} {rows[-1]['pages_ok']}/{rows[-1]['urls']} pages, {rows[-1]['bytes'] / 1e3:.0f} kB, "
                  f"{sum(m.parts.values()):.2f} J")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "meta.json").write_text(json.dumps({
        "model": "BAAI/bge-small-en-v1.5 (fastembed, ONNX, CPU)", "model_load_j": load_j, "chunk_chars": CHUNK_CHARS,
        "chunks_per_source": CHUNKS_PER_SOURCE, "queries": len(rows), "seed": args.seed,
        "energy_counter": "proc_pid_rusage RUSAGE_INFO_V6 ri_energy_nj (process CPU energy, all threads)",
        "machine": os.uname().machine}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

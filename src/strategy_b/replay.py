"""Replay the synthesis call of finished Strategy B runs with another synthesis prompt, without searching again.

For every ok run of ``source_variant`` in ``source_dir/runs.jsonl``, the run of the same task, model and run_index
is made with the plan, the searches and the evidence of that run, and only the synthesis call is new. The two
variants then differ only in the synthesis prompt (e.g. evidence only vs evidence and memory).

The plan call and the searches of the source run are kept in the record and counted in usage (tokens, cost,
search calls), so a replayed run costs what the whole pipeline would have; the event ``replayed_from:<variant>``
marks it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .graph import AgentContext, build_replay_graph
from .runner import Job, Runner


def load_source(path: Path, variant: str) -> dict[tuple[str, str, int], dict]:
    """Ok runs of ``variant`` by (task_id, model, run_index). A line still being written is ignored."""
    runs = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            r = json.loads(line)
            if r["variant"] == variant and r["status"] == "ok":
                runs[(r["task_id"], r["model"], r["run_index"])] = r
    return runs


class _Replay:
    """Stands in for the agent graph of ``Runner``: builds the state of the source run, then synthesizes."""

    def __init__(self, graph, source: dict, model_id: str, source_variant: str):
        self.graph, self.source, self.model_id, self.source_variant = graph, source, model_id, source_variant

    def invoke(self, state: dict) -> dict:
        task, run_index = state["task"], state["run_index"]
        src = self.source[(task.task_id, self.model_id, run_index)]
        plan_calls = [{**c, "replayed": True} for c in src["calls"] if c["step"].startswith("plan")]
        state = {
            **state,
            "labels": src["label_map"],
            "queries": src["plan_parsed"],
            "plan_parsed": src["plan_parsed"],
            "plan_entity": src["plan_entity"],
            "search_log": src["queries"],
            "results": src["results"],
            "evidence": src["evidence_passed"],
            "calls": plan_calls,
            "events": [f"replayed_from:{self.source_variant}"],
        }
        return self.graph.invoke(state)


class ReplayRunner(Runner):
    def __init__(self, cfg, source_dir: str | Path, source_variant: str, **kw):
        super().__init__(cfg, **kw)
        self.source_path = cfg.path(str(source_dir)) / "runs.jsonl"
        self.source_variant = source_variant

    def prepare(self) -> None:
        if self._prepared:
            return
        super().prepare()
        self.source = load_source(self.source_path, self.source_variant)
        self.graphs = {
            (v.id, m.id): _Replay(
                build_replay_graph(AgentContext(self.cfg, v, self.callers[m.id], self.search, self.plan_tpls[v.id],
                                                self.synth_tpls[v.id], self.tokenizers[m.id].count, self.notes)),
                self.source, m.id, self.source_variant)
            for v in self.variants for m in self.models
        }

    def jobs(self) -> list[Job]:
        """Only the runs that have a finished source run."""
        return [j for j in super().jobs() if (j.task.task_id, j.model.id, j.run_index) in self.source]

"""Small synthetic evidence probes. Optional original sources arrive via stdin, never HEAD."""

from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from statistics import median
from unittest.mock import patch
import hashlib
import inspect
import json
import sys


class Rows(list):
    def __init__(self, values=()):
        super().__init__(values)
        self.visits = 0
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        for value in super().__iter__():
            self.visits += 1
            yield value

    def reset(self):
        self.visits = 0
        self.passes = 0


def budget_state(decisions=80, tasks=400, rewards=80):
    return {
        "action_records": Rows(
            {
                "record_id": f"d{i}",
                "final_action": {"round_budget_review": {"choice": "new_search"}},
                "execution_result": {
                    "result": {
                        "tasks": [
                            {"task_id": str(i % tasks)},
                            {"task_id": str(i % tasks)},
                            {"task_id": "missing"},
                        ]
                    }
                },
            }
            for i in range(decisions)
        ),
        "tasks": Rows(
            {
                "task_id": str(i),
                "parent_decision_id": f"d{i % decisions}",
                "model_version": f"m{i % 2}",
                "status": "failed" if i % 11 == 0 else "completed",
                "actual_cost": None if i % 13 == 0 else (i % 5) / 10,
            }
            for i in range(tasks)
        ),
        "rewards": Rows(
            {
                "batch_id": f"r{i // 2}",
                "task_ids": [str(i % tasks), str((i * 17) % tasks), str(i % tasks)],
                "reward": i / 100,
                "model_version": f"m{i % 2}",
            }
            for i in range(rewards)
        ),
    }


def stages_state(groups):
    return {
        "tasks": Rows(
            {
                "task_id": str(i),
                "model_version": f"m{i % 2}",
                "search_group_index": i % groups,
                "stage": "deep_search",
                "status": "failed" if i % 11 == 0 else "completed",
            }
            for i in range(4000)
        ),
        "processed_task_ids": Rows(str(i) for i in range(0, 4000, 2)),
    }


def measured(name, fn, state):
    times = []
    for _ in range(3):
        for value in state.values():
            if isinstance(value, Rows):
                value.reset()
        start = perf_counter()
        result = fn(state)
        times.append(perf_counter() - start)
    return {
        "case": name,
        "median_seconds": median(times),
        "visits": {
            key: {"passes": v.passes, "items": v.visits}
            for key, v in state.items()
            if isinstance(v, Rows)
        },
        "result_count": len(result["groups"]) if isinstance(result, dict) else len(result),
        "result_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }


def run(budget, rounds):
    results = []
    for d, t, r in [
        (8, 400, 80),
        (80, 400, 80),
        (240, 400, 80),
        (80, 100, 80),
        (80, 1000, 80),
        (80, 4000, 80),
        (80, 400, 800),
    ]:
        state = budget_state(d, t, r)
        results.append(measured(f"budget-{d}-{t}-{r}", budget, state))
    state = budget_state(240, 400, 80)

    def recent(s):
        return (
            budget(s, limit=8)
            if "limit" in inspect.signature(budget).parameters
            else budget(s)[-8:]
        )

    results.append(measured("budget-recent-8", recent, state))
    for groups in [1, 20, 100]:
        results.append(measured(f"round-groups-{groups}", rounds, stages_state(groups)))
    return results


def quality_probe(root):
    from tests.test_phase_quality import sample
    from phase_agent.analysis.phase import phase_quality_reports as q
    from phase_agent.evidence.build_decision_context import build_decision_context

    state = sample()
    state["phase_diagrams"]["mlip"]["entries"][0]["structure_path"] = str(root / "missing.vasp")
    state["tasks"] = [
        {"task_id": str(i), "model_version": "m1", "status": "completed", "diagnostic": "x" * 64}
        for i in range(1024)
    ]
    stages = {}

    def wrap(label, callback):
        def call(*args, **kwargs):
            start = perf_counter()
            try:
                return callback(*args, **kwargs)
            finally:
                row = stages.setdefault(label, {"calls": 0, "seconds": 0})
                row["calls"] += 1
                row["seconds"] += perf_counter() - start

        return call

    class Cache(OrderedDict):
        def get(self, *args):
            return wrap("cache_lookup", super().get)(*args)

    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        return wrap("file_stat", original_stat)(path, *args, **kwargs)

    original_digest = q._digest

    def digest(value):
        return wrap(
            "fingerprint" if "snapshot" in value and "tasks" in value else "report_checksum",
            original_digest,
        )(value)

    result = []
    with (
        patch.object(q, "_REPORT_CACHE", Cache()),
        patch.object(Path, "stat", stat),
        patch.object(q, "_digest", digest),
        patch.object(q, "diagnose_phase_diagram", wrap("diagnose", q.diagnose_phase_diagram)),
    ):
        for label in ["first", "memory_hit"]:
            stages.clear()
            start = perf_counter()
            report = q.get_phase_quality_report(state)
            result.append(
                {
                    "case": label,
                    "seconds": perf_counter() - start,
                    "stages": deepcopy(stages),
                    "status": report["status"],
                }
            )
    failed = sample()
    failed["qbc_candidates"] = [
        {"candidate_id": "c", "model_version": "m1", "ehull": 0.01, "qbc": {"status": "completed"}}
    ]
    with patch.object(
        q, "get_phase_quality_report", side_effect=ValueError("synthetic unavailable")
    ) as get:
        context = build_decision_context(failed)
    result.append(
        {
            "case": "failed_context",
            "get_calls": get.call_count,
            "status": context["current_phase_diagram"]["mlip"]["quality_diagnostics"]["status"]
            if "current_phase_diagram" in context
            else None,
        }
    )
    return result


if __name__ == "__main__":
    from phase_agent.evidence.round_budget_evidence import budget_decision_outcomes
    from phase_agent.evidence.summarize_task_rounds import summarize_task_rounds

    if "--reference-stdin" in sys.argv:
        sources = json.load(sys.stdin)
        old = {}
        for path, source in sources.items():
            namespace = {}
            exec(compile(source, path, "exec"), namespace)
            old[Path(path).stem] = namespace
        old["summarize_task_rounds"]["summarize_task_stages"] = old["summarize_task_stages"][
            "summarize_task_stages"
        ]
        reference = run(
            old["round_budget_evidence"]["budget_decision_outcomes"],
            old["summarize_task_rounds"]["summarize_task_rounds"],
        )
    else:
        reference = None
    current = run(budget_decision_outcomes, summarize_task_rounds)
    if reference is not None:
        assert all(a["result_sha256"] == b["result_sha256"] for a, b in zip(reference, current))
    with TemporaryDirectory(prefix="phase-small-evidence-") as directory:
        print(
            json.dumps(
                {
                    "reference": reference,
                    "current": current,
                    "quality": quality_probe(Path(directory)),
                }
            )
        )

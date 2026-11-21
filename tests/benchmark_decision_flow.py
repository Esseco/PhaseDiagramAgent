"""Synthetic native decision-flow metrics; no model, project copy or scientific execution."""

import inspect
import json
import statistics
import time
from tempfile import TemporaryDirectory
from pathlib import Path

from tests.test_codex_native_team import make_control, team, advice


def measure():
    runs = []
    for _ in range(5):
        with TemporaryDirectory(prefix="phase-flow-") as directory:
            control, _context = make_control(Path(directory))
            t = team(control)
            calls = 0
            experts = 0
            response_bytes = 0

            def call(method, *args, **kwargs):
                nonlocal calls, response_bytes
                calls += 1
                value = method(*args, **kwargs)
                response_bytes += len(
                    json.dumps(
                        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf8")
                )
                return value

            started = time.perf_counter()
            view = call(t.start, question="Compare routes", consult=["exploration", "reliability"])
            for role in view["consult"]:
                call(t.advise, view["team_id"], 1, advice(view, role), compact=True)
                experts += 1
            call(
                t.draft,
                view["team_id"],
                1,
                {"tool": "pause_search"},
                rationale="Review current evidence",
            )
            if "replace_draft" in inspect.signature(t.draft).parameters:
                view = call(
                    t.draft,
                    view["team_id"],
                    1,
                    {"tool": "pause_search"},
                    rationale="Explain the same decision more precisely",
                    replace_draft=True,
                )
                reconsults = 0
                invalidation = "none: same source, question and active guidance"
            else:
                view = call(
                    t.feedback,
                    view["team_id"],
                    1,
                    "revise",
                    "Explain the same decision more precisely",
                )
                for role in view["consult"]:
                    call(
                        t.advise,
                        view["team_id"],
                        view["revision"],
                        advice(view, role),
                        compact=True,
                    )
                    experts += 1
                view = call(
                    t.draft,
                    view["team_id"],
                    view["revision"],
                    {"tool": "pause_search"},
                    rationale="Explain the same decision more precisely",
                )
                reconsults = 2
                invalidation = "revision: all opinions cleared"
            runs.append(
                dict(
                    seconds=time.perf_counter() - started,
                    tool_roundtrips=calls,
                    expert_calls=experts,
                    reconsultations=reconsults,
                    clarification_rounds=0,
                    invalidation_reason=invalidation,
                    first_draft_passed=True,
                    response_bytes=response_bytes,
                    final_status=view["status"],
                    real_model_calls=0,
                )
            )
    return {
        "scenario": "same-facts unaccepted draft explanation edit",
        "runs": runs,
        "median_seconds": statistics.median(r["seconds"] for r in runs),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), ensure_ascii=False))

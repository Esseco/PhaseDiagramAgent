"""Bounded synthetic hotspot probes. No real model, project or scientific action."""

import gc
import json
import sqlite3
import threading
import time
from copy import deepcopy
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import psutil


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode())


def measured(name, callback, counters=None):
    gc.collect()
    process = psutil.Process()
    before = process.memory_info()
    samples = [before.rss]
    private = [getattr(before, "private", 0)]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            m = process.memory_info()
            samples.append(m.rss)
            private.append(getattr(m, "private", 0))

    worker = threading.Thread(target=sample)
    worker.start()
    started = time.perf_counter()
    try:
        output = callback()
        elapsed = time.perf_counter() - started
        output_bytes = size(output)
    finally:
        stop.set()
        worker.join()
    del output
    gc.collect()
    end = process.memory_info()
    samples.append(end.rss)
    return dict(
        case=name,
        seconds=elapsed,
        rss_before=before.rss,
        rss_peak=max(samples),
        rss_end=end.rss,
        private_peak=max(private),
        output_bytes=output_bytes,
        counts=deepcopy(counters or {}),
    )


def run(root):
    from phase_agent.agents.strategic.codex_team import CodexTeam
    from phase_agent.agents.strategic.context import with_advice
    from phase_agent.graphs.business_checkpoint import checkpoint_frame
    from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde
    from phase_agent.persistence.compact_checkpoints import CompressedSerde
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    results = []
    for revisions in [1, 10, 100]:
        events = [
            {
                "revision": i,
                "kind": kind,
                "advice": {"assessment": "x" * 2048},
                "plan": {"checks": ["x" * 2048]},
                "action": {"tool": "pause_search"},
            }
            for i in range(revisions)
            for kind in ["expert_advice", "draft", "simulation_plan", "human_revise"]
        ]
        session = dict(
            team_id="team-" + "a" * 32,
            revision=revisions,
            consult=[],
            question="q",
            advice=[{"role": "exploration", "assessment": "current"}],
            guidance=["original"] * revisions,
            events=events,
            binding="b",
            draft={"action": {"tool": "pause_search"}, "rationale": "r", "memory_ids": []},
        )
        results.append(measured(f"audit-revisions-{revisions}", lambda: CodexTeam._action(session)))
    for count in [512, 4096, 16384]:
        rows = [
            {"task_id": str(i), "values": list(range(16)), "payload": "x" * 256}
            for i in range(count)
        ]
        state = {"tasks": rows}
        frame = {"loaded_state": state, "result": {"state": state}, "snapshot": {"state": state}}
        old = checkpoint_frame(frame)
        results.append(
            measured(
                f"project-unchanged-{count}", lambda: {"same": checkpoint_frame(frame, old) == old}
            )
        )
        changed = deepcopy(frame)
        changed["loaded_state"]["tasks"][-1]["status"] = "completed"
        results.append(
            measured(
                f"project-local-change-{count}",
                lambda: {"same": checkpoint_frame(changed, old) == old},
            )
        )
        payload = {"state": {}, "decision_context": {"rows": rows}, "instruction": "q"}
        results.append(
            measured(
                f"with-advice-{count}",
                lambda: with_advice(payload, {"snapshot_id": "s", "facts": {}}, []),
            )
        )
        with closing(sqlite3.connect(":memory:")) as db:
            serde = CheckpointBlobSerde(
                CompressedSerde(JsonPlusSerializer()), db, threading.RLock()
            )
            counts = {"encode": 0, "hash": 0, "compressor": 0}
            import phase_agent.persistence.checkpoint_blobs as blobs

            original_encode = JsonPlusSerializer.dumps_typed
            original_hash, original_compress = blobs.sha256, blobs.zlib.compressobj

            def encode(self, value):
                counts["encode"] += 1
                return original_encode(self, value)

            def digest(*args, **kwargs):
                counts["hash"] += 1
                return original_hash(*args, **kwargs)

            def compress(*args, **kwargs):
                counts["compressor"] += 1
                return original_compress(*args, **kwargs)

            with (
                patch.object(JsonPlusSerializer, "dumps_typed", encode),
                patch.object(blobs, "sha256", digest),
                patch.object(blobs.zlib, "compressobj", compress),
            ):
                for repeat in range(4):
                    for k in counts:
                        counts[k] = 0

                    def write():
                        kind, wire = serde.dumps_typed(frame)
                        db.commit()
                        return {
                            "wire_bytes": len(wire),
                            "stored_blobs": db.execute(
                                "select count(*) from checkpoint_payloads"
                            ).fetchone()[0],
                            "kind": kind,
                        }

                    results.append(measured(f"checkpoint-{count}-{repeat}", write, counts))
    from phase_agent.runtime.codex.reports import save_codex_report

    report_root = root / "report"
    (report_root / "workflow_state").mkdir(parents=True)
    state = {
        "confirmed_config": {"storage": {"workspace_root": str(report_root)}},
        "pending_execution_policies": {
            "p": {
                "agent_proposal": {
                    "recommended_action": "pause_search",
                    "raw_action": {"tool": "pause_search", "reason": "review", "parameters": {}},
                }
            }
        },
    }
    path = report_root / "workflow_state" / "state.json"
    path.write_text(json.dumps(state))
    result = {"status": "awaiting_approval", "record_id": "p", "state": state}
    for repeat in range(3):
        results.append(
            measured(f"report-{repeat}", lambda: {"path": save_codex_report(path, result)})
        )
    results.append(
        {"case": "report-artifacts", "files": sum(p.is_file() for p in report_root.rglob("*"))}
    )
    return results


def retention_and_timeline(root):
    from phase_agent.agents.strategic.context import with_advice
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.runtime.codex.reports import save_codex_report
    from tests.test_event_loop import _session
    from tests.test_decision_team import FakeTeam
    from tests.test_simulation_agent import SimulationModel

    retention = []
    payload = {
        "state": {},
        "decision_context": {"rows": [{"id": i, "values": list(range(16))} for i in range(4096)]},
    }
    for i in range(20):
        retention.append(
            measured(
                f"retention-{i}",
                lambda: with_advice(payload, {"snapshot_id": "s", "facts": {}}, []),
            )
        )
    project = root / "timeline"
    state_path = project / "workflow_state" / "state.json"
    state_path.parent.mkdir(parents=True)
    write_json(state_path, {})
    stages = []
    models = []

    class Client(FakeTeam):
        supports_simulation = True

        def __call__(self, request):
            started = time.perf_counter()
            if request["mode"].startswith("simulation_"):
                result = SimulationModel()(request)
            else:
                result = super().__call__(request)
                if request["mode"] == "autonomous_search":
                    result.update(tool="check_convergence", parameters={}, budget=0)
            models.append(
                {
                    "mode": request["mode"],
                    "role": request.get("agent_role"),
                    "seconds": time.perf_counter() - started,
                    "input_bytes": size(request),
                }
            )
            return result

    state = read_json(state_path)
    stages.append(measured("read", lambda: read_json(state_path)))
    result_holder = {}
    executed = []

    def propose():
        result = run_tool_step(
            state,
            _session(),
            agent_client=Client(),
            registry=create_tool_registry(
                {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}}
            ),
            execution_mode="interactive",
            invocation_id="synthetic-timeline",
        )
        result_holder.update(result)
        return result

    stages.append(measured("consult-strategy-simulation-validation", propose))
    state = result_holder["state"]
    state.setdefault("confirmed_config", {}).setdefault("storage", {})["workspace_root"] = str(
        project
    )
    state["workspace_layout"] = {"workspace_root": str(project)}
    stages.append(measured("persist", lambda: write_json(state_path, state)))
    report_result = {**result_holder, "state": state}
    stages.append(
        measured("report", lambda: {"path": save_codex_report(state_path, report_result)})
    )
    stages.append(
        measured("return", lambda: {k: v for k, v in result_holder.items() if k != "state"})
    )
    assert not executed and result_holder["status"] == "awaiting_approval"
    return {
        "retention": retention,
        "timeline": stages,
        "model_stages": models,
        "status": result_holder["status"],
        "scientific_tool_executed": False,
    }


def history_probes(root):
    from phase_agent.agents.strategic.codex_team import CodexTeam
    from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde
    from phase_agent.persistence.compact_checkpoints import CompressedSerde
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    session = dict(
        team_id="team-" + "a" * 32,
        revision=1,
        consult=[],
        question="q",
        advice=[],
        guidance=[],
        events=[{"kind": "draft", "rationale": "x" * 4096}],
        binding="b",
        draft={"action": {"tool": "pause_search"}, "rationale": "r", "memory_ids": []},
    )
    action = CodexTeam._action(session)
    rows = [{"task_id": str(i), "payload": "x" * 256} for i in range(128)]
    results = []
    for histories in [0, 10, 100]:
        state = {
            "tasks": rows,
            "action_records": [
                {"record_id": str(i), "final_action": action} for i in range(histories)
            ],
        }
        frame = {
            "loaded_state": state,
            "result": {"state": state},
            "snapshot": {"config_version": "c1", "config": {}},
        }
        with closing(sqlite3.connect(":memory:")) as db:
            serde = CheckpointBlobSerde(
                CompressedSerde(JsonPlusSerializer()), db, threading.RLock()
            )
            serde._table()

            def encode():
                kind, wire = serde.dumps_typed(frame)
                db.commit()
                return {
                    "wire_bytes": len(wire),
                    "raw_json_bytes": size(frame),
                    "blob_bytes": db.execute(
                        "select coalesce(sum(length(data)),0) from checkpoint_payloads"
                    ).fetchone()[0],
                    "stored_blobs": db.execute(
                        "select count(*) from checkpoint_payloads"
                    ).fetchone()[0],
                }

            summary = encode()
            results.append({**measured(f"history-{histories}", encode), **summary})
    state = {"tasks": [{"task_id": str(i), "payload": "x" * 256} for i in range(4096)]}
    frame = {
        "loaded_state": state,
        "result": {"state": state},
        "snapshot": {"config_version": "c1", "config": {}},
    }
    with closing(sqlite3.connect(root / "checkpoint-count.sqlite")) as db:
        db.execute(
            "create table synthetic_checkpoints(sequence integer primary key, kind text, wire blob)"
        )
        serde = CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), db, threading.RLock())
        for i in range(20):
            kind, wire = serde.dumps_typed(frame)
            db.execute("insert into synthetic_checkpoints values (?,?,?)", (i, kind, wire))
            db.commit()
            if i + 1 in [1, 5, 20]:
                results.append(
                    {
                        "case": f"checkpoint-count-{i + 1}",
                        "wire_bytes": len(wire),
                        "stored_checkpoints": i + 1,
                        "stored_blobs": db.execute(
                            "select count(*) from checkpoint_payloads"
                        ).fetchone()[0],
                        "blob_bytes": db.execute(
                            "select sum(length(data)) from checkpoint_payloads"
                        ).fetchone()[0],
                        "database_bytes": (root / "checkpoint-count.sqlite").stat().st_size,
                    }
                )
    return results


def audit_flow(root):
    from tests.test_codex_native_team import make_control, team, advice
    from tests.test_simulation_agent import SimulationModel

    result = []
    for revisions in [1, 5, 10]:
        project = root / str(revisions)
        project.mkdir()
        control, context = make_control(project)
        context["allowed_tools"] = ["pause_search", "prepare_local_batch_files"]
        t = team(control)
        v = t.start(question="synthetic audit", consult=["exploration"])
        for revision in range(1, revisions + 1):
            opinion = advice(v)
            opinion.update(
                recommendation="insufficient_evidence",
                evidence_gaps=["Explain coverage uncertainty"],
            )
            t.advise(v["team_id"], revision, opinion, compact=True)
            t.clarify(v["team_id"], revision, "exploration", 0)
            t.advise(v["team_id"], revision, advice(v), compact=True)
            action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_inputs"}}
            v = t.draft(v["team_id"], revision, action, rationale="synthetic")
            t.simulate(v["team_id"], revision, SimulationModel()(v["requests"]["simulation"]))
            v = t.draft(
                v["team_id"], revision, action, rationale="clarify explanation", replace_draft=True
            )
            v = t.simulate(v["team_id"], revision, SimulationModel()(v["requests"]["simulation"]))
            if revision < revisions:
                v = t.feedback(
                    v["team_id"], revision, "revise", "synthetic revision " + str(revision)
                )
        v = t.feedback(v["team_id"], revisions, "accept", "synthetic direction only")
        session = t._load()
        frozen = t._action(session)
        assert control.state_path.read_text() == "{}"
        result.append(
            {
                "revisions": revisions,
                "events": len(session["events"]),
                "audit_file_bytes": t._path(v["team_id"]).stat().st_size,
                "frozen_action_bytes": size(frozen),
                "current_advice": len(session["advice"]),
                "simulation_plans": sum(e["kind"] == "simulation_plan" for e in session["events"]),
                "execution_approved": v["execution_approved"],
                "scientific_state_bytes": 2,
            }
        )
    return result


if __name__ == "__main__":
    with TemporaryDirectory(prefix="phase-hotspots-") as directory:
        import sys

        print(
            json.dumps(
                retention_and_timeline(Path(directory))
                if "--timeline" in sys.argv
                else audit_flow(Path(directory))
                if "--audit" in sys.argv
                else history_probes(Path(directory))
                if "--history" in sys.argv
                else run(Path(directory))
            )
        )

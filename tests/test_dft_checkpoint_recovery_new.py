import hashlib
import sqlite3
import pytest
from types import SimpleNamespace
from phase_agent.tools.workflows.saved_dft_input_frame import (
    prepare_saved_dft_input_frame,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.agents.strategic.codex_stale_submission import (
    stale_submission_checkpoint,
)
from phase_agent.tools.step_runner.file_protocol import write_json


@pytest.mark.parametrize(
    "block", [None, "state", "claim", "results", "wrong_request", "wrong_checkpoint"]
)
def test_dft_recovery_preserves_observation_and_rejects_unproven_effect(
    tmp_path, block
):
    state = {
        "execution_recovery_report": {"status": "clear", "unsettled": []},
        "tasks": [],
    }
    path = tmp_path / "state.json"
    write_json(path, {**state, **({"new_results": [1]} if block == "state" else {})})
    initial = {
        "request_id": "req",
        "phase": "initialized",
        "current_node": "initialize_confirmed_run",
        "business_frame": {"loaded_state": state},
    }
    observed = {
        "request_id": "req",
        "phase": "wait_checked",
        "current_node": "results_wait_gate",
        "node_status": "completed",
        "branch:to:analyze": None,
        "business_frame": {
            "combined_results": [1] if block == "results" else [],
            "feedback": {},
        },
    }
    serde = JsonPlusSerializer()
    t1, b1 = serde.dumps_typed({"channel_values": initial})
    t2, b2 = serde.dumps_typed({"channel_values": observed})
    with sqlite3.connect(tmp_path / "langgraph_lifecycle.sqlite") as c:
        c.execute(
            "create table checkpoints(thread_id text,checkpoint_ns text,checkpoint_id text,parent_checkpoint_id text,type text,checkpoint blob)"
        )
        c.execute(
            "insert into checkpoints values(?,?,?,?,?,?)", ("t", "", "a", None, t1, b1)
        )
        c.execute(
            "insert into checkpoints values(?,?,?,?,?,?)", ("t", "", "b", "a", t2, b2)
        )
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as c:
        c.execute(
            "create table execution_events(action_id text,event text,payload text,created_at text)"
        )
        c.execute(
            "create table receipts(action_id text,identity text,phase text,outcome text)"
        )
        if block == "claim":
            c.execute(
                "insert into execution_events values(?,?,?,?)",
                ("req", "claimed", "{}", "now"),
            )
    session = {
        "status": "submitting",
        "draft": {"action": {"tool": "select_dft_candidates"}},
    }
    kwargs = {
        "checkpoint_id": "wrong" if block == "wrong_checkpoint" else "b",
        "request_id": "other" if block == "wrong_request" else "req",
        "checkpoint_sha256": hashlib.sha256(b2).hexdigest(),
    }
    if block:
        with pytest.raises(ValueError):
            stale_submission_checkpoint(path, session, **kwargs)
    else:
        result = stale_submission_checkpoint(path, session, **kwargs)
        assert result["resume_from_wait_checked"] and result["executed"] is False


@pytest.mark.parametrize(
    "block",
    [
        None,
        "model",
        "config",
        "candidate",
        "receipt",
        "new_result",
        "active",
        "return",
        "pending",
        "unprocessed",
        "team",
    ],
)
def test_frozen_dft_draft_skips_only_unchanged_idle_evidence(
    tmp_path, monkeypatch, block
):
    action = {
        "tool": "select_dft_candidates",
        "_decision_team": {"team_id": "t"},
        "scientific_decision": {"model_version": "m", "config_version": "c"},
        "parameters": {
            "decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]
        },
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "confirmed_config": {},
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m"}},
        "qbc_candidates": [{"candidate_id": "s"}],
        "tasks": [],
        "pending_execution_policies": {},
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "runtime_adapters": {},
        "state_path": str(tmp_path / "state.json"),
        "effective_config": {},
        "snapshot": {"config_version": "c", "config": {}},
        "agent_client": SimpleNamespace(
            _codex_structured=True, _frozen_dft_action=action
        ),
    }
    if block == "model":
        state["active_model_version"] = "other"
    if block == "config":
        frame["snapshot"]["config"] = {"changed": True}
    if block == "candidate":
        state["qbc_candidates"] = []
    if block == "new_result":
        frame["recovered_results"] = [{"task_id": "new"}]
    if block == "active":
        state["tasks"] = [{"task_id": "busy", "status": "pending"}]
    if block == "pending":
        state["pending_execution_policies"] = {"other": {}}
    if block == "unprocessed":
        state["processed_task_ids"] = ["unknown"]
    if block == "team":
        action.pop("_decision_team")
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return",
        lambda *a, **k: block == "return",
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report",
        lambda *a: {"status": "clear", "unsettled": ["x"] if block == "receipt" else []},
    )
    assert bool(prepare_saved_dft_input_frame(frame).get("saved_input_only")) == (
        block is None
    )


def test_verified_resume_accepts_only_proven_next_node():
    from phase_agent.graphs.project.runner import _validate_verified_resume

    blob = b"checkpoint"
    snapshot = SimpleNamespace(
        config={"configurable": {"checkpoint_id": "cp"}},
        values={"request_id": "req"},
        next=("analyze",),
    )
    evidence = {
        "checkpoint_id": "cp",
        "request_id": "req",
        "resume_next": "analyze",
        "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
    }
    _validate_verified_resume(snapshot, "req", evidence, (blob,))
    evidence["resume_next"] = "react"
    with pytest.raises(ValueError, match="verified_resume_checkpoint_changed"):
        _validate_verified_resume(snapshot, "req", evidence, (blob,))

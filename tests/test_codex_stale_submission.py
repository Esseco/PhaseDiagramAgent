import hashlib
import json
import sqlite3

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from phase_agent.agents.strategic.codex_stale_submission import (
    _proven_superseded_refresh, stale_submission_checkpoint,
)
from phase_agent.tools.step_runner.file_protocol import write_json


def fixture(tmp_path, change=None):
    path = tmp_path / "state.json"
    state = {"execution_recovery_report": {"status": "clear", "unsettled": []}, "tasks": []}
    write_json(path, state)
    values = {"request_id": "old", "phase": "initialized", "current_node": "initialize_confirmed_run",
              "node_status": "completed", "business_frame": {"automatic_results": [], "loaded_state": state}}
    if change:
        change(values)
    kind, blob = JsonPlusSerializer().dumps_typed({"channel_values": values})
    with sqlite3.connect(tmp_path / "langgraph_lifecycle.sqlite") as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, type TEXT, checkpoint BLOB)")
        conn.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)", ("thread", "", "cp", kind, blob))
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (action_id TEXT, identity TEXT, phase TEXT, outcome TEXT)")
        conn.execute("CREATE TABLE execution_events (action_id TEXT, event TEXT, payload TEXT, created_at TEXT)")
    session = {"submission": {"status": "execution_reconciliation_required"}, "draft": {"action": {
        "tool": "prepare_local_batch_files", "parameters": {"mode": "model_refresh_inputs"}, "task_key": "refresh"
    }}}
    proof = {"checkpoint_id": "cp", "request_id": "old", "checkpoint_sha256": hashlib.sha256(blob).hexdigest()}
    return path, session, proof


def test_old_initialized_checkpoint_and_no_refresh_effect_are_not_replayed(tmp_path):
    path, session, proof = fixture(tmp_path)
    before = path.read_bytes()
    assert stale_submission_checkpoint(path, session, **proof)["executed"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [
    lambda v: v.update(phase="finalized"),
    lambda v: v.update(completed_response={"status": "prepared"}),
    lambda v: v["business_frame"].update(automatic_results=[{"status": "completed"}]),
])
def test_unproven_old_checkpoint_fails_closed(tmp_path, change):
    path, session, proof = fixture(tmp_path, change)
    with pytest.raises(ValueError):
        stale_submission_checkpoint(path, session, **proof)


def test_refresh_claim_or_task_fails_closed(tmp_path):
    path, session, proof = fixture(tmp_path)
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("INSERT INTO execution_events VALUES (?, ?, ?, ?)",
                     ("id", "claimed", json.dumps({"action": {"task_key": "refresh"}}), "now"))
    with pytest.raises(ValueError, match="execution_receipt"):
        stale_submission_checkpoint(path, session, **proof)


def test_submitting_is_reconciled_only_after_checkpoint_proof(tmp_path):
    from phase_agent.agents.strategic.codex_team import CodexTeam
    from tests.test_codex_native_team import make_control

    control, _ = make_control(tmp_path)
    path, session, proof = fixture(tmp_path)
    team = CodexTeam(control)
    view = team.start(question="Recover interrupted refresh", consult=[])
    saved = team._load()
    saved.update(status="submitting", submission=session["submission"])
    saved["draft"] = {**session["draft"], "rationale": "Recover", "memory_ids": []}
    team._save(saved)
    before = path.read_bytes()
    result = team.reconcile(view["team_id"], 1, **proof, evidence_note="Verify no execution")
    assert result["status"] == "ready_to_propose"
    assert result["events"][-1]["kind"] == "verified_stale_lifecycle_block"
    assert path.read_bytes() == before


def test_submitting_with_execution_claim_remains_blocked(tmp_path):
    path, session, proof = fixture(tmp_path)
    session["status"] = "submitting"
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("INSERT INTO execution_events VALUES (?, ?, ?, ?)",
                     ("id", "claimed", json.dumps({"action": {"task_key": "refresh"}}), "now"))
    with pytest.raises(ValueError, match="execution_receipt"):
        stale_submission_checkpoint(path, session, **proof)


def test_unpersisted_checkpoint_state_blocks_reconciliation(tmp_path):
    path, session, proof = fixture(tmp_path, lambda v: v["business_frame"]["loaded_state"].update(
        new_dft_records=[{"id": "not-yet-persisted"}]))
    with pytest.raises(ValueError, match="not_proven_persisted"):
        stale_submission_checkpoint(path, session, **proof)


def assessed(values):
    state = values["business_frame"]["loaded_state"]
    values.update(phase="assessed", current_node="assess_and_export_round", **{"branch:to:react": None})
    values["business_frame"].update(pre_reconciled={"state": state},
                                    feedback={"state": state, "recorded": [], "rejected": []},
                                    combined_results=[], training_returns=[], training_handoffs=[])


def test_assessed_state_can_only_be_recovered_when_analysis_was_saved(tmp_path):
    path, session, proof = fixture(tmp_path, assessed)
    assert stale_submission_checkpoint(path, session, **proof)["executed"] is False


def test_unsaved_analysis_blocks_assessed_recovery(tmp_path):
    def changed(values):
        assessed(values)
        values["business_frame"]["feedback"]["state"] = {"unsaved": "scientific result"}

    path, session, proof = fixture(tmp_path, changed)
    with pytest.raises(ValueError, match="analysis_not_proven_persisted"):
        stale_submission_checkpoint(path, session, **proof)


def superseded_case():
    from copy import deepcopy

    record = {"data_id": "d1", "energy": -1}
    old = {"active_model_version": "m1", "active_model": {"version": "m1"},
           "model_refresh": {"status": "completed", "new_model_version": "m1", "task_ids": ["t1"]},
           "tasks": [{"task_id": "t1", "status": "completed"}],
           "dft_training_records": [record], "new_dft_records": [record],
           "model_registry": {"m1": {"status": "active", "sha256": "abc"}},
           "phase_diagrams": {"mlip": {"version": "v1", "entries": [1]}},
           "model_activation_history": [], "model_transitions": [], "model_update_epochs": [],
           "training_handoff_history": []}
    current = deepcopy(old)
    current.update(active_model_version="m2", active_model={"version": "m2"}, new_dft_records=[],
                   model_refresh={"old_model_version": "m1", "new_model_version": "m2"})
    current["model_registry"]["m1"]["status"] = "historical_active"
    current["model_activation_history"].append({"old_model_version": "m1", "new_model_version": "m2",
                                                 "approval_reason": "approved"})
    return old, current


def test_superseded_snapshot_keeps_training_and_finished_tasks():
    old, current = superseded_case()
    assert _proven_superseded_refresh(old, current)


@pytest.mark.parametrize("change", [
    lambda s: s["dft_training_records"].clear(),
    lambda s: s["tasks"][0].update(status="running"),
    lambda s: s["model_activation_history"][0].update(approval_reason=""),
    lambda s: s["model_registry"]["m1"].update(sha256="different"),
    lambda s: s["phase_diagrams"]["mlip"]["entries"].clear(),
])
def test_superseded_snapshot_rejects_missing_or_changed_evidence(change):
    old, current = superseded_case()
    change(current)
    assert not _proven_superseded_refresh(old, current)


def test_reconciled_pre_dispatch_parent_first_clears_when_unsettled_empty(tmp_path):
    path, session, proof = fixture(tmp_path)
    session["draft"]["action"] = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_parent_first_recovery",
        "parameters": {"mode": "parent_first_recovery_inputs"},
    }
    res = stale_submission_checkpoint(path, session, **proof)
    assert res["executed"] is False
    assert res["classification"] == "pre_dispatch_execution_reconciliation_cleared"

    import os
    conn = sqlite3.connect(tmp_path / "execution_receipts.sqlite")
    try:
        with conn:
            conn.execute(
                "INSERT INTO receipts VALUES (?, ?, ?, ?)",
                (
                    json.dumps([os.path.normcase(str(path.resolve())), "i1"]),
                    json.dumps({"tool": "prepare_local_batch_files", "invocation_id": "i1"}),
                    "claimed",
                    json.dumps({}),
                ),
            )
    finally:
        conn.close()
    with pytest.raises(ValueError, match="execution_reconciliation_required"):
        stale_submission_checkpoint(path, session, **proof)

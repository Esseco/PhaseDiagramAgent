import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.agents.strategic.codex_stale_submission import blocked_energy_submission_checkpoint
from phase_agent.tools.state.execution_reconciliation import propose_reconciliation, apply_reconciliation
from phase_agent.tools.step_runner.file_protocol import write_json
from tests.test_abandoned_region_generation import setup_case


def fixture(tmp_path):
    path, old, identity, directory, _ = setup_case(tmp_path)
    old.update(execution_recovery_report={"status": "reconciliation_required", "unsettled": [{"identity": {"invocation_id": "old"}}]},
               execution_reconciliations={}, recovery_review_history=[], pending_execution_recoveries={}, budget_usage={})
    old_snapshot = deepcopy(old)
    att = {"evidence_note": "用户选择弃用暂存，保留文件", "external_jobs": "not_applicable", "output_inventory": "abandoned_local_staging", "external_cost": 0}
    plan = propose_reconciliation(path, old, "old", "verified_abandoned_staging", att)
    current = apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive", reviewer_comment="confirmed")["state"]
    write_json(path, current)
    frame = {"automatic_results": [], "combined_results": [], "training_returns": [],
             "recovered_count": 0, "feedback": {"state": old_snapshot, "recorded": [], "rejected": []}}
    result = {"status": "execution_reconciliation_required", "state": old_snapshot, "steps_executed": 0,
              "submitted": False, "recovery_report": {"unsettled": [{"identity": identity}]}}
    values = {"request_id": "blocked", "phase": "wait_checked", "current_node": "results_wait_gate",
              "node_status": "waiting", "response": {"status": "execution_reconciliation_required"},
              "business_frame": frame, "completed_response": result}
    kind, blob = JsonPlusSerializer().dumps_typed({"channel_values": values})
    with sqlite3.connect(path.parent / "langgraph_lifecycle.sqlite") as c:
        c.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, type TEXT, checkpoint BLOB)")
        c.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)", ("thread", "", "cp", kind, blob))
    session = {"submission": {"status": "execution_reconciliation_required"}, "draft": {"action": {
        "tool": "prepare_local_batch_files", "parameters": {"mode": "mc_energy_repair_inputs"}, "task_key": "energy"}}}
    proof = {"checkpoint_id": "cp", "request_id": "blocked", "checkpoint_sha256": hashlib.sha256(blob).hexdigest()}
    return path, old_snapshot, current, session, proof, directory


def test_blocked_energy_proof_preserves_checkpoint_and_files(tmp_path):
    path, _, current, session, proof, directory = fixture(tmp_path)
    before = (path.parent / "langgraph_lifecycle.sqlite").read_bytes()
    assert blocked_energy_submission_checkpoint(path, session, **proof)["executed"] is False
    assert (path.parent / "langgraph_lifecycle.sqlite").read_bytes() == before
    assert len(list(directory.rglob("*.vasp"))) == 2
    assert current["execution_recovery_report"]["status"] == "reconciliation_required"


@pytest.mark.parametrize("change", ["scientific_state", "staging_file", "execution_claim", "task", "review_hash"])
def test_blocked_energy_proof_fails_closed(tmp_path, change):
    path, old, current, session, proof, directory = fixture(tmp_path)
    if change == "scientific_state":
        current["phase_records"].append({"record_id": "unrelated"}); write_json(path, current)
    elif change == "staging_file":
        next(directory.rglob("S-*.vasp")).write_text("tampered", encoding="utf-8")
    elif change == "execution_claim":
        with sqlite3.connect(path.parent / "execution_receipts.sqlite") as c:
            c.execute("INSERT INTO execution_events VALUES (?, ?, ?, ?)", ("fake", "claimed", json.dumps({"action": {"task_key": "energy"}}), "now"))
    elif change == "task":
        current["tasks"].append({"task_key": "energy", "status": "completed"}); write_json(path, current)
    else:
        proof["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError):
        blocked_energy_submission_checkpoint(path, session, **proof)

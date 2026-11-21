import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from tests.test_codex_native_team import make_control


def fixture(tmp_path):
    control, _ = make_control(tmp_path)
    state = {"tasks": [{"task_id": "old", "stage": "dft_single_point", "status": "pending", "batch_id": "b", "input_path": "x"}], "pending_execution_policies": {}}
    control.state_path.write_text(json.dumps(state), encoding="utf8")
    values = {"request_id": "req", "phase": "wait_checked", "current_node": "results_wait_gate", "node_status": "waiting", "business_frame": {"runtime_state_path": str(control.state_path)}, "completed_response": {"status": "awaiting_manual_submission", "steps_executed": 0, "submitted": False, "manual_wait": {"task_ids": ["old"], "waiting_task_count": 1, "waiting_by_stage": {"dft_single_point": 1}}}}
    kind, data = JsonPlusSerializer().dumps_typed({"channel_values": values})
    with sqlite3.connect(tmp_path / "langgraph_lifecycle.sqlite") as c:
        c.execute("CREATE TABLE checkpoints (thread_id TEXT,checkpoint_ns TEXT,checkpoint_id TEXT,type TEXT,checkpoint BLOB)")
        c.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", ("t", "", "cp", kind, data))
    session = {"submission": {"status": "awaiting_manual_submission"}, "draft": {"action": {"tool": "select_dft_candidates", "operation": "select_dft_candidates"}}}
    return control, session, values, {"checkpoint_id": "cp", "request_id": "req", "checkpoint_sha256": hashlib.sha256(data).hexdigest()}


def save_checkpoint(path, values):
    kind, data = JsonPlusSerializer().dumps_typed({"channel_values": values})
    with sqlite3.connect(path / "langgraph_lifecycle.sqlite") as c:
        c.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    return hashlib.sha256(data).hexdigest()


def test_bound_wait_checkpoint_proves_no_execution(tmp_path):
    from phase_agent.agents.strategic.codex_wait_submission import waiting_checkpoint
    control, session, _, proof = fixture(tmp_path)
    before = control.state_path.read_bytes()
    assert waiting_checkpoint(control.state_path, session, **proof)["executed"] is False
    assert control.state_path.read_bytes() == before


@pytest.mark.parametrize("case", ["wrong_request", "wrong_project", "wrong_node", "executed", "submitted", "changed_wait", "pending_plan", "receipt", "wrong_action"])
def test_wait_proof_rejects_uncertain_or_changed_evidence(tmp_path, case):
    from phase_agent.agents.strategic.codex_wait_submission import waiting_checkpoint
    control, session, values, proof = fixture(tmp_path)
    if case == "wrong_request": values["request_id"] = "other"
    elif case == "wrong_project": values["business_frame"]["runtime_state_path"] = "other"
    elif case == "wrong_node": values["current_node"] = "other"
    elif case == "executed": values["completed_response"]["steps_executed"] = 1
    elif case == "submitted": values["completed_response"]["submitted"] = True
    elif case == "changed_wait": values["completed_response"]["manual_wait"]["task_ids"] = ["other"]
    elif case == "pending_plan":
        state = json.loads(control.state_path.read_text())
        state["pending_execution_policies"] = {"x": {}}
        control.state_path.write_text(json.dumps(state))
    elif case == "receipt":
        with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as c:
            c.execute("CREATE TABLE receipts (identity TEXT)")
            c.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req"}),))
    else: session["draft"]["action"]["tool"] = "generate_branches"
    proof["checkpoint_sha256"] = save_checkpoint(tmp_path, values)
    with pytest.raises(ValueError): waiting_checkpoint(control.state_path, session, **proof)


def test_team_reconcile_preserves_frozen_waiting_action(tmp_path):
    from phase_agent.agents.strategic.codex_team import CodexTeam
    control, blocked, _, proof = fixture(tmp_path)
    t = CodexTeam(control)
    started = t.start(question="review the same frozen inputs", consult=[])
    session = t._load()
    session.update(status="submission_uncertain", submission=blocked["submission"], draft={**blocked["draft"], "rationale": "preserve scope", "memory_ids": []})
    t._save(session)
    before = control.state_path.read_bytes()
    result = t.reconcile(started["team_id"], 1, **proof, evidence_note="Bound waiting checkpoint has zero executed steps")
    assert result["status"] == "ready_to_propose"
    assert result["draft"]["action"] == blocked["draft"]["action"]
    assert control.state_path.read_bytes() == before


@pytest.mark.parametrize("case", ["valid", "not_input_only", "no_simulation", "executed", "pending_plan"])
def test_training_wait_proof_requires_frozen_inputs_and_zero_execution(tmp_path, case):
    from phase_agent.agents.strategic.codex_wait_submission import waiting_checkpoint
    control, session, values, proof = fixture(tmp_path)
    session["draft"]["action"] = {"tool": "update_mlip", "operation": "prepare_mlip_training", "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True}, "_simulation_plan": {"steps": [1]}}
    if case == "not_input_only":
        session["draft"]["action"]["parameters"]["prepare_inputs_only"] = False
    elif case == "no_simulation":
        session["draft"]["action"].pop("_simulation_plan")
    elif case == "executed":
        values["completed_response"]["steps_executed"] = 1
        proof["checkpoint_sha256"] = save_checkpoint(tmp_path, values)
    elif case == "pending_plan":
        state = json.loads(control.state_path.read_text())
        state["pending_execution_policies"] = {"x": {}}
        control.state_path.write_text(json.dumps(state))
    if case == "valid":
        assert waiting_checkpoint(control.state_path, session, **proof)["executed"] is False
    else:
        with pytest.raises(ValueError):
            waiting_checkpoint(control.state_path, session, **proof)

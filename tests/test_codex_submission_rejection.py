import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.agents.strategic.codex_submission_rejection import rejection_checkpoint
from phase_agent.agents.strategic.codex_team import CodexTeam
from tests.test_codex_native_team import make_control


def fixture(tmp_path, change=None):
    control, _ = make_control(tmp_path)
    reason = "llm_failed: ValueError: Invalid budget trade-off: round_budget_review: compare supplement_dft and new_search before selecting an action"
    response = {
        "status": "not_configured",
        "submitted": False,
        "reason": reason,
        "steps_executed": 1,
        "events": [{"status": "not_configured", "reason": reason}],
    }
    values = {
        "request_id": "req1",
        "phase": "finalized",
        "current_node": "prepare_inputs_and_finalize",
        "node_status": "completed",
        "business_frame": {"runtime_state_path": str(control.state_path)},
        "completed_response": response,
    }
    if change:
        change(values)
    kind, data = JsonPlusSerializer().dumps_typed({"channel_values": values})
    with sqlite3.connect(tmp_path / "langgraph_lifecycle.sqlite") as conn:
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT,checkpoint_ns TEXT,checkpoint_id TEXT,type TEXT,checkpoint BLOB)"
        )
        conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", ("t1", "", "cp1", kind, data))
    session = {
        "submission": {"status": "not_configured", "reason": reason},
        "draft": {
            "action": {
                "tool": "update_mlip",
                "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
            }
        },
    }
    proof = {
        "checkpoint_id": "cp1",
        "request_id": "req1",
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
    }
    return control, session, proof


def test_completed_proposal_validation_failure_is_not_a_scientific_execution(tmp_path):
    c, s, p = fixture(tmp_path)
    before = c.state_path.read_bytes()
    assert rejection_checkpoint(c.state_path, s, **p)["executed"] is False
    t = CodexTeam(c)
    view = t.start(question="repair metadata", consult=[])
    persisted = t._load()
    s["draft"].update(rationale="repair metadata", memory_ids=[])
    persisted.update(s, status="submission_uncertain")
    t._save(persisted)
    result = t.reconcile(
        view["team_id"],
        1,
        **p,
        evidence_note="Checked original checkpoint; failed before proposal existed",
    )
    assert result["status"] == "validation_rejected"
    assert c.state_path.read_bytes() == before
    assert t.feedback(view["team_id"], 1, "revise", "correct metadata")["revision"] == 2


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(phase="running"),
        lambda v: v.update(request_id="different"),
        lambda v: v["completed_response"].update(submitted=True),
        lambda v: v["completed_response"].update(execution={"status": "completed"}),
        lambda v: v["completed_response"]["events"].append({"status": "prepared"}),
        lambda v: v["business_frame"].update(runtime_state_path="other-state.json"),
    ],
)
def test_nonzero_or_unbound_checkpoint_is_rejected(tmp_path, change):
    c, s, p = fixture(tmp_path, change)
    with pytest.raises(ValueError):
        rejection_checkpoint(c.state_path, s, **p)


def test_changed_hash_and_any_execution_receipt_fail_closed(tmp_path):
    c, s, p = fixture(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        rejection_checkpoint(c.state_path, s, **{**p, "checkpoint_sha256": "0" * 64})
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, s, **p)


def test_mlip_budget_review_rejected_before_dispatch(tmp_path):
    c, context = make_control(tmp_path)
    context["allowed_tools"] = ["update_mlip"]
    context["decision_context"]["round_budget_evidence"] = {"required": True}
    called = []
    c.chat_handler.workflow_kwargs = {}
    c.chat_handler._run = lambda *a, **k: called.append("run")
    with pytest.raises(ValueError, match="round_budget_review"):
        c._propose_codex(
            {
                "tool": "update_mlip",
                "operation": "prepare_mlip_training",
                "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
            },
            context_version="v1",
            _on_dispatch=lambda: called.append("dispatch"),
        )
    assert called == []


def test_scientific_contract_failure_checkpoint_can_be_proven(tmp_path):
    from phase_agent.agents.strategic.scientific_decisions import scientific_decision_errors

    c, session, proof = fixture(tmp_path)
    error = scientific_decision_errors(session["draft"]["action"], {}, {})[0]
    reason = "llm_failed: ValueError: Invalid scientific decision: " + error
    session["submission"]["reason"] = reason
    path = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(path) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        serde = JsonPlusSerializer()
        values = serde.loads_typed((kind, data))
        response = values["channel_values"]["completed_response"]
        response["reason"] = reason
        response["events"][0]["reason"] = reason
        kind, data = serde.dumps_typed(values)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["executed"] is False


def test_all_exposed_scientific_contracts_checked_before_dispatch(tmp_path):
    c, context = make_control(tmp_path)
    context["allowed_tools"] = ["update_mlip"]
    context["required_contracts"] = {"scientific_decision": "required"}
    c.chat_handler.workflow_kwargs = {}
    calls = []
    c.chat_handler._run = lambda *a, **k: calls.append("run")
    with pytest.raises(ValueError, match="scientific_decision"):
        c._propose_codex(
            {
                "tool": "update_mlip",
                "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
            },
            context_version="v1",
            _on_dispatch=lambda: calls.append("dispatch"),
        )
    assert calls == []


def test_exact_branch_reference_rejection_can_be_proven_without_execution(tmp_path):
    c, session, proof = fixture(tmp_path)
    ref = "branch_pool:abcd"
    reason = (
        "llm_failed: ValueError: branch_review_invalid_after_one_correction: "
        "branch_pool_review.evidence_refs: unknown source ['branch_pool:abcd']; "
        "branch_pool_review.group[0].evidence_refs: valid evidence refs required"
    )
    session["submission"]["reason"] = reason
    session["draft"]["action"] = {
        "tool": "generate_branches",
        "branch_pool_review": {"evidence_refs": [ref]},
    }
    path = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(path) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        serde = JsonPlusSerializer()
        values = serde.loads_typed((kind, data))
        response = values["channel_values"]["completed_response"]
        response["reason"] = reason
        response["events"][0]["reason"] = reason
        kind, data = serde.dumps_typed(values)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["executed"] is False
    session["draft"]["action"]["branch_pool_review"]["evidence_refs"] = ["branch_pool:other"]
    with pytest.raises(ValueError, match="no_proven"):
        rejection_checkpoint(c.state_path, session, **proof)

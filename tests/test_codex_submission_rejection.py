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


@pytest.mark.parametrize("ref", ["branch_pool:abcd", "phase_diagram:mlip_active_model:pure", "phase_diagram:mlip:current"])
def test_exact_branch_reference_rejection_can_be_proven_without_execution(tmp_path, ref):
    c, session, proof = fixture(tmp_path)
    reason = (
        "llm_failed: ValueError: branch_review_invalid_after_one_correction: "
        f"branch_pool_review.evidence_refs: unknown source ['{ref}']; "
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


def test_simulation_index_rejection_requires_proof_and_no_execution(tmp_path):
    from phase_agent.agents.simulation.agent import action_hash

    control, session, proof = fixture(tmp_path)
    reason = "Simulation planning: simulation_intent_changed; request new planning and approval"
    intent = {
        "tool": "generate_branches",
        "parameters": {},
        "branch_pool_review": {"existing_branch_groups": [{"branch_indices": [0]}]},
    }
    plan = {
        "action_hash": action_hash(intent),
        "rationale": "planned",
        "steps": [
            {"kind": "preflight", "description": "check", "tool": None},
            {"kind": "invoke", "description": "run", "tool": "generate_branches"},
            {"kind": "verify", "description": "verify", "tool": None},
        ],
        "checks": ["check"],
        "failure_policy": "return_to_strategic",
    }
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {**intent, "_simulation_intent": intent, "_simulation_plan": plan}
    path = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(path) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        serde = JsonPlusSerializer()
        payload = serde.loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )
        kind, data = serde.dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(control.state_path, session, **proof)["executed"] is False
    session["draft"]["action"]["_simulation_plan"]["action_hash"] = "changed"
    with pytest.raises(ValueError, match="simulation_plan_stale_action"):
        rejection_checkpoint(control.state_path, session, **proof)


def test_dft_validation_rejection_requires_terminal_proof(tmp_path):
    c, s, p = fixture(tmp_path)
    reason = "DFT 身份或D/H审阅未通过：dft_DH_group_requires_selection_or_reasoned_deferral:D:group; independent_dft_audit_required"
    s["submission"] = {"status": "rejected", "reason": reason}
    s["draft"]["action"] = {"tool": "select_dft_candidates", "parameters": {"decisions": []}}
    db = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(db) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    p["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, s, **p)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, s, **p)


def test_direct_refresh_gate_rejection_requires_zero_execution_proof(tmp_path):
    c, s, p = fixture(tmp_path)
    reason = "刷新第1批：失败原因：ValueError: refresh_relaxation_not_converged"
    s["submission"] = {"status": "not_configured", "reason": reason}
    s["draft"]["action"] = {
        "operation": "sample_direct_extrapolation",
        "tool": "prepare_local_batch_files",
        "parameters": {"mode": "direct_extrapolation", "scope_id": "epoch_refresh"},
    }
    db = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(db) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(reason=reason, events=[{"status": "not_configured", "reason": reason}])
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    p["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, s, **p)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, s, **p)


def test_direct_refresh_policy_rejection_requires_zero_execution_proof(tmp_path):
    c, s, p = fixture(tmp_path)
    reason = "方案未通过执行约束：new_model_structure_refresh_required"
    s["submission"] = {"status": "rejected", "reason": reason}
    s["draft"]["action"] = {
        "operation": "sample_direct_extrapolation",
        "tool": "prepare_local_batch_files",
        "parameters": {"mode": "direct_extrapolation", "scope_id": "epoch_refresh"},
    }
    db = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(db) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    p["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, s, **p)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, s, **p)


def test_dft_simulation_rejection_still_requires_checkpoint_and_receipt_proof(tmp_path):
    from phase_agent.agents.simulation.agent import action_hash

    control, session, proof = fixture(tmp_path)
    reason = "Simulation planning: simulation_intent_changed; request new planning and approval"
    intent = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "S-1"}]},
        "budget": 14004.07,
    }
    plan = {
        "action_hash": action_hash(intent),
        "rationale": "planned",
        "steps": [
            {"kind": "preflight", "description": "check", "tool": None},
            {"kind": "invoke", "description": "run", "tool": "select_dft_candidates"},
            {"kind": "verify", "description": "verify", "tool": None},
        ],
        "checks": ["check"],
        "failure_policy": "return_to_strategic",
    }
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {**intent, "_simulation_intent": intent, "_simulation_plan": plan}
    db = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(db) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        payload["channel_values"]["completed_response"].update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(control.state_path, session, **proof)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(control.state_path, session, **proof)


def test_gs_selection_rejection_requires_exact_reason_and_zero_execution(tmp_path):
    c, session, proof = fixture(tmp_path)
    reason = "DFT 方案未通过筛选/预算校验，需修订后再批准：['ground_state_selection_required:s1']"
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {"tool": "select_dft_candidates", "parameters": {"decisions": []}}
    db = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(db) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["executed"] is False
    session["submission"]["reason"] = (
        "DFT 方案未通过筛选/预算校验，需修订后再批准：['ground_state_selection_required:s1', 'other_failure']"
    )
    with pytest.raises(ValueError, match="no_proven"):
        rejection_checkpoint(c.state_path, session, **proof)
    session["submission"]["reason"] = reason
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, session, **proof)


def test_mc_stage_limit_rejection_requires_terminal_checkpoint_and_no_receipt(tmp_path):
    reason = "方案未通过执行约束：stage_task_limit"

    def change(values):
        values["completed_response"].update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )

    c, session, proof = fixture(tmp_path, change)
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {"tool": "allocate_mc_bohb"}
    assert rejection_checkpoint(c.state_path, session, **proof)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, session, **proof)


def test_mc_batch_single_target_guard_proves_zero_execution(tmp_path):
    reason = "模型未能返回完整有效方案，未暂停搜索或执行任务。请继续重试。原因：llm_failed: ValueError: restart_failed_task requires a retryable scientific task_id"

    def change(values):
        values["completed_response"]["reason"] = reason
        values["completed_response"]["events"][0]["reason"] = reason

    control, session, proof = fixture(tmp_path, change)
    session["submission"]["reason"] = reason
    session["draft"]["action"] = {
        "tool": "restart_failed_task",
        "parameters": {"repair_profile": "mc_threshold_recovery"},
        "target_ids": ["m0", "m1"],
    }
    assert rejection_checkpoint(control.state_path, session, **proof)["executed"] is False
    session["draft"]["action"]["parameters"]["repair_profile"] = "other"
    with pytest.raises(ValueError, match="no_proven"):
        rejection_checkpoint(control.state_path, session, **proof)


def test_mc_input_preparation_old_task_limit_rejection_proves_zero_execution(tmp_path):
    reason = "方案未通过执行约束：stage_task_limit"

    def change(values):
        response = values["completed_response"]
        response.update(status="rejected", reason=reason)
        response["events"] = [{"status": "rejected", "reason": reason}]

    control, session, proof = fixture(tmp_path, change)
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_mlip_mc",
        "parameters": {"mode": "mc_inputs", "expected_task_ids": ["a", "b"]},
    }
    assert rejection_checkpoint(control.state_path, session, **proof)["executed"] is False
    session["draft"]["action"]["parameters"]["expected_task_ids"] = ["a", "a"]
    with pytest.raises(ValueError, match="no_proven"):
        rejection_checkpoint(control.state_path, session, **proof)


def test_dft_legacy_stage_rejection_requires_exact_selection_and_no_receipt(tmp_path):
    c, session, proof = fixture(tmp_path)
    decisions = [{"candidate_id": "c1", "action": "DFT_SINGLE_POINT"},
                 {"candidate_id": "c2", "action": "DFT_SINGLE_POINT"}]
    failures = [{"candidate_id": row["candidate_id"], "action": row["action"],
                 "reason": "dft_budget", "details": ["stage_task_limit"]} for row in decisions]
    reason = "DFT 方案未通过筛选/预算校验，需修订后再批准：" + repr(failures)
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {"tool": "select_dft_candidates", "parameters": {"decisions": decisions}}
    path = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(path) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        serde = JsonPlusSerializer()
        values = serde.loads_typed((kind, data))
        response = values["channel_values"]["completed_response"]
        response.update(status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}])
        kind, data = serde.dumps_typed(values)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["classification"] == "dft_legacy_stage_limit_rejected"
    session["draft"]["action"]["parameters"]["decisions"].pop()
    with pytest.raises(ValueError, match="no_proven"):
        rejection_checkpoint(c.state_path, session, **proof)
    session["draft"]["action"]["parameters"]["decisions"] = decisions + [{"candidate_id": "c2", "action": "DFT_SINGLE_POINT"}]
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, session, **proof)


def test_relax_preview_rejection_requires_bound_plan_and_zero_execution(tmp_path):
    from phase_agent.agents.simulation.agent import action_hash

    control, session, proof = fixture(tmp_path)
    reason = "Simulation planning: simulation_intent_changed; request new planning and approval"
    intent = {"tool": "prepare_local_batch_files", "operation": "prepare_relax_mc_batch",
              "target_ids": ["B-1"], "parameters": {"mode": "relax_mc_inputs",
              "relax_mc_preview": {"checksum": "old"}}}
    plan = {"action_hash": action_hash(intent), "rationale": "planned",
            "steps": [{"kind": "preflight", "description": "check", "tool": None},
                      {"kind": "invoke", "description": "prepare", "tool": "prepare_local_batch_files"},
                      {"kind": "verify", "description": "verify", "tool": None}],
            "checks": ["check"], "failure_policy": "return_to_strategic"}
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {**intent, "_simulation_intent": intent, "_simulation_plan": plan}
    db = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(db) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        serde = JsonPlusSerializer()
        payload = serde.loads_typed((kind, data))
        payload["channel_values"]["completed_response"].update(
            status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}]
        )
        kind, data = serde.dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(control.state_path, session, **proof)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(control.state_path, session, **proof)


def test_frozen_dft_filtered_pool_rejection_requires_bound_ids_and_zero_receipts(tmp_path):
    from phase_agent.agents.simulation.agent import action_hash

    c, session, proof = fixture(tmp_path)
    intent = {"tool": "select_dft_candidates", "parameters": {"decisions": [
        {"candidate_id": "new-frame", "action": "DFT_SINGLE_POINT"},
        {"candidate_id": "completed-frame", "action": "DEFER", "reason": "already completed"},
    ]}}
    plan = {"action_hash": action_hash(intent), "rationale": "frozen selection", "steps": [
        {"kind": "preflight", "description": "check identity"},
        {"kind": "invoke", "tool": "select_dft_candidates", "description": "select"},
        {"kind": "verify", "description": "verify"},
    ], "checks": ["same frames"], "failure_policy": "return_to_strategic"}
    session["draft"]["action"] = {**intent, "_simulation_intent": intent, "_simulation_plan": plan}
    reason = ("DFT 身份或D/H审阅未通过：unknown_candidate:new-frame; unknown_candidate:completed-frame"
              "；DFT 方案未通过筛选/预算校验，需修订后再批准：['unknown_candidate:new-frame']")
    session["submission"] = {"status": "rejected", "reason": reason}
    path = tmp_path / "langgraph_lifecycle.sqlite"
    with sqlite3.connect(path) as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        serde = JsonPlusSerializer()
        values = serde.loads_typed((kind, data))
        response = values["channel_values"]["completed_response"]
        response.update(status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}])
        kind, data = serde.dumps_typed(values)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["executed"] is False
    bad = deepcopy(session)
    bad["submission"]["reason"] = reason.replace("unknown_candidate:completed-frame", "unknown_candidate:other")
    with pytest.raises(ValueError, match="no_proven"):
        rejection_checkpoint(c.state_path, bad, **proof)
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, session, **proof)


def test_independent_audit_rejection_requires_terminal_checkpoint_and_no_receipt(tmp_path):
    c, session, proof = fixture(tmp_path)
    reason = "DFT 方案未通过筛选/预算校验，需修订后再批准：['independent_dft_audit_required']"
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {"tool": "select_dft_candidates", "parameters": {"decisions": []}}
    with sqlite3.connect(tmp_path / "langgraph_lifecycle.sqlite") as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}])
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["executed"] is False
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, session, **proof)


def test_registered_dft_input_recovery_rejection_is_reconciled_without_execution(tmp_path):
    c, session, proof = fixture(tmp_path)
    reason = "方案未通过执行约束：post_recovery_only_exact_candidate_relax_allowed"
    session["submission"] = {"status": "rejected", "reason": reason}
    session["draft"]["action"] = {
        "tool": "prepare_local_batch_files", "operation": "prepare_dft_inputs",
        "parameters": {"mode": "dft_inputs"}, "target_ids": ["task1", "task2"],
        "_simulation_program": {"protocol": "registered_dft_inputs/v1", "status": "passed",
                                "target_ids": ["task1", "task2"]},
    }
    with sqlite3.connect(tmp_path / "langgraph_lifecycle.sqlite") as conn:
        kind, data = conn.execute("SELECT type,checkpoint FROM checkpoints").fetchone()
        payload = JsonPlusSerializer().loads_typed((kind, data))
        response = payload["channel_values"]["completed_response"]
        response.update(status="rejected", reason=reason, events=[{"status": "rejected", "reason": reason}])
        kind, data = JsonPlusSerializer().dumps_typed(payload)
        conn.execute("UPDATE checkpoints SET type=?,checkpoint=?", (kind, data))
    proof["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    assert rejection_checkpoint(c.state_path, session, **proof)["classification"] == "dft_input_recovery_gate_rejected"
    bad = deepcopy(session)
    bad["draft"]["action"]["target_ids"] = ["task1"]
    with pytest.raises(ValueError):
        rejection_checkpoint(c.state_path, bad, **proof)
    with sqlite3.connect(tmp_path / "execution_receipts.sqlite") as conn:
        conn.execute("CREATE TABLE receipts (identity TEXT)")
        conn.execute("INSERT INTO receipts VALUES (?)", (json.dumps({"invocation_id": "req1"}),))
    with pytest.raises(ValueError, match="execution_receipt"):
        rejection_checkpoint(c.state_path, session, **proof)

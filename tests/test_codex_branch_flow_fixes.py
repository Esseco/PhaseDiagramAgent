import hashlib
import json
import sqlite3
from copy import deepcopy

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from phase_agent.agents.simulation.agent import action_hash, prepare_proposal
from phase_agent.agents.strategic.codex_submission_rejection import rejection_checkpoint
from phase_agent.runtime.codex.context import build_codex_context
from tests.test_codex_native_team import make_control, team
from tests.test_codex_submission_rejection import fixture


def test_codex_context_excludes_stale_pool_reference(tmp_path):
    control, _ = make_control(tmp_path)
    control.state_path.write_text(
        json.dumps(
            {
                "current_state_snapshot": {
                    "decision_context": {
                        "evidence_catalog": {
                            "references": {"branch_pool:stale": {"kind": "branch_pool"}}
                        }
                    }
                }
            }
        ),
        encoding="utf8",
    )
    control.chat_handler.workflow_kwargs = {
        "config_session": {"config": {"agent": {"allowed_tools": []}}}
    }
    refs = build_codex_context(control.chat_handler)["decision_context"]["evidence_catalog"][
        "references"
    ]
    assert "branch_pool:stale" not in refs


def test_default_branch_decision_expands_before_plan_and_rejects_changes(tmp_path, monkeypatch):
    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["generate_branches"]
    context["decision_context"]["branch_pool_facts"] = {
        "existing_branches": [{"branch_id": "b1"}, {"branch_id": "b2"}]
    }
    monkeypatch.setattr(
        "phase_agent.agents.common.action_contracts.action_contract_errors", lambda a: []
    )
    monkeypatch.setattr("phase_agent.agents.permissions.missing_role_reasons", lambda *a, **k: {})
    t = team(control)
    view = t.start(question="sample", consult=[])
    action = {
        "tool": "generate_branches",
        "parameters": {},
        "branch_pool_review": {
            "existing_branch_groups": [],
            "default_existing_branch_decision": "defer",
            "default_existing_branch_reason": "keep for later",
            "default_existing_branch_evidence_refs": ["e1"],
        },
    }
    draft = t.draft(view["team_id"], 1, action, rationale="sample")
    frozen = draft["draft"]["action"]
    review = frozen["branch_pool_review"]
    assert "default_existing_branch_decision" not in review
    assert review["existing_branch_groups"] == [
        {
            "decision": "defer",
            "branch_ids": ["b1", "b2"],
            "reason": "keep for later",
            "evidence_refs": ["e1"],
        }
    ]
    plan = {
        "action_hash": action_hash(frozen),
        "rationale": "planned",
        "steps": [
            {"kind": "preflight", "description": "check"},
            {"kind": "invoke", "tool": "generate_branches", "description": "run"},
            {"kind": "verify", "description": "verify"},
        ],
        "checks": ["check"],
        "failure_policy": "return_to_strategic",
    }
    plan["context_binding"] = draft["requests"]["simulation"]["context_binding"]
    t.simulate(view["team_id"], 1, plan)
    action = t._load()["draft"]["action"]
    prepare_proposal({"raw_action": deepcopy(action)}, None, state={}, config={}, registry={})
    changed = deepcopy(action)
    changed["branch_pool_review"]["existing_branch_groups"][0]["branch_ids"].pop()
    with pytest.raises(ValueError, match="simulation_intent_changed"):
        prepare_proposal({"raw_action": changed}, None, state={}, config={}, registry={})


def test_default_expansion_rejection_reconciles_only_with_proof_and_no_receipt(tmp_path):
    control, session, proof = fixture(tmp_path)
    reason = "Simulation planning: simulation_intent_changed; request new planning and approval"
    intent = {
        "tool": "generate_branches",
        "parameters": {},
        "branch_pool_review": {
            "existing_branch_groups": [],
            "default_existing_branch_decision": "defer",
            "default_existing_branch_reason": "later",
            "default_existing_branch_evidence_refs": ["e1"],
        },
    }
    plan = {
        "action_hash": action_hash(intent),
        "rationale": "planned",
        "steps": [
            {"kind": "preflight", "description": "check"},
            {"kind": "invoke", "tool": "generate_branches", "description": "run"},
            {"kind": "verify", "description": "verify"},
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

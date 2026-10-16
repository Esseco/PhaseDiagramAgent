from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.graphs.dialogue.nodes import route_request
from phase_agent.graphs.dialogue.state import TurnRuntime
from phase_agent.tools.policy.file_approval import proposal_hash
from phase_agent.tools.state.execution_receipts import (
    begin_execution,
    record_execution_return,
    recovery_report,
)
from phase_agent.tools.workflows.prepare_dft_proposal import prepare_dft_proposal
from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection
from tests.test_approval_history_performance import rejected_template_state


@pytest.mark.parametrize(
    "revision",
    [
        None,
        {"revision_status": "revision_failed", "analysis": "model unavailable"},
        {"revision_status": "revised", "action": None},
    ],
)
def test_preparation_failure_retains_original_action_and_reason(monkeypatch, revision):
    monkeypatch.setattr(
        "phase_agent.tools.workflows.prepare_dft_proposal.attach_dft_preview",
        lambda *args: (None, "original validation error"),
    )
    action = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]},
        "post_dft_review": {"evidence": "preserved"},
    }
    before = deepcopy(action)
    result = prepare_dft_proposal(
        action, {}, {}, {}, agent_client=Mock(), revise=Mock(return_value=revision)
    )
    assert result.status == "rejected"
    assert "original validation error" in result.error
    assert result.action == before == action
    assert result.action is not action


def test_initial_proposal_handles_failed_preparation_without_none_access(monkeypatch):
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    monkeypatch.setattr(
        "phase_agent.tools.workflows.prepare_dft_proposal.attach_dft_preview",
        lambda *args: (None, "original validation error"),
    )
    forbidden = Mock(side_effect=AssertionError("must not execute"))
    action = {"tool": "select_dft_candidates", "parameters": {}, "post_dft_review": {"saved": True}}
    revise = Mock(
        return_value={"revision_status": "revision_failed", "analysis": "model unavailable"}
    )
    result = select_initial_proposal(
        current={},
        mode="interactive",
        config={"agent": {"allowed_tools": ["select_dft_candidates"]}},
        context={"unified_dialogue": True},
        decision_state={},
        registry={"select_dft_candidates": {"handler": forbidden}},
        agent_client=Mock(),
        invocation_id="new",
        propose=Mock(return_value=action),
        revise=revise,
        prepare_debug=forbidden,
        is_verified_second=lambda *a: False,
        mc_block=forbidden,
        model_failed=lambda a: False,
        record_id_factory=forbidden,
    )
    assert result["status"] == "rejected"
    assert "original validation error" in result["reason"]
    assert result["post_dft_review"] == {"saved": True}
    revise.assert_called_once()
    forbidden.assert_not_called()


def _failed(state_path, error):
    state = rejected_template_state()
    record = state["action_records"][-1]
    action = record["agent_proposal"]["raw_action"]
    record["status"] = "failed"
    record["execution_result"] = {
        "status": "failed",
        "error": error,
        "operation_outcome": {
            "config_version": "v",
            "model_version": "m",
            "task_ids": [],
            "artifacts": [],
        },
    }
    identity = {
        "invocation_id": record["record_id"],
        "config_version": "v",
        "tool": action["tool"],
        "action_hash": proposal_hash(action),
    }
    begin_execution(state_path, identity, action=action)
    record_execution_return(state_path, identity, "failed")
    state["invocations"] = {
        record["record_id"]: {"status": "failed", "execution": record["execution_result"]}
    }
    return state, identity


@pytest.mark.parametrize(
    "error", ["TypeError", "disk full", "connection lost", "unexpected internal failure"]
)
def test_any_failed_side_effecting_execution_requires_recovery_not_new_decision(tmp_path, error):
    path = tmp_path / "state.json"
    state, identity = _failed(path, error)
    before = deepcopy(state)
    report = recovery_report(path, state)
    assert report["status"] == "reconciliation_required"
    assert any(row["identity"] == identity for row in report["unsettled"])
    assert reusable_selection(state) is None
    handler = SimpleNamespace(config_delegate=None, state_path=path, conversation_id=None)
    for command in ["继续", "同意"]:
        context = TurnRuntime(handler, [], "same-thread")
        result = route_request(
            {"message": command, "facts": state}, SimpleNamespace(context=context)
        )
        assert "原选点与审批方案保留" in result["reply"]
        assert context.scientific_call is None
    assert state == before


def test_missing_receipt_and_parent_reservation_do_not_erase_failed_plan(tmp_path):
    state = rejected_template_state()
    record = state["action_records"][-1]
    record["status"] = "failed"
    record["execution_result"] = {"status": "failed", "error": "arbitrary failure"}
    report = recovery_report(tmp_path / "state.json", state)
    assert report["status"] == "reconciliation_required"
    assert report["unsettled"][0]["reason"] == "failed_action_requires_recovery"


@pytest.mark.parametrize(
    "change", [None, "config", "model", "identity", "tasks", "registered_effects"]
)
def test_exact_selection_restored_only_after_bound_no_effect_review(tmp_path, change):
    path = tmp_path / "state.json"
    state, identity = _failed(path, "arbitrary failure, not classified by string")
    state["execution_reconciliations"] = {
        "old": {
            "resolution": "verified_no_effect",
            "identity": identity,
            "evidence": {"attestation": "reviewed"},
            "reviewed_at": "2026-10-16",
        }
    }
    state["action_records"][-1]["status"] = "reconciled_no_effect"
    if change == "config":
        state["confirmed_config_version"] = "changed"
    elif change == "model":
        state["active_model_version"] = "changed"
    elif change == "identity":
        identity["action_hash"] = "changed"
    elif change == "tasks":
        state["tasks"] = [{"parent_decision_id": "old"}]
    elif change == "registered_effects":
        state["execution_reconciliations"]["old"]["resolution"] = "verified_registered_effects"
    before = deepcopy(state)
    result = reusable_selection(state)
    assert state == before
    if change is None:
        assert result["raw_action"] == state["action_records"][-1]["agent_proposal"]["raw_action"]
        assert recovery_report(path, state)["status"] == "clear"
        assert not begin_execution(path, identity)["allowed"]  # Original call is never replayed.
    else:
        assert result is None

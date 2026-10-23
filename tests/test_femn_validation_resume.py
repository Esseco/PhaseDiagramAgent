"""Resume a pre-task DFT validation rejection without another model selection."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection
from tests.test_approval_history_performance import rejected_template_state
from tests.test_dft_DH_coverage import candidate


def rejected_selection():
    state = rejected_template_state()
    record = state["action_records"][-1]
    action = record["agent_proposal"]["raw_action"]
    action["target_ids"] = ["s"]
    action["parameters"]["decisions"] = [
        {
            "candidate_id": "s",
            "action": "DFT_SINGLE_POINT",
            "reason": "independent D group",
        },
        {
            "candidate_id": "deferred",
            "action": "DEFER",
            "reason": "conditional group deferred",
        },
    ]
    action["parameters"]["dft_input_preview"] = {
        "schema_version": 2,
        "relative_cost": 999,
    }
    state["qbc_candidates"] = [candidate("s", "A"), candidate("deferred", "B")]
    for row in state["qbc_candidates"]:
        row["atom_count"] = 40
    execution = record["execution_result"]
    execution["result"] = {
        "status": "rejected",
        "tasks": [],
        "validation": {
            "valid": False,
            "errors": ["unknown_candidate:deferred"],
            "rejected": [],
        },
    }
    execution["operation_outcome"].update(
        artifacts=[],
        dispatch_status="completed",
        result_status="rejected",
        tool="select_dft_candidates",
    )
    return state


def resume(state, config=None):
    forbidden = Mock(
        side_effect=AssertionError("must not call model or prepare/execute tasks")
    )
    config = config or {
        "agent": {"allowed_tools": ["select_dft_candidates"]},
        "qbc": default_dft_decision_config(),
    }
    result = select_initial_proposal(
        current=state,
        mode="interactive",
        config=config,
        context={"user_message": "继续"},
        decision_state={},
        registry={"select_dft_candidates": {"handler": forbidden}},
        agent_client=forbidden,
        invocation_id="new",
        propose=forbidden,
        revise=forbidden,
        prepare_debug=forbidden,
        is_verified_second=forbidden,
        mc_block=forbidden,
        model_failed=forbidden,
        record_id_factory=lambda *args: "new",
    )
    forbidden.assert_not_called()
    return result


def test_resume_revalidates_complete_decisions_without_model_or_mutation():
    state = rejected_selection()
    before = deepcopy(state)
    result = resume(state)
    proposal = result["proposal"]
    raw = proposal["raw_action"]
    assert proposal["reuse_source_record_id"] == "old"
    assert (
        raw["parameters"]["decisions"]
        == before["action_records"][-1]["agent_proposal"]["raw_action"]["parameters"][
            "decisions"
        ]
    )
    assert raw["target_ids"] == ["s"]
    assert raw["parameters"]["dft_input_preview"]["task_count"] == 1
    assert raw["budget"] != 999
    assert result["record_id"] == "new"
    assert "human_feedback" not in result and "execution" not in result
    assert state == before


def test_resumed_plan_requires_new_approval_even_if_old_feedback_is_approve(
    monkeypatch,
):
    from phase_agent.tools.workflows.react_proposal import prepare_react_proposal

    state = rejected_selection()
    state["pending_execution_policies"] = {}
    before = deepcopy(state)
    config = {
        "agent": {"allowed_tools": ["select_dft_candidates"]},
        "qbc": default_dft_decision_config(),
    }
    forbidden = Mock(side_effect=AssertionError("must not execute or call model"))
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.agent_state_summary", lambda s: {}
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.react_proposal.validate_tool_action",
        lambda *a: {"valid": True},
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_template_review.gate_template",
        lambda proposal, *a, **kw: (proposal, False),
    )
    frame = {
        "current": state,
        "context": {"user_message": "继续"},
        "invocation_id": "new",
        "config": config,
        "session": {},
        "mode": "interactive",
        "agent_client": forbidden,
        "registry": {"select_dft_candidates": {"handler": forbidden}},
        "human_feedback": {"decision": "approve"},
    }
    result = prepare_react_proposal(frame)
    assert result["_graph_prepared"]
    assert result["human_feedback"] is None and result["stored"] is None
    assert result["proposal"]["reuse_source_record_id"] == "old"
    assert result["record_id"] != "old"
    assert state == before
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "defect", ["missing_group", "duplicate", "unknown", "environment", "audit"]
)
def test_current_preflight_failure_preserves_plan_without_reselection(defect):
    state = rejected_selection()
    action = state["action_records"][-1]["agent_proposal"]["raw_action"]
    config = {
        "agent": {"allowed_tools": ["select_dft_candidates"]},
        "qbc": default_dft_decision_config(),
    }
    if defect == "missing_group":
        action["parameters"]["decisions"].pop()
    elif defect == "duplicate":
        action["parameters"]["decisions"].append(
            deepcopy(action["parameters"]["decisions"][0])
        )
    elif defect == "unknown":
        action["parameters"]["decisions"][0]["candidate_id"] = "unknown"
    elif defect == "environment":
        config["upload_batches_directory"] = "synthetic-remote-inputs"
    elif defect == "audit":
        state["dft_audit_candidate_ids"] = ["deferred"]
    before = deepcopy(state)
    result = resume(state, config)
    assert result["status"] == "rejected" and result["submitted"] is False
    assert "原方案保留" in result["reason"]
    assert "proposal" not in result
    assert state == before


@pytest.mark.parametrize(
    "effect",
    [
        "tasks",
        "artifacts",
        "submitted",
        "parent",
        "effective",
        "config",
        "model",
        "failed",
        "unknown_validation",
    ],
)
def test_resume_remains_closed_for_effects_or_changed_binding(effect):
    state = rejected_selection()
    record = state["action_records"][-1]
    outcome = record["execution_result"]["operation_outcome"]
    if effect == "tasks":
        outcome["task_ids"] = ["existing"]
    elif effect == "artifacts":
        outcome["artifacts"] = [{"path": "existing"}]
    elif effect == "submitted":
        outcome["submitted"] = True
    elif effect == "parent":
        state["tasks"] = [{"parent_decision_id": "old"}]
    elif effect == "effective":
        state["effective_decisions"] = {"frozen-dft": {"status": "completed"}}
    elif effect == "config":
        state["confirmed_config_version"] = "changed"
    elif effect == "model":
        state["active_model_version"] = "changed"
    elif effect == "failed":
        record["execution_result"]["status"] = "failed"
    elif effect == "unknown_validation":
        record["execution_result"]["result"].pop("validation")
    before = deepcopy(state)
    assert reusable_selection(state) is None
    assert state == before

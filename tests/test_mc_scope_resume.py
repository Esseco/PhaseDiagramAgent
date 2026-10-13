from copy import deepcopy
from unittest.mock import patch

from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.tools.workflows.run_tool_step import _initialize_tool_step, run_tool_step
from phase_agent.tools.workflows.tool_outcomes import _store_completed
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.policy.execution_policy import build_agent_proposal


def fixture():
    session = confirm_config_snapshot(
        create_config_draft(default_layered_search_config()), user_confirmed=True
    )
    action = {
        "tool": "allocate_mc_bohb",
        "target_ids": [f"B{i}" for i in range(24)],
        "parameters": {
            "mc_budget": 900,
            "budget_preview": {
                "compression_requires_user_choice": True,
                "full_plan_steps": 1880,
                "full_plan_branch_count": 24,
                "selected_branch_count": 9,
                "target_steps": 900,
                "allocations": [],
            },
        },
        "budget": 0,
        "reason": "MC",
        "expected_purpose": "prepare MC inputs",
    }
    proposal = build_agent_proposal(action, {})
    waiting = {
        "status": "awaiting_approval",
        "agent_proposal": proposal,
        "execution": {
            "status": "completed",
            "result": {"status": "awaiting_approval", "reason": "choose_full_plan_or_compression"},
        },
    }
    state = {
        "tasks": [],
        "pending_execution_policies": {"plan": {"record_id": "plan", "agent_proposal": proposal}},
        "invocations": {"plan": waiting},
    }
    return session, state


def test_legacy_scope_wait_is_not_completed_replay():
    session, state = fixture()
    frame = _initialize_tool_step(
        state, session, registry={}, invocation_id="plan", execution_mode="interactive"
    )
    assert frame.get("_graph_prepared")
    assert not frame.get("idempotent_replay")


def test_scope_wait_is_not_stored_as_completed():
    _, state = fixture()
    _store_completed(state, "another", state["invocations"]["plan"])
    assert "another" not in state["invocations"]


def test_completed_execution_stays_idempotent():
    session, state = fixture()
    state["invocations"]["plan"]["status"] = "prepared"
    frame = _initialize_tool_step(state, session, registry={}, invocation_id="plan")
    assert frame["idempotent_replay"]


def test_wait_record_with_existing_child_task_cannot_reexecute():
    session, state = fixture()
    state["tasks"] = [{"task_id": "already-created", "parent_decision_id": "plan"}]
    frame = _initialize_tool_step(state, session, registry={}, invocation_id="plan")
    assert frame["idempotent_replay"]


def test_legacy_wait_then_full_scope_then_approve_executes_once():
    session, state = fixture()
    calls = []
    registry = create_tool_registry(
        {"allocate_mc_bohb": lambda **kw: calls.append(kw["action"]) or {"status": "prepared"}}
    )

    def preview(action, *args):
        result = deepcopy(action)
        result["parameters"]["budget_preview"].update(
            compression_requires_user_choice=False, selected_branch_count=24, requested_steps=1880
        )
        return result

    def valid(*args, **kwargs):
        return {
            "valid": True,
            "errors": [],
            "config_version": session["confirmed_snapshot"]["config_version"],
        }

    context = {"user_message": "完整执行", "unified_dialogue": True}
    with (
        patch(
            "phase_agent.tools.workflows.prepare_mc_preview.prepare_mc_preview", side_effect=preview
        ),
        patch("phase_agent.tools.workflows.run_tool_step.validate_tool_action", side_effect=valid),
        patch("phase_agent.tools.workflows.react_proposal.validate_tool_action", side_effect=valid),
    ):
        chosen = run_tool_step(
            state,
            session,
            registry=registry,
            invocation_id="plan",
            execution_mode="interactive",
            context=context,
        )
        assert chosen["mc_coverage_question"] and not calls
        params = chosen["state"]["pending_execution_policies"]["plan"]["agent_proposal"][
            "raw_action"
        ]["parameters"]
        assert params["mc_budget"] == 1880
        record = chosen["state"]["pending_execution_policies"]["plan"]
        assert record["original_record_id"] == "plan"
        assert record["record_id"].startswith("plan:mc-scope:")
        approved = run_tool_step(
            chosen["state"],
            session,
            registry=registry,
            invocation_id="plan",
            execution_mode="interactive",
            context={"unified_dialogue": True},
            human_feedback={"decision": "approve", "comment": "同意"},
        )
        assert approved["status"] == "prepared", approved
        assert len(calls) == 1 and calls[0]["parameters"]["mc_budget"] == 1880
        repeated = run_tool_step(
            approved["state"],
            session,
            registry=registry,
            invocation_id="plan",
            execution_mode="interactive",
            human_feedback={"decision": "approve", "comment": "同意"},
        )
        assert repeated["idempotent_replay"] and len(calls) == 1

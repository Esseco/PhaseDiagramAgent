from copy import deepcopy
import pytest
from phase_agent.tools.state.mc_threshold_retry import validate_mc_threshold_retry
from phase_agent.tools.workflows.create_workflow_handlers import _restart_failed_task


def data():
    rows = []
    for n in range(3):
        rows.append(
            {
                "task_id": f"m{n}",
                "task_key": f"k{n}",
                "stage": "deep_search",
                "status": "failed",
                "model_version": "v",
                "parent_decision_id": "d",
                "budget": 1,
                "outputs": {
                    "error": "initial_relaxation_extrapolation_failure",
                    "extrapolation_monitor": {
                        "extrapolation_failure": True,
                        "failures": [{"score": 1.5, "trigger": "final"}],
                    },
                },
            }
        )
    return {"tasks": rows, "processed_task_ids": ["m0", "m1", "m2"], "active_model_version": "v"}


def test_scope_accepts_only_same_verified_mc_threshold_failure():
    s = data()
    ids = ["m0", "m1"]
    assert len(validate_mc_threshold_retry(s, ids)) == 2
    for mut in (
        lambda x: x["tasks"][1].update(model_version="old"),
        lambda x: x["tasks"][1].update(parent_decision_id="other"),
        lambda x: x["processed_task_ids"].remove("m1"),
        lambda x: x["tasks"][1]["outputs"]["extrapolation_monitor"]["failures"][0].update(
            score=3.1
        ),
        lambda x: x["tasks"][1].update(status="completed"),
    ):
        bad = deepcopy(s)
        mut(bad)
        with pytest.raises(ValueError):
            validate_mc_threshold_retry(bad, ids)
    with pytest.raises(ValueError):
        validate_mc_threshold_retry(s, ["m0", "m0"])


def test_batch_retry_is_atomic_and_preserves_original_tasks():
    s = data()
    before = deepcopy(s)
    action = {"target_ids": ["m0", "m1"], "parameters": {"repair_profile": "mc_threshold_recovery"}}
    result = _restart_failed_task(
        action=action,
        context={
            "event_state": s,
            "effective_config": {"budgets": {"stage_limits": {"deep_search": {}}}},
            "config_version": "c",
        },
    )
    assert result["status"] == "prepared"
    assert [x["retry_of"] for x in result["tasks"]] == ["m0", "m1"]
    assert s == before
    assert result["state"]["tasks"][:3] == before["tasks"]
    assert len(result["state"]["tasks"]) == 5
    assert all(x["status"] == "pending" for x in result["tasks"])


def test_batch_rejects_existing_retry_without_partial_state():
    s = data()
    s["tasks"].append({"task_id": "old", "task_key": "k1:retry:1"})
    before = deepcopy(s)
    result = _restart_failed_task(
        action={
            "target_ids": ["m0", "m1"],
            "parameters": {"repair_profile": "mc_threshold_recovery"},
        },
        context={
            "event_state": s,
            "effective_config": {"budgets": {"stage_limits": {"deep_search": {}}}},
            "config_version": "c",
        },
    )
    assert result["status"] != "prepared"
    assert result["state"] == before and s == before


def test_policy_allows_verified_batch_and_rejects_mixed_scope():
    from phase_agent.policies.validate_tool_action import validate_tool_action

    s = data()
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": "c",
            "config": {
                "agent": {"allowed_tools": ["restart_failed_task"]},
                "budgets": {"total_relative_cost": 100},
                "frozen_parameters": [],
            },
        },
    }
    action = {
        "tool": "restart_failed_task",
        "task_key": "retry-batch",
        "target_ids": ["m0", "m1"],
        "parameters": {"repair_profile": "mc_threshold_recovery"},
        "budget": 0,
    }
    registry = {"restart_failed_task": {"handler": lambda **_: None}}
    assert (
        "retry_target_not_failed_task"
        not in validate_tool_action(action, s, session, registry)["errors"]
    )
    s["tasks"][1]["parent_decision_id"] = "other"
    assert (
        "mc_threshold_retry_mixed_actions"
        in validate_tool_action(action, s, session, registry)["errors"]
    )


def test_budget_failure_rolls_back_all_prior_retry_reservations():
    s = data()
    before = deepcopy(s)
    budgets = {"enforcement": "hard", "stage_limits": {"deep_search": {"max_tasks": 1}}}
    result = _restart_failed_task(
        action={
            "target_ids": ["m0", "m1"],
            "parameters": {"repair_profile": "mc_threshold_recovery"},
        },
        context={"event_state": s, "effective_config": {"budgets": budgets}, "config_version": "c"},
    )
    assert result["status"] == "budget_exhausted"
    assert result["state"] == before and s == before


def test_advisory_repair_ignores_cumulative_old_task_count():
    s = data()
    s["budget_usage"] = {"stages": {"deep_search": {"tasks": 884}}}
    limits = {"enforcement": "advisory", "stage_limits": {"deep_search": {"max_tasks": 300}}}
    original = deepcopy(limits)
    result = _restart_failed_task(
        action={
            "target_ids": ["m0", "m1"],
            "parameters": {"repair_profile": "mc_threshold_recovery"},
        },
        context={"event_state": s, "effective_config": {"budgets": limits}, "config_version": "c"},
    )
    assert result["status"] == "prepared"
    assert limits == original


def test_proposer_admits_only_listed_batch_mc_retry_targets():
    from phase_agent.agents.strategic.propose_tool_action import propose_agent_tool_action

    state = {
        "config_version": "c",
        "decision_context": {"retryable_tasks": [{"task_id": "m0"}, {"task_id": "m1"}]},
    }
    action = {
        "tool": "restart_failed_task",
        "task_key": "retry-batch",
        "target_ids": ["m0", "m1"],
        "parameters": {"repair_profile": "mc_threshold_recovery"},
        "budget": 0,
    }
    result = propose_agent_tool_action(
        state, agent_client=lambda _: action, allowed_tools=["restart_failed_task"]
    )
    assert result["tool"] == "restart_failed_task" and result["target_ids"] == ["m0", "m1"]
    bad = {**action, "target_ids": ["m0", "other"]}
    result = propose_agent_tool_action(
        state, agent_client=lambda _: bad, allowed_tools=["restart_failed_task"]
    )
    assert result["tool"] != "restart_failed_task"

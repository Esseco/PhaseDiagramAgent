from copy import deepcopy

import pytest

from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.decisions.qbc_selection.dft_round_policy import dft_task_limit, round_dft_tasks
from phase_agent.decisions.qbc_selection.recommend_post_mc_dft import recommend_post_mc_dft
from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs


def config(legacy_cost=None):
    return {
        "mlip": {"name": "m"},
        "qbc": default_dft_decision_config(),
        "dft": {
            "selection": {"max_tasks_per_round": 100, "single_point_cost_per_round": legacy_cost}
        },
    }


def candidate(index, *, atoms=40):
    return {
        "candidate_id": f"new-{index}",
        "atom_count": atoms,
        "phase": "O3",
        "x_Na_per_O2": 0.5,
        "predicted_Ehull": 0.01,
        "geometry_optimization_evidence": {
            "source_task_id": f"relax-{index}",
            "residual_force_ev_per_angstrom": 0.1,
            "force_tolerance_ev_per_angstrom": 0.05,
        },
    }


def prior_tasks(count):
    return [
        {
            "task_id": f"old-{i}",
            "structure_id": f"old-structure-{i}",
            "stage": "dft_single_point" if i % 2 else "dft_relax",
            "model_version": "m",
            "generation_cycle": 1,
            "status": "completed" if i % 2 else "pending",
        }
        for i in range(count)
    ]


def preview(actions, *, prior=0, atoms=40, legacy_cost=None):
    candidates = [candidate(i, atoms=atoms) for i in range(len(actions))]
    state = {
        "active_model_version": "m",
        "generation_history": [{}],
        "tasks": prior_tasks(prior),
        "qbc_candidates": candidates,
    }
    action = {
        "parameters": {
            "decisions": [
                {"candidate_id": row["candidate_id"], "action": name, "reason": "geometry and hull"}
                for row, name in zip(candidates, actions)
            ]
        }
    }
    original = deepcopy((action, state))
    result = preview_dft_inputs(action, state, config(legacy_cost))
    assert (action, state) == original
    return result


@pytest.mark.parametrize("action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
def test_99_existing_tasks_allow_one_more(action):
    data, error = preview([action], prior=99)
    assert error is None and data["task_count"] == 1


def test_99_existing_tasks_reject_two_more():
    data, error = preview(["DFT_SINGLE_POINT", "DFT_RELAX"], prior=99)
    assert data is None and "合计最多 100" in error


def test_single_points_and_relaxations_share_the_100_task_cap():
    data, error = preview(["DFT_SINGLE_POINT"] * 90 + ["DFT_RELAX"] * 10)
    assert error is None
    assert (data["single_point_count"], data["relax_count"]) == (90, 10)
    data, error = preview(["DFT_SINGLE_POINT"] * 95 + ["DFT_RELAX"] * 6)
    assert data is None and "合计最多 100" in error


@pytest.mark.parametrize("legacy_cost", [None, 10, 5000])
def test_preview_above_5000_keeps_the_cost_estimate(legacy_cost):
    data, error = preview(["DFT_SINGLE_POINT"], atoms=240, legacy_cost=legacy_cost)
    assert error is None
    assert data["relative_cost"] == 6480


def test_small_plan_is_not_filled_to_100():
    data, error = preview(["DFT_SINGLE_POINT"] * 3)
    assert error is None
    assert data["task_count"] == 3 and data["relative_cost"] == 90


def test_round_count_only_includes_current_model_cycle_and_non_cancelled_dft():
    tasks = prior_tasks(2)
    tasks += [
        {**tasks[0], "task_id": "cancelled", "status": "cancelled"},
        {**tasks[0], "task_id": "old-model", "model_version": "old"},
        {**tasks[0], "task_id": "old-cycle", "generation_cycle": 0},
        {**tasks[0], "task_id": "mc", "stage": "deep_search"},
        {**tasks[0], "task_id": "failed", "status": "failed"},
    ]
    state = {"active_model_version": "m", "generation_history": [{}], "tasks": tasks}
    settings = config()
    settings["mlip"]["name"] = "stale-name"
    assert [row["task_id"] for row in round_dft_tasks(state, settings)] == [
        "old-0",
        "old-1",
        "failed",
    ]


def test_advisory_budget_does_not_remove_recommendation_task_cap():
    settings = config()
    settings["budgets"] = {"enforcement": "advisory"}
    pool = [candidate(i, atoms=240) for i in range(101)]
    state = {"active_model_version": "m", "generation_history": [{}]}
    result = recommend_post_mc_dft(pool, settings, state)
    assert result["summary"]["selected"] == 100
    assert result["summary"]["relative_cost"] == 648000
    state["tasks"] = prior_tasks(99)
    result = recommend_post_mc_dft(pool[:2], settings, state)
    assert result["summary"]["selected"] == 1
    state["tasks"] = prior_tasks(100)
    assert recommend_post_mc_dft(pool[:2], settings, state)["selected_candidates"] == []


def test_legacy_single_point_reference_count_does_not_replace_total_cap():
    settings = config()
    settings["dft"]["selection"]["single_point_max_per_round"] = 1
    result = recommend_post_mc_dft([candidate(i) for i in range(3)], settings)
    assert result["summary"]["selected"] == 3


@pytest.mark.parametrize("value", [None, 0, -1, 101, True, 1.5, "100"])
def test_invalid_total_task_limits_are_explicit(value):
    settings = config()
    settings["dft"]["selection"]["max_tasks_per_round"] = value
    with pytest.raises(ValueError, match="max_tasks_per_round"):
        dft_task_limit(settings)


def test_advisory_cumulative_stage_count_does_not_block_this_round():
    settings = config()
    settings["budgets"] = {
        "enforcement": "advisory",
        "stage_limits": {"dft_single_point": {"max_tasks": 1}},
    }
    state = {"budget_usage": {"stages": {"dft_single_point": {"tasks": 1}}}}
    result = recommend_post_mc_dft([candidate(i) for i in range(2)], settings, state)
    assert result["summary"]["selected"] == 2


def tool_validation(*, existing, selected=1, direct_stage=None):
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.tools.policy.validate_tool_action import validate_tool_action

    settings = default_layered_search_config()
    settings["budgets"]["enforcement"] = "advisory"
    settings["dft"]["selection"]["max_tasks_per_round"] = 100
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": "c",
            "config": settings,
        },
    }
    state = {
        "active_model_version": "m",
        "generation_history": [{}],
        "confirmed_config_version": "c",
        "tasks": prior_tasks(existing),
    }
    tool = "run_calculation_stage" if direct_stage else "select_dft_candidates"
    parameters = (
        {}
        if direct_stage
        else {
            "decisions": [
                {"candidate_id": f"new-{i}", "action": "DFT_SINGLE_POINT"} for i in range(selected)
            ],
            "dft_input_preview": {"schema_version": 2, "task_count": selected},
        }
    )
    action = {
        "tool": tool,
        "task_key": "old-preview",
        "target_ids": ["new-0"],
        "parameters": parameters,
        "budget": 30.0,
        "reason": "existing selection",
    }
    if direct_stage:
        action["stage"] = direct_stage
    registry = {
        tool: {
            "owner": "calculation" if direct_stage else "decision_layer",
            "handler": lambda **kwargs: None,
        }
    }
    return validate_tool_action(action, state, session, registry)


def test_stored_preview_is_rechecked_against_current_task_count():
    accepted = tool_validation(existing=99, selected=1)
    assert accepted["valid"], accepted["errors"]
    rejected = tool_validation(existing=99, selected=2)
    assert not rejected["valid"]
    assert "dft_round_task_limit" in rejected["errors"]


@pytest.mark.parametrize("stage", ["dft_single_point", "dft_relax"])
def test_direct_dft_stage_obeys_the_same_live_cap(stage):
    accepted = tool_validation(existing=99, direct_stage=stage)
    assert accepted["valid"], accepted["errors"]
    rejected = tool_validation(existing=100, direct_stage=stage)
    assert not rejected["valid"]
    assert "dft_round_task_limit" in rejected["errors"]

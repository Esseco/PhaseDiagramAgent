from phase_agent.tools.budget.check_budget import check_budget
from phase_agent.tools.budget.reserve_budget import reserve_budget
from phase_agent.tools.workflows.create_dft_selection_handler import _remaining_dft_budget


def test_task_limits_apply_and_llm_advisory_limits_still_warn():
    limits = {
        "total_relative_cost": 1,
        "stage_limits": {"deep_search": {"max_tasks": 0, "max_cost": 0}},
        "llm_limits": {"max_calls": 0, "max_calls_per_iteration": 0},
    }
    check = check_budget(
        {}, {"stage": "deep_search", "relative_cost": 100, "llm_usage": {"calls": 1}}, limits
    )
    assert not check["allowed"] and check["reasons"] == ["stage_task_limit"]
    assert set(check["warnings"]) == {"llm_calls_limit", "llm_iteration_call_limit"}
    result = reserve_budget({}, task_key="a", stage="deep_search", amount=100, limits=limits)
    assert result["status"] == "rejected"
    limits["stage_limits"]["deep_search"]["max_tasks"] = 2
    result = reserve_budget({}, task_key="a", stage="deep_search", amount=100, limits=limits)
    assert result["status"] == "reserved"
    assert (
        reserve_budget(
            result["state"], task_key="a", stage="deep_search", amount=100, limits=limits
        )["status"]
        == "existing"
    )
    limits["enforcement"] = "hard"
    assert check_budget({}, {"stage": "deep_search", "relative_cost": 100}, limits)["allowed"]
    assert not check_budget({}, {"stage": "undefined"}, {**limits, "enforcement": "advisory"})[
        "allowed"
    ]


def test_dft_reference_budget_does_not_cut_off_selection():
    config = {"budget_limits": {"total_relative_cost": 1}}
    state = {"budget_usage": {"total_relative_cost": 10}}
    assert _remaining_dft_budget(state, config) == float("inf")
    config["budget_limits"]["enforcement"] = "hard"
    assert _remaining_dft_budget(state, config) == float("inf")


def test_exhausted_reference_cost_does_not_end_phase_search():
    from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence
    from tests.test_single_point_convergence import RULES

    state = {"budget_remaining": 0, "budget_limits": {"enforcement": "advisory"}}
    assert check_global_convergence(state, rules=RULES)["status"] != "budget_exhausted"
    state["budget_limits"]["enforcement"] = "hard"
    assert check_global_convergence(state, rules=RULES)["status"] != "budget_exhausted"

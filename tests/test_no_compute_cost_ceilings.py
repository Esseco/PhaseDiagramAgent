from copy import deepcopy

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.configuration.runtime.build_effective_run_config import build_effective_run_config
from phase_agent.configuration.schema.validate_search_config import validate_search_config
from phase_agent.tools.budget.check_budget import check_budget


def test_cost_only_legacy_profile_loads_without_migrating_state(tmp_path):
    from phase_agent.configuration.session.project_config_json import (
        FORMAT_ID,
        PROFILE_ID,
        _legacy_cost_profile_digest,
        expand_project_config,
    )

    document = {"_format": FORMAT_ID, "profile": PROFILE_ID,
                "profile_digest": _legacy_cost_profile_digest(), "config": {}, "overrides": {}}
    result = expand_project_config(document, source=tmp_path / "project.json")
    assert result["budgets"]["total_relative_cost"] is None
    assert result["budgets"]["stage_limits"]["deep_search"]["max_tasks"] == 300


def test_legacy_cost_ceilings_do_not_block_but_task_count_does():
    limits = default_budget_rules()
    limits.update(enforcement="hard", total_relative_cost=1)
    limits["stage_limits"]["deep_search"].update(max_cost=1, max_tasks=2)
    state = {"budget_usage": {"total_relative_cost": 100, "stages": {"deep_search": {"cost": 100, "tasks": 1}}}}
    assert check_budget(state, {"stage": "deep_search", "relative_cost": 1000, "tasks": 1}, limits)["allowed"]
    result = check_budget(state, {"stage": "deep_search", "relative_cost": 1, "tasks": 2}, limits)
    assert not result["allowed"] and "stage_task_limit" in result["reasons"]


def test_task_limit_is_hard_even_with_advisory_cost_policy():
    limits = default_budget_rules()
    limits["stage_limits"]["deep_search"]["max_tasks"] = 0
    result = check_budget({}, {"stage": "deep_search", "relative_cost": 1}, limits)
    assert not result["allowed"] and "stage_task_limit" in result["reasons"]


def test_model_call_limits_remain_in_force():
    limits = default_budget_rules()
    limits["enforcement"] = "hard"
    limits["llm_limits"]["max_calls"] = 0
    result = check_budget({}, {"llm_usage": {"calls": 1}}, limits)
    assert "llm_calls_limit" in result["reasons"]


def test_unlimited_compute_cost_does_not_accept_invalid_accounting_values():
    limits = default_budget_rules()
    for cost in (-1, float("nan"), float("inf"), True):
        assert not check_budget({}, {"stage": "deep_search", "relative_cost": cost}, limits)["allowed"]


def test_new_defaults_and_old_effective_config_have_no_compute_cost_caps():
    config = default_layered_search_config()
    assert config["budgets"]["total_relative_cost"] is None
    assert all(row["max_cost"] is None for row in config["budgets"]["stage_limits"].values())
    assert "budgets.total_relative_cost" not in validate_search_config(config).get("ambiguities", [])
    legacy = deepcopy(config)
    legacy["budgets"]["total_relative_cost"] = 1
    legacy["budgets"]["stage_limits"]["deep_search"]["max_cost"] = 1
    legacy["mc_policy"]["max_cumulative_cost_per_branch"] = 1
    legacy["round_strategy"]["maximum_dft_budget"] = 1
    task_limit = legacy["budgets"]["stage_limits"]["deep_search"]["max_tasks"]
    llm_limits = deepcopy(legacy["budgets"]["llm_limits"])
    session = {"status": "confirmed", "confirmed_snapshot": {"config": legacy, "config_version": "v1"}}
    effective = build_effective_run_config(session)
    assert effective["budgets"]["total_relative_cost"] is None
    assert effective["budgets"]["stage_limits"]["deep_search"]["max_cost"] is None
    assert effective["budgets"]["stage_limits"]["deep_search"]["max_tasks"] == task_limit
    assert effective["budgets"]["llm_limits"] == llm_limits
    assert effective["mc_policy"]["max_cumulative_cost_per_branch"] is None
    assert effective["round_strategy"]["maximum_dft_budget"] is None
    assert session["confirmed_snapshot"]["config"] == legacy

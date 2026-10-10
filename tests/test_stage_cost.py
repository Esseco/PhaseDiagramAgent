import pytest

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def test_size_and_mc_work_change_estimate():
    assert estimate_stage_cost("dft_single_point", atom_count=40)["value"] == 30
    assert estimate_stage_cost("dft_single_point", atom_count=80)["value"] == 240
    assert estimate_stage_cost("dft_relax", atom_count=40)["value"] == 900
    assert estimate_stage_cost("dft_relax", atom_count=80)["value"] == 7200
    assert estimate_stage_cost("deep_search", atom_count=40, mc_steps=10)["value"] == 1
    assert estimate_stage_cost("deep_search", atom_count=80, mc_steps=10)["value"] == pytest.approx(2**1.2)
    assert estimate_stage_cost("dft_single_point")["basis"] == "reference_size_assumed"
    with pytest.raises(ValueError):
        estimate_stage_cost("dft_single_point", atom_count=float("nan"))


def test_confirmed_costs_propagate_and_running_relax_consumes_stage_capacity():
    budgets = default_budget_rules()
    budgets["stage_limits"]["dft_single_point"]["task_cost"] = 30
    config = default_dft_decision_config(budgets=budgets)
    config["selection_policy"] = {"max_relax_fraction": 1.0}  # Isolate stage capacity, not selection policy.
    assert config["action_costs"]["DFT_SINGLE_POINT"] == 30
    state = {"budget_reservations": {"running": {"status": "running", "stage": "dft_relax", "reserved_cost": 900}}}
    proposal = {"decisions": [{"candidate_id": str(i), "action": "DFT_RELAX"} for i in range(2)]}
    result = validate_dft_agent_decisions(proposal, [{"candidate_id": str(i), "atom_count": 40} for i in range(2)], state, config=config, config_version="v", remaining_budget=3000)
    assert len(result["accepted"]) == 1
    assert result["reserved_cost"] == 900
    assert "stage_task_limit" in result["rejected"][0]["details"]

"""Configuration for fixed and agent QBC decision modes."""

from config_layer.defaults.default_budget_rules import default_budget_rules
from copy import deepcopy


def default_dft_decision_config(*, budgets=None) -> dict:
    defaults = default_budget_rules()
    supplied = deepcopy(budgets or {})
    budgets = {**defaults, **supplied}
    budgets["stage_limits"] = {name: {**row, **supplied.get("stage_limits", {}).get(name, {})} for name, row in defaults["stage_limits"].items()}
    return {
        "mode": "agent",
        "allowed_actions": ["DFT_SINGLE_POINT", "DFT_RELAX", "DEFER", "REJECT"],
        "action_costs": {"DFT_SINGLE_POINT": budgets["stage_limits"]["dft_single_point"]["task_cost"], "DFT_RELAX": budgets["stage_limits"]["dft_relax"]["task_cost"], "DEFER": 0.0, "REJECT": 0.0},
        "cost_model": budgets.get("cost_model", {}),
        "budget_limits": budgets,
        "extreme_uncertainty": {"f_std_max": None, "energy_std": None, "hard_action": "DFT_SINGLE_POINT"},
        "near_hull_threshold": 0.10,
        "fixed": {"batch_size": 10, "seed": 0, "weights": None, "audit_fraction": 0.1},
        "retrain": {"minimum_new_dft_records": 10},
    }

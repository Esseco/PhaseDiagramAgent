"""Estimate only configured relative bounds during dialogue mode."""


def estimate_draft_cost(config: dict) -> dict:
    budgets = config.get("budgets") or {}
    stages = budgets.get("stage_limits") or {}
    known = {name: item.get("max_cost") for name, item in stages.items()}
    return {"status": "estimated", "kind": "configured_upper_bounds", "total_relative_cost": budgets.get("total_relative_cost"), "stage_max_costs": known, "wall_time": None, "note": "No formal task was submitted."}

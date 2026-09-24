"""Validate LLM strategy before it can define a round snapshot."""


def validate_round_strategy(strategy: dict, summary: dict, *, config: dict) -> dict:
    errors = []
    allowed = set(config.get("allowed_generation_strategies") or [])
    quotas = strategy.get("generation_quotas")
    if not isinstance(quotas, dict) or any(name not in allowed or not isinstance(value, int) or value < 0 for name, value in (quotas or {}).items()):
        errors.append("invalid_generation_quotas")
    elif sum(quotas.values()) > int(config.get("generation_quota_total", 0)):
        errors.append("generation_quota_exceeded")
    regions = strategy.get("focus_regions")
    known = set(summary.get("known_region_ids") or [])
    if not isinstance(regions, list) or (known and any(item not in known for item in regions)):
        errors.append("unknown_focus_region")
    for field, maximum in (("mc_budget", config.get("maximum_mc_budget")), ("dft_budget", config.get("maximum_dft_budget"))):
        value = strategy.get(field)
        if not isinstance(value, (int, float)) or value < 0 or (maximum is not None and value > maximum):
            errors.append(f"invalid_{field}")
    exploration = strategy.get("exploration_fraction", config.get("minimum_exploration_fraction", 0.1))
    if not isinstance(exploration, (int, float)) or exploration < config.get("minimum_exploration_fraction", 0.1) or exploration > 1:
        errors.append("invalid_exploration_fraction")
    forbidden = {"branch_id", "branch_ids", "selected_branches", "branch_scores", "branch_budgets", "task_id", "task_key", "budget_level", "promotion", "dft_structure_ids", "energy", "ehull", "converged"} & set(strategy)
    if forbidden:
        errors.append("task_level_fields_forbidden")
    return {"valid": not errors, "errors": errors}

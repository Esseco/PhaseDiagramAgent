"""Classify configuration paths as hard constraints or adjustable policy."""


def classify_config_permission(path: str, config: dict) -> str:
    hard_prefixes = ("system.constraints", "system.boundary", "frozen_parameters", "dft.parameters", "budgets.total_relative_cost", "convergence")
    if path in set(config.get("frozen_parameters") or []) or path.startswith(hard_prefixes):
        return "hard_constraint"
    adjustable = ("generation_actions.quotas", "agent", "calculation.mc_allocator", "budgets.stage_limits")
    if path.startswith(adjustable):
        return "adjustable_policy"
    return "unclassified"

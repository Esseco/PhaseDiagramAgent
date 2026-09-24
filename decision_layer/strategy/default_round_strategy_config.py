"""Allowed bounds for round-level LLM strategy decisions."""


def default_round_strategy_config() -> dict:
    return {
        "allowed_generation_strategies": ["coverage", "composition", "competing_phase", "tm_ordering", "periodic_extension", "random_exploration"],
        "generation_quota_total": 20,
        "minimum_exploration_fraction": 0.1,
        "maximum_mc_budget": 1000,
        "maximum_dft_budget": 5000,
        "rule_default": {"focus_regions": [], "generation_quotas": {"coverage": 10, "random_exploration": 2}, "mc_budget": 300, "dft_budget": 600, "reason": "rule fallback"},
    }

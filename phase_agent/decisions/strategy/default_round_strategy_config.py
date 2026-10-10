"""Allowed bounds for round-level LLM strategy decisions."""


def default_round_strategy_config() -> dict:
    return {
        "allowed_generation_strategies": [
            "coverage",
            "composition",
            "competing_phase",
            "tm_ordering",
            "periodic_extension",
            "random_exploration",
        ],
        "generation_quota_total": 300,
        "minimum_exploration_fraction": 0.1,
        "maximum_mc_budget": 3000,
        "maximum_dft_budget": 6000,
        "rule_default": {
            "focus_regions": [],
            "generation_quotas": {
                "coverage": 200,
                "composition": 40,
                "competing_phase": 20,
                "tm_ordering": 20,
                "periodic_extension": 20,
            },
            "mc_budget": 900,
            "dft_budget": 720,
            "reason": "rule fallback",
        },
    }

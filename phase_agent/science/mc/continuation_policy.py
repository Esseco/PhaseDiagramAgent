"""Shared runtime MC continuation limits; old confirmed policy remains readable."""

from copy import deepcopy


def effective_mc_policy(config):
    policy = deepcopy(config.get("mc_policy") or {})
    policy["max_segments_per_branch"] = 3  # Initial segment plus two additions.
    policy["minimum_continuation_tasks"] = max(
        10, int(policy.get("minimum_continuation_tasks", 10))
    )
    policy["budget_enforcement"] = (config.get("budgets") or {}).get("enforcement", "advisory")
    # Apply the approved shared policy to new work, including old saved configs.
    policy.update(shared_hb_policy())
    policy["protected_ehull_ev_per_atom"] = float(
        (config.get("convergence") or {}).get("phase_competition_gap_ev_per_atom", 0.003)
    )
    return policy


def shared_hb_policy():
    return {
        "successive_halving_enabled": True,
        "halving_eta": 2,
        "coverage_structure": True,
        "initial_ehull_bands": [
            {"max_ehull_ev_per_atom": gap, "max_mc_steps": steps, "patience_steps": steps // 10}
            for gap, steps in ((0.010, 100), (0.020, 80), (0.040, 60))
        ],
        "exploration_max_mc_steps": 40,
        "exploration_patience_steps": 4,
        "high_ehull_defer_threshold": 0.040,
        "improvement_strong_ev_per_atom": 0.003,
        "improvement_moderate_ev_per_atom": 0.001,
        "improvement_small_ev_per_atom": 0.0005,
        "continuation_exploration_fraction": 0.10,
        "continuation_resource_tiers": {
            "high": {"max_mc_steps": 200, "patience_steps": 20},
            "medium": {"max_mc_steps": 150, "patience_steps": 15},
            "low": {"max_mc_steps": 100, "patience_steps": 10},
            "exploration": {"max_mc_steps": 50, "patience_steps": 5},
        },
        "stopping_threshold_source": "library_default",
    }

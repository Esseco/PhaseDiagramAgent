"""Configuration for fair branch-surrogate comparisons."""


def default_surrogate_comparison_config() -> dict:
    return {
        "target_key": "distance_to_fixed_hull",
        "trial_energy_key": "trial_energy_per_atom",
        "low_energy_fraction": 0.1,
        "selection_batch_size": 5,
        "initial_revealed": 10,
        "total_cost_budget": None,
        "cost_basis": "proxy_relative",
        "measured_cost_unit": None,
        "cost_model": {"reference_atoms": 1.0, "atom_exponent": 1.0, "scale": 1.0},
        "structure_limits": {"max_atoms": 500, "max_det_H": 64, "max_proxy_cost_per_task": None},
        "random_seed": 0,
        "methods": ["random", "trial_energy", "manual_rf", "embedding_rf", "hybrid_rf", "mattertune"],
        "method_costs": {"random": 0.0, "trial_energy": 0.0, "manual_rf": None, "embedding_rf": None, "hybrid_rf": None, "mattertune": None},
        "recommendation": {"primary_metric": "important_miss_rate", "require_replay": True, "minimum_test_rows": 10},
    }

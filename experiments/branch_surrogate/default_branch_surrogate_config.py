"""Default, deliberately strict dataset settings."""


def default_branch_surrogate_config() -> dict:
    return {
        "task": "mlip_mc_fixed_budget_hull_distance",
        "target_mc_budget": None,
        "mlip_version": None,
        "hull_reference_version": None,
        "energy_unit": "eV",
        "feature_fields": ["P", "H", "det_H", "x", "T", "composition"],
        "allow_config_features": True,
        "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 0},
        "group_mode": "framework",
    }

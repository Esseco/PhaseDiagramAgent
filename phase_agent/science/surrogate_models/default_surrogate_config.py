"""Default configuration for branch surrogate experiments."""


def default_surrogate_config() -> dict:
    return {
        "feature_set_version": "manual-v1",
        "enabled_features": ["atom_count", "volume_per_atom"],
        "relax": {
            "model_path": None,
            "model_version": None,
            "environment": "py-mace",
            "parameters": {},
            "cost_model": {"reference_atoms": 1.0, "atom_exponent": 1.0, "scale": 1.0},
        },
        "embedding": {"enabled": False, "extractor": None, "version": None},
        "selection": {
            "missing_fraction_max": 0.0,
            "variance_min": 1e-12,
            "redundancy_correlation_max": 0.95,
            "relevance_top_k": None,
            "model_top_k": None,
        },
        "rf": {"n_estimators": 200, "random_state": 0, "n_jobs": -1},
        "methods": ["manual_rf", "embedding_rf", "hybrid_rf", "mattertune"],
        "mattertune": {"enabled": False, "adapter": None},
    }

"""Orchestrate cached relax, feature extraction and method readiness."""

from scientific_layer.surrogate_models.build_surrogate_feature_table import build_surrogate_feature_table
from scientific_layer.surrogate_models.compare_surrogate_methods import compare_surrogate_methods
from scientific_layer.surrogate_models.create_feature_registry import create_feature_registry


def run_surrogate_experiment(dataset: dict, *, config: dict, cache_directory, relax_backend, extra_features=None, family_checker=None) -> dict:
    registry = create_feature_registry(extra_features)
    table = build_surrogate_feature_table(dataset, config=config, registry=registry, relax_backend=relax_backend, cache_directory=cache_directory, family_checker=family_checker)
    return {"feature_table": table, "comparison": compare_surrogate_methods(table, config=config), "feature_definitions": {name: {key: value for key, value in item.items() if key != "function"} for name, item in registry.items()}}

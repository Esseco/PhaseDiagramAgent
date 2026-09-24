"""Compare configured surrogate methods on one pool, split and target."""

from experiments.branch_surrogate.evaluate_surrogate_predictions import evaluate_surrogate_predictions
from experiments.branch_surrogate.fit_surrogate_method import fit_surrogate_method
from experiments.branch_surrogate.predict_surrogate_method import predict_surrogate_method
from experiments.branch_surrogate.prepare_surrogate_method_data import prepare_surrogate_method_data
from experiments.branch_surrogate.recommend_surrogate_config import recommend_surrogate_config
from experiments.branch_surrogate.replay_sequential_selection import replay_sequential_selection
from experiments.branch_surrogate.assess_surrogate_value import assess_surrogate_value


def compare_branch_surrogates(dataset: dict, feature_table: dict | None, *, surrogate_config: dict, comparison_config: dict, hull_metric=None) -> dict:
    prepared = prepare_surrogate_method_data(dataset, feature_table, config=comparison_config)
    train = [row for row in prepared["rows"] if row.get("split") == "train" and row.get("target") is not None]
    validation = [row for row in prepared["rows"] if row.get("split") == "validation" and row.get("target") is not None]
    test = [row for row in prepared["rows"] if row.get("split") == "test" and row.get("target") is not None]
    methods = {}
    for method in comparison_config.get("methods", []):
        fitted = fit_surrogate_method(method, train, surrogate_config=surrogate_config)
        if fitted.get("status") != "completed":
            methods[method] = {"status": "skipped", "reason": fitted.get("reason"), "validation": {"status": "skipped"}, "test": {"status": "skipped"}, "replay": {"status": "skipped"}}
            continue
        validation_pred = predict_surrogate_method(fitted, validation, seed=comparison_config.get("random_seed", 0))
        test_pred = predict_surrogate_method(fitted, test, seed=comparison_config.get("random_seed", 0))
        feature_values = [row.get("feature_cost") for row in prepared["rows"]]
        acquisition = (sum(feature_values) if all(isinstance(value, (int, float)) for value in feature_values) else None) if method in {"manual_rf", "embedding_rf", "hybrid_rf", "mattertune"} else 0.0
        methods[method] = {"status": "completed", "feature_selection": fitted.get("feature_selection"), "validation": evaluate_surrogate_predictions(validation_pred, validation, low_energy_fraction=comparison_config.get("low_energy_fraction", .1)), "test": evaluate_surrogate_predictions(test_pred, test, low_energy_fraction=comparison_config.get("low_energy_fraction", .1)), "cost": {"feature_acquisition": acquisition, "training": fitted.get("training_cost"), "validation_prediction": validation_pred.get("prediction_cost"), "test_prediction": test_pred.get("prediction_cost"), "units_note": "wall-clock training/prediction time is reported separately from ledger search cost"}, "replay": replay_sequential_selection(prepared["rows"], method=method, surrogate_config=surrogate_config, comparison_config=comparison_config, hull_metric=hull_metric)}
    primary = (comparison_config.get("recommendation") or {}).get("primary_metric", "important_miss_rate")
    report = {"status": "completed", "dataset_task": prepared.get("task"), "split_manifest": prepared.get("split_manifest"), "candidate_count": len(prepared["rows"]), "methods": methods, "surrogate_value_assessment": assess_surrogate_value(methods, primary_metric=primary), "bohb_assessment": {"status": "not_evaluated", "reason": "BOHB is a separate resource-allocation experiment"}, "production_policy_changed": False}
    report["recommendation"] = recommend_surrogate_config(report, config=comparison_config)
    return report

"""Validate a candidate MLIP against user-confirmed anomaly limits."""

from __future__ import annotations

from typing import Any, Callable


REQUIRED_LIMITS = (
    "max_energy_mae",
    "max_critical_failure_fraction",
    "max_near_hull_ranking_reversals",
)


def validate_mlip(
    old_model: dict[str, Any],
    new_model: dict[str, Any],
    validation_data: list[dict[str, Any]],
    *,
    evaluator: Callable[..., dict[str, float]] | None = None,
    criteria: dict[str, float] | None = None,
    validation_data_version: str,
) -> dict[str, Any]:
    """Validation never activates a model and does not require monotonic MAE."""
    limits = dict(criteria or {})
    missing_limits = [name for name in REQUIRED_LIMITS if limits.get(name) is None]
    base = {
        "old_model_version": old_model.get("version"),
        "new_model_version": new_model.get("version"),
        "training_data_version": new_model.get("dataset_version"),
        "validation_data_version": validation_data_version,
        "activate_new_model": False,
    }
    if missing_limits:
        return {
            **base,
            "status": "needs_user_confirmation",
            "passed": False,
            "recommend_activation": False,
            "missing_criteria": missing_limits,
            "criteria": limits,
            "error": None,
        }
    if evaluator is None:
        return {
            **base,
            "status": "not_configured",
            "passed": False,
            "recommend_activation": False,
            "error": "validation evaluator 未配置",
        }
    try:
        old = evaluator(model=old_model, data=validation_data)
        new = evaluator(model=new_model, data=validation_data)
        required_metrics = [
            "energy_mae",
            "critical_failure_fraction",
            "near_hull_ranking_reversals",
        ]
        missing_metrics = [name for name in required_metrics if new.get(name) is None]
        if limits.get("max_force_rmse") is not None and new.get("force_rmse") is None:
            missing_metrics.append("force_rmse")
        if missing_metrics:
            return {
                **base,
                "status": "insufficient_validation_evidence",
                "passed": False,
                "recommend_activation": False,
                "old_metrics": old,
                "new_metrics": new,
                "missing_metrics": missing_metrics,
                "criteria": limits,
                "error": None,
            }
        checks = {
            "energy_mae_absolute": float(new["energy_mae"]) <= float(limits["max_energy_mae"]),
            "critical_failures": float(new["critical_failure_fraction"])
            <= float(limits["max_critical_failure_fraction"]),
            "near_hull_ranking": int(new["near_hull_ranking_reversals"])
            <= int(limits["max_near_hull_ranking_reversals"]),
        }
        if limits.get("max_force_rmse") is not None:
            checks["force_rmse_absolute"] = float(new["force_rmse"]) <= float(
                limits["max_force_rmse"]
            )
        # Optional user-confirmed change limits are anomaly guards, never implicit monotonicity.
        if limits.get("max_energy_mae_increase") is not None:
            checks["energy_mae_change"] = float(new["energy_mae"]) - float(
                old["energy_mae"]
            ) <= float(limits["max_energy_mae_increase"])
        if limits.get("max_force_rmse_increase") is not None:
            checks["force_rmse_change"] = float(new["force_rmse"]) - float(
                old["force_rmse"]
            ) <= float(limits["max_force_rmse_increase"])
        passed = all(checks.values())
        gain = float(old["energy_mae"]) - float(new["energy_mae"])
        threshold = float(limits.get("minimum_distinguishable_energy_mae_gain", 0.0) or 0.0)
        benefit = "improved" if gain > threshold else "no_distinguishable_benefit"
        anomalies = [name for name, ok in checks.items() if not ok]
        return {
            **base,
            "status": "completed",
            "passed": passed,
            "recommend_activation": passed,
            "old_metrics": old,
            "new_metrics": new,
            "checks": checks,
            "anomalies": anomalies,
            "benefit_status": benefit,
            "criteria": limits,
            "error": None,
        }
    except Exception as error:
        return {
            **base,
            "status": "failed",
            "passed": False,
            "recommend_activation": False,
            "error": f"{type(error).__name__}: {error}",
        }

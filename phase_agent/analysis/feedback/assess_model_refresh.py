"""Pause on user-defined post-activation anomalies; do not demand monotonic MAE."""

from copy import deepcopy


def assess_model_refresh(state, refresh_summary, *, limits=None):
    settings = dict(limits or {})
    missing_limits = [
        name
        for name in ("max_failure_fraction", "max_near_hull_ranking_reversals")
        if settings.get(name) is None
    ]
    current = deepcopy(state)
    if missing_limits:
        return _pause(
            current,
            refresh_summary,
            "needs_user_confirmation",
            {},
            {"missing_limits": missing_limits},
        )
    missing_metrics = [
        name
        for name in ("failure_fraction", "near_hull_ranking_reversals")
        if refresh_summary.get(name) is None
    ]
    if (
        settings.get("max_energy_mae_ev_per_atom") is not None
        and refresh_summary.get("energy_mae_ev_per_atom") is None
    ):
        missing_metrics.append("energy_mae_ev_per_atom")
    if missing_metrics:
        return _pause(
            current,
            refresh_summary,
            "insufficient_refresh_evidence",
            {},
            {"missing_metrics": missing_metrics},
        )
    checks = {
        "failure_fraction": float(refresh_summary["failure_fraction"])
        <= float(settings["max_failure_fraction"]),
        "near_hull_ranking": int(refresh_summary["near_hull_ranking_reversals"])
        <= int(settings["max_near_hull_ranking_reversals"]),
    }
    if settings.get("max_energy_mae_ev_per_atom") is not None:
        checks["energy_mae"] = float(refresh_summary["energy_mae_ev_per_atom"]) <= float(
            settings["max_energy_mae_ev_per_atom"]
        )
    if not all(checks.values()):
        return _pause(current, refresh_summary, "paused_anomaly", checks, {})
    current.setdefault("model_refresh_assessments", []).append(
        {
            "model_version": current.get("active_model_version"),
            "checks": checks,
            "summary": deepcopy(refresh_summary),
            "status": "accepted",
        }
    )
    return {"status": "accepted", "state": current, "checks": checks, "rollback_recommended": False}


def _pause(current, summary, status, checks, extra):
    current["run_status"] = "paused"
    current.setdefault("pause_reasons", []).append("post_activation_refresh_" + status)
    recommendation = {
        "from_model_version": current.get("active_model_version"),
        "checks": checks,
        "summary": deepcopy(summary),
        **extra,
    }
    current["rollback_recommended"] = recommendation
    current.setdefault("model_refresh_assessments", []).append(
        {
            "model_version": current.get("active_model_version"),
            "checks": checks,
            "summary": deepcopy(summary),
            "status": status,
            **extra,
        }
    )
    return {
        "status": status,
        "state": current,
        "checks": checks,
        "rollback_recommended": True,
        **extra,
    }

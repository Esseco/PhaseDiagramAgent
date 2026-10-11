"""Evaluate only user-confirmed numerical convergence rules."""

from __future__ import annotations

from typing import Any
from math import isfinite
from phase_agent.analysis.convergence.single_point_validation import single_point_validation
from phase_agent.tools.state.task_waiting import awaiting_task_result


REQUIRED_RULES = (
    "hull_change_tolerance",
    "stable_model_update_epochs",
    "final_energy_mae_tolerance",
    "minimum_recent_dft_checks",
)


def check_global_convergence(
    state: dict[str, Any], *, rules: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Keep pending work, missing evidence, budget exhaustion and acceptance distinct."""
    config = _normalize_rules(rules or {})
    missing_rules = [name for name in REQUIRED_RULES if config.get(name) is None]
    if missing_rules:
        return {
            "status": "needs_confirmed_convergence_rules",
            "converged": False,
            "criteria_satisfied": False,
            "reason": "confirmed_thresholds_missing",
            "missing_rules": missing_rules,
            "config_version": state.get("confirmed_config_version"),
        }
    required = int(config["stable_model_update_epochs"])
    minimum_checks = int(config["minimum_recent_dft_checks"])
    invalid_limits = any(
        config.get(name) is not None and (not _number(config[name]) or config[name] < 0)
        for name in (
            "hull_change_tolerance",
            "final_energy_mae_tolerance",
            "final_energy_rmse_tolerance",
            "final_force_mae_tolerance",
            "final_force_rmse_tolerance",
            "phase_competition_gap_ev_per_atom",
        )
    )
    if required <= 0 or minimum_checks < 0 or invalid_limits:
        return {
            "status": "needs_confirmed_convergence_rules",
            "converged": False,
            "criteria_satisfied": False,
            "reason": "confirmed_thresholds_invalid",
        }
    active = [item for item in state.get("tasks", []) if awaiting_task_result(item)]
    if active:
        return {
            "status": "tasks_awaiting_results",
            "converged": False,
            "criteria_satisfied": False,
            "reason": "tasks_in_progress_or_unknown",
            "active_task_ids": [row.get("task_id") for row in active],
        }
    epochs = _unique_model_epochs(state.get("model_update_epochs") or [])
    tail = epochs[-required:]
    enough_epochs = len(tail) == required
    hull_values_known = enough_epochs and all(_number(row.get("hull_change")) for row in tail)
    ground_states_known = enough_epochs and all(
        row.get("ground_state_unchanged") is not None for row in tail
    )
    errors = (
        list(state.get("final_frame_dft_errors") or [])[-minimum_checks:] if minimum_checks else []
    )
    errors_known = len(errors) == minimum_checks and all(
        _number(row.get("error_ev_per_atom")) for row in errors
    )
    recent_mae = (
        sum(abs(float(row["error_ev_per_atom"])) for row in errors) / len(errors)
        if errors_known and errors
        else 0.0
        if errors_known
        else None
    )
    version = state.get("active_model_version") or (tail[-1].get("model_version") if tail else None)
    validation = single_point_validation(
        state,
        model_version=version,
        minimum_checks=minimum_checks,
        phase_gap=config.get("phase_competition_gap_ev_per_atom"),
    )
    metrics = validation["metrics"]
    if (
        state.get("dft_dataset_records")
        or config.get("phase_competition_gap_ev_per_atom") is not None
    ):
        errors_known = metrics["enough_checks"]
        recent_mae = metrics["energy_per_atom"]["mae"]
    checks = {
        "model_update_epoch_count": enough_epochs,
        "hull_stable": hull_values_known
        and all(
            abs(float(row["hull_change"])) <= float(config["hull_change_tolerance"]) for row in tail
        ),
        "ground_states_stable": ground_states_known
        and all(row.get("ground_state_unchanged") is True for row in tail),
        "final_frame_dft_calibrated": errors_known
        and recent_mae is not None
        and recent_mae <= float(config["final_energy_mae_tolerance"]),
    }
    if config.get("phase_competition_gap_ev_per_atom") is not None:
        checks["competing_phases_agree"] = validation["phase_competition"]["passed"]
    for rule, metric, statistic in (
        ("final_energy_rmse_tolerance", "energy_per_atom", "rmse"),
        ("final_force_mae_tolerance", "forces", "mae"),
        ("final_force_rmse_tolerance", "forces", "rmse"),
    ):
        if config.get(rule) is not None:
            value = metrics[metric][statistic]
            checks[rule] = errors_known and value is not None and value <= float(config[rule])
    missing = []
    if not enough_epochs:
        missing.append("consecutive_unique_model_update_epochs")
    if enough_epochs and not hull_values_known:
        missing.append("model_epoch_hull_change")
    if enough_epochs and not ground_states_known:
        missing.append("model_epoch_ground_state_stability")
    if not errors_known:
        missing.append("final_frame_dft_error")
    if validation["phase_competition"]["status"] == "missing_evidence":
        missing.append("competing_phase_same_frame_dft")
    coverage = state.get("coverage") or state.get("coverage_summary")
    coverage_risk = _coverage_risk(coverage)
    coverage_risk["unreturned_dft_wait_waived_task_ids"] = [
        t.get("task_id")
        for t in state.get("tasks", [])
        if t.get("recovery_wait_waived") is True
        and t.get("status") in {"pending", "running", "submitted", "unknown"}
    ]
    if all(checks.values()):
        accepted = state.get("user_accepted_convergence") is True
        return {
            "status": "finished" if accepted else "numerical_criteria_satisfied",
            "converged": accepted,
            "criteria_satisfied": True,
            "requires_user_acceptance": not accepted,
            "reason": "user_accepted_numerical_convergence"
            if accepted
            else "numerical_criteria_met_coverage_review_required",
            "checks": checks,
            "coverage_evidence": coverage,
            "coverage_risk": coverage_risk,
            "coverage_is_hard_condition": False,
            "recent_final_frame_mae_ev_per_atom": recent_mae,
            "single_point_validation": validation,
            "model_update_epochs_counted": [row.get("model_version") for row in tail],
            "config_version": state.get("confirmed_config_version"),
        }
    if state.get("budget_remaining") is not None and state["budget_remaining"] <= 0:
        status, reason = "budget_exhausted", "budget_exhausted_before_convergence"
    elif missing:
        status, reason = "insufficient_evidence", "required_results_missing"
    else:
        status, reason = "continue", "numerical_checks_not_met"
    return {
        "status": status,
        "converged": False,
        "criteria_satisfied": False,
        "reason": reason,
        "checks": checks,
        "missing_evidence": missing,
        "coverage_evidence": coverage,
        "coverage_risk": coverage_risk,
        "coverage_is_hard_condition": False,
        "recent_final_frame_mae_ev_per_atom": recent_mae,
        "single_point_validation": validation,
        "model_update_epochs_counted": [row.get("model_version") for row in tail],
    }


def _normalize_rules(rules):
    config = dict(rules)
    for old, new in (
        ("stable_iterations", "stable_model_update_epochs"),
        ("final_frame_error_tolerance", "final_energy_mae_tolerance"),
        ("hull_tolerance", "hull_change_tolerance"),
    ):
        if new not in config and old in config:
            config[new] = config[old]
    return config


def _unique_model_epochs(rows):
    ordered, positions = [], {}
    for row in rows:
        version = row.get("model_version") if isinstance(row, dict) else None
        if not version:
            continue
        if version in positions:
            ordered[positions[version]] = row
        else:
            positions[version] = len(ordered)
            ordered.append(row)
    return ordered


def _coverage_risk(coverage):
    if not coverage:
        return {"status": "unknown", "message": "coverage evidence missing; user review required"}
    fraction = coverage.get("fraction") if isinstance(coverage, dict) else None
    if _number(fraction) and float(fraction) < 1.0:
        return {
            "status": "incomplete",
            "fraction": float(fraction),
            "message": "phase × SOC coverage is incomplete but is not an automatic convergence gate",
        }
    return {"status": "reported", "message": "coverage is evidence for final user review"}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)

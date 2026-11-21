"""Validate a frozen retry scope for MC failures caused by an overly strict threshold."""

import math


def validate_mc_threshold_retry(state, target_ids):
    if (
        not isinstance(target_ids, list)
        or not target_ids
        or any(not isinstance(x, str) or not x for x in target_ids)
        or len(set(target_ids)) != len(target_ids)
    ):
        raise ValueError("invalid_mc_threshold_retry_targets")
    tasks = {}
    for row in state.get("tasks") or []:
        tasks.setdefault(row.get("task_id"), []).append(row)
    processed = set(state.get("processed_task_ids") or [])
    model = state.get("active_model_version")
    scopes = set()
    selected = []
    for target in target_ids:
        matches = tasks.get(target) or []
        if len(matches) != 1 or target not in processed or not model:
            raise ValueError("mc_threshold_retry_unverified_target")
        task = matches[0]
        output = task.get("outputs") or {}
        monitor = output.get("extrapolation_monitor") or {}
        failures = monitor.get("failures")
        if (
            task.get("stage") != "deep_search"
            or task.get("status") != "failed"
            or task.get("model_version") != model
            or not task.get("task_key")
            or not task.get("parent_decision_id")
            or task.get("recovery_cancelled")
            or output.get("error") != "initial_relaxation_extrapolation_failure"
            or monitor.get("extrapolation_failure") is not True
            or not isinstance(failures, list)
            or not failures
        ):
            raise ValueError("mc_threshold_retry_scope_mismatch")
        if any(
            not isinstance(row, dict)
            or isinstance(row.get("score"), bool)
            or not isinstance(row.get("score"), (int, float))
            or not math.isfinite(row["score"])
            or not 1 < row["score"] <= 3
            for row in failures
        ):
            raise ValueError("mc_threshold_retry_score_outside_scope")
        if monitor.get("feature_difference_threshold", 1.0) != 1.0:
            raise ValueError("mc_threshold_retry_original_threshold_mismatch")
        scopes.add(task["parent_decision_id"])
        selected.append(task)
    if len(scopes) != 1:
        raise ValueError("mc_threshold_retry_mixed_actions")
    return selected

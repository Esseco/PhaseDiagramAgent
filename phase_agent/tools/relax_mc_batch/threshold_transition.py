"""Read-only reassessment of verified legacy Relax–MC monitor observations."""

import math
from copy import deepcopy


def reassess_legacy_monitor(monitor, *, original_threshold=1.0, new_threshold=3.0):
    """Derive a new verdict without changing a legacy monitor or omitting jumps.

    This requires an original monitor that recorded all final/jump scores strictly
    above the original threshold. It is invalid to lower the threshold here.
    """
    if original_threshold != 1.0 or new_threshold != 3.0:
        raise ValueError("unsupported_transition_policy")
    if not isinstance(monitor, dict):
        raise ValueError("monitor_missing")
    value = monitor.get("final_feature_difference")

    def finite_score(x):
        return (
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
        )

    failures = monitor.get("failures")
    flag = monitor.get("extrapolation_failure")
    if not finite_score(value) or not isinstance(failures, list) or type(flag) is not bool:
        raise ValueError("monitor_evidence_incomplete")
    if bool(failures) is not flag:
        raise ValueError("monitor_flag_mismatch")
    final_events = []
    for event in failures:
        if not isinstance(event, dict) or event.get("trigger") not in {"final", "force_jump"}:
            raise ValueError("monitor_event_invalid")
        score = event.get("score")
        if not finite_score(score) or score <= original_threshold:
            raise ValueError("legacy_threshold_inconsistent")
        if event["trigger"] == "final":
            final_events.append(score)
    if (
        len(final_events) > 1
        or (value > original_threshold and final_events != [value])
        or (value <= original_threshold and final_events)
    ):
        raise ValueError("monitor_final_event_mismatch")
    selected = [deepcopy(e) for e in failures if e["score"] > new_threshold]
    return {
        "original_threshold": original_threshold,
        "effective_threshold": new_threshold,
        "original_failure": flag,
        "effective_failure": bool(selected),
        "final_feature_difference": value,
        "effective_failure_events": selected,
    }


def reassess_verified_task(task, raw_result, *, original_threshold=1.0, new_threshold=3.0):
    """Reassess a caller-verified Relax result; do not alter the signed artifact."""
    for field in (
        "task_id",
        "task_key",
        "stage",
        "structure_id",
        "model_version",
        "config_version",
    ):
        if not task.get(field) or task[field] != raw_result.get(field):
            raise ValueError("task_binding_mismatch:" + field)
    if task["stage"] != "relax_and_feature" or raw_result.get("status") != "completed":
        raise ValueError("completed_relax_required")
    output = raw_result.get("outputs") or {}
    if output.get("optimizer_converged") is not True or raw_result.get("converged") is not True:
        raise ValueError("relax_not_converged")
    monitor = output.get("extrapolation_monitor")
    review = reassess_legacy_monitor(
        monitor, original_threshold=original_threshold, new_threshold=new_threshold
    )
    identity = output.get("relaxation_observation") or {}
    if (
        identity.get("identity_scheme") != "periodic_input_v1"
        or not identity.get("structure_id")
        or identity.get("monitor_complete") is not True
        or identity.get("extrapolation_failure") is not review["original_failure"]
    ):
        raise ValueError("relax_monitor_identity_invalid")
    return {
        "task_id": task["task_id"],
        "structure_id": task["structure_id"],
        "monitor_structure_id": identity["structure_id"],
        "model_version": task["model_version"],
        "config_version": task["config_version"],
        **review,
    }

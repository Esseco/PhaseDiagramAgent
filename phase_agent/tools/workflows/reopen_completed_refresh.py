"""Build a guarded state transition for an explicitly superseded refresh."""

from copy import deepcopy


def reopen_state(state, preview, *, expected_model, expected_plan, expected_hull, expected_preview):
    """Prepare a new, unapproved refresh while preserving the completed attempt."""
    refresh = state.get("model_refresh") or {}
    cycle = state.get("extrapolation_recovery_cycle") or {}
    hull = (state.get("phase_diagrams") or {}).get("mlip") or {}
    plan = preview.get("plan") or {}
    if (preview.get("status") != "preview_only" or preview.get("submitted") is not False
        or preview.get("prior_plan_checksum") != expected_plan
        or preview.get("prior_hull_version") != expected_hull
        or plan.get("checksum") != expected_preview
        or expected_preview == expected_plan
        or plan.get("new_model_version") != expected_model
        or (cycle and plan.get("baseline_hull_version") != cycle.get("baseline_hull_version"))
        or (not cycle and plan.get("reopened_from_plan") != expected_plan)
        or not plan.get("hd_refresh_reference")):
        raise ValueError("reopen_preview_binding_invalid")
    if (refresh.get("status") != "completed"
        or refresh.get("new_model_version") != expected_model
        or state.get("active_model_version") != expected_model
        or (refresh.get("plan") or {}).get("checksum") != expected_plan
        or hull.get("version") != expected_hull
        or (cycle and (
            cycle.get("status") != "completed"
            or cycle.get("replacement_model_version") != expected_model
            or cycle.get("replacement_hull_version") != expected_hull
        ))
        or state.get("pending_execution_policies")):
        raise ValueError("reopen_context_changed")
    baseline = (cycle.get("baseline_diagram") if cycle else refresh.get("old_diagram")) or {}
    baseline_model = cycle.get("baseline_model_version") if cycle else refresh.get("old_model_version")
    if (baseline.get("status") != "completed"
        or not baseline.get("version")
        or (cycle and baseline.get("version") != cycle.get("baseline_hull_version"))
        or not baseline_model or baseline.get("model_version") != baseline_model):
        raise ValueError("reopen_baseline_invalid")
    updated = dict(state)
    history = list(state.get("completed_refresh_replans") or [])
    history.append({"prior_refresh": deepcopy(refresh), "prior_hull": deepcopy(hull),
                    "prior_recovery_cycle": deepcopy(cycle), "new_preview_checksum": expected_preview})
    updated["completed_refresh_replans"] = history
    updated["model_refresh"] = {
        "status": "approval_required", "old_model_version": baseline_model,
        "new_model_version": expected_model, "wave": 0, "old_diagram": deepcopy(baseline),
        "reopened_from_plan": expected_plan, "reopened_from_hull": expected_hull,
        "reopened_preview_checksum": expected_preview,
    }
    if cycle:
        updated["extrapolation_recovery_cycle"] = {
            **cycle, "status": "awaiting_new_model_refresh", "replacement_hull_version": None,
        }
    diagrams = dict(state.get("phase_diagrams") or {})
    diagrams["mlip"] = {"status": "not_ready", "model_version": expected_model,
                        "reason": "completed_refresh_superseded", "entries": []}
    updated["phase_diagrams"] = diagrams
    return updated

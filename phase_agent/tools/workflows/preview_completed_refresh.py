"""Build a non-mutating preview for an explicitly selected completed refresh."""

from copy import deepcopy


def preview_completed_refresh(state, manager, config, *, expected_model, expected_plan, expected_hull):
    """Run the normal refresh preview on a temporary recovery state view."""
    refresh = state.get("model_refresh") or {}
    cycle = state.get("extrapolation_recovery_cycle") or {}
    hull = (state.get("phase_diagrams") or {}).get("mlip") or {}
    old_plan = refresh.get("plan") or {}
    if (state.get("active_model_version") != expected_model
        or refresh.get("new_model_version") != expected_model
        or refresh.get("status") != "completed"
        or old_plan.get("checksum") != expected_plan
        or hull.get("version") != expected_hull):
        raise ValueError("completed_refresh_binding_changed")
    if state.get("pending_execution_policies"):
        raise ValueError("completed_refresh_pending_approval")
    if cycle and (cycle.get("policy_version") != "epoch_refresh_v1"
        or cycle.get("status") != "completed"
        or cycle.get("replacement_model_version") != expected_model
        or cycle.get("replacement_hull_version") != expected_hull):
        raise ValueError("completed_refresh_recovery_binding_changed")
    baseline = (cycle.get("baseline_diagram") if cycle else refresh.get("old_diagram")) or {}
    baseline_model = cycle.get("baseline_model_version") if cycle else refresh.get("old_model_version")
    if (baseline.get("status") != "completed"
        or not baseline.get("version")
        or (cycle and baseline.get("version") != cycle.get("baseline_hull_version"))
        or not baseline_model or baseline.get("model_version") != baseline_model):
        raise ValueError("completed_refresh_baseline_invalid")
    ids = refresh.get("task_ids") or []
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("completed_refresh_task_scope_invalid")
    id_set = set(ids)
    rows = [t for t in state.get("tasks") or [] if t.get("task_id") in id_set]
    if len(rows) != len(ids) or any(t.get("status") not in {"completed", "failed", "cancelled"} for t in rows):
        raise ValueError("completed_refresh_tasks_unsettled")
    preview_state = dict(state)
    preview_state["model_refresh"] = {
        "status": "approval_required", "old_model_version": baseline_model,
        "new_model_version": expected_model, "wave": 0, "old_diagram": deepcopy(baseline),
    }
    if cycle:
        preview_state["extrapolation_recovery_cycle"] = {**cycle, "status": "awaiting_new_model_refresh"}
    else:
        preview_state["model_refresh"]["reopened_from_plan"] = expected_plan
        preview_state["completed_refresh_replans"] = [
            *(state.get("completed_refresh_replans") or []), {"prior_refresh": refresh}
        ]
    from phase_agent.tools.workflows.model_refresh_state import refresh_preview

    plan = refresh_preview(preview_state, manager, config)
    return {"status": "preview_only", "submitted": False, "prior_plan_checksum": expected_plan,
            "prior_hull_version": expected_hull, "plan": plan}

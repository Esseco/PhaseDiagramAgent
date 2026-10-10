"""Activate only a passed model and trigger version-scoped refresh markers."""

from copy import deepcopy

from phase_agent.persistence.models.mark_model_dependent_results_stale import (
    mark_model_dependent_results_stale,
)


def activate_validated_model(
    state: dict, new_model: dict, validation: dict, *, approval_reason=None
) -> dict:
    if validation.get("status") != "completed" or validation.get("passed") is not True:
        return {"status": "rejected", "state": state, "reason": "new_model_not_validated"}
    old = state.get("active_model_version")
    new = new_model.get("version")
    if not new:
        return {"status": "rejected", "state": state, "reason": "new_model_version_missing"}
    if not isinstance(approval_reason, str) or not approval_reason.strip():
        return {
            "status": "awaiting_user_approval",
            "state": state,
            "reason": "explicit_activation_approval_reason_required",
        }
    refreshed = mark_model_dependent_results_stale(
        state, old_model_version=old, new_model_version=new
    )
    updated = refreshed["state"]
    from phase_agent.persistence.models.require_structure_refresh import require_structure_refresh

    require_structure_refresh(updated, old, new)
    registry = updated.setdefault("model_registry", {})
    if old and old not in registry:
        registry[old] = {
            "model": deepcopy(state.get("active_model") or {"version": old}),
            "status": "historical_active",
        }
    registry[new] = {
        "model": deepcopy(new_model),
        "status": "active",
        "validation": deepcopy(validation),
        "approval_reason": approval_reason.strip(),
    }
    updated["active_model"] = deepcopy(new_model)
    updated.setdefault("model_activation_history", []).append(
        {
            "old_model_version": old,
            "new_model_version": new,
            "dft_data_version": new_model.get("dataset_version"),
            "validation_data_version": validation.get("validation_data_version"),
            "validation_summary": deepcopy(validation),
            "approval_reason": approval_reason.strip(),
        }
    )
    return {
        "status": "activated",
        "state": updated,
        "marked": refreshed["marked"],
        "validation_data_version": validation.get("validation_data_version"),
    }

"""Audited rollback to a previously activated model without deleting history."""

from copy import deepcopy
from data_layer.models.mark_model_dependent_results_stale import mark_model_dependent_results_stale


def rollback_model(state, *, target_version, user_approved, reason):
    current = deepcopy(state)
    if not user_approved:
        return {"status": "awaiting_approval", "state": current}
    registry = current.get("model_registry") or {}
    if target_version not in registry:
        return {"status": "rejected", "reason": "rollback_target_unknown", "state": current}
    old = current.get("active_model_version")
    refreshed = mark_model_dependent_results_stale(current, old_model_version=old,
                                                    new_model_version=target_version)
    current = refreshed["state"]
    current["active_model"] = deepcopy(registry[target_version]["model"])
    if current.get("model_refresh"):
        current.setdefault("model_refresh_history", []).append({**current.pop("model_refresh"), "status": "cancelled_by_rollback"})
    current["run_status"] = "paused"
    current.setdefault("model_rollback_history", []).append({
        "from_model_version": old, "to_model_version": target_version,
        "reason": str(reason), "user_approved": True,
        "preserved_candidate_models": sorted((current.get("candidate_models") or {}).keys()),
    })
    return {"status": "rolled_back_paused", "state": current, "marked": refreshed["marked"]}

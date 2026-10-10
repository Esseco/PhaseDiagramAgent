"""Preserve the prior hull and mark the new epoch as requiring approval."""

from copy import deepcopy


def require_structure_refresh(state, old_version, new_version):
    if old_version == new_version:
        return
    rounds = state.setdefault("upload_layout", {}).setdefault("model_rounds", {})
    if old_version and old_version not in rounds:
        rounds[old_version] = max(rounds.values(), default=0) + 1
    if new_version not in rounds:
        rounds[new_version] = max(rounds.values(), default=0) + 1
    diagrams = state.get("phase_diagrams") or {}
    prior = (
        diagrams.get("mlip")
        if isinstance(diagrams, dict)
        else next(
            (row for row in reversed(diagrams) if row.get("model_version") == old_version), {}
        )
    )
    state["model_refresh"] = {
        "status": "approval_required",
        "old_model_version": old_version,
        "new_model_version": new_version,
        "wave": 0,
        "old_diagram": deepcopy(
            (state.get("phase_diagrams_by_model") or {}).get(old_version) or prior or {}
        ),
    }

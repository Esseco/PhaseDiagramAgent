"""Preserve the prior hull and mark the new epoch as requiring approval."""

from copy import deepcopy


def require_structure_refresh(state, old_version, new_version):
    if old_version == new_version:
        return
    recovery = state.get("extrapolation_recovery_cycle") or {}
    from phase_agent.evidence.model_epoch import model_epoch
    parent_epoch = model_epoch(state, old_version)
    external_update = (recovery.get("status") == "awaiting_reviewed_dft_labels"
                       and recovery.get("policy_version") == "epoch_refresh_v1")
    if external_update:
        parent_epoch = parent_epoch or recovery.get("origin_epoch")
        if not parent_epoch:
            raise ValueError("scientific_epoch_unknown")
    scientific = (state.get("upload_layout") or {}).get("scientific_epochs") or {}
    scientific_number = None
    if parent_epoch and (external_update or scientific):
        if not parent_epoch.startswith("epoch") or not parent_epoch[5:].isdigit():
            raise ValueError("scientific_epoch_invalid")
        scientific_number = int(parent_epoch[5:]) + 1
        if new_version in scientific and scientific[new_version] != scientific_number:
            raise ValueError("scientific_epoch_conflict")
    if recovery.get("status") == "awaiting_reviewed_dft_labels":
        from phase_agent.evidence.extrapolation_recovery import recovery_label_facts

        if recovery_label_facts(state)["status"] != "complete":
            raise ValueError("extrapolation_recovery_dft_labels_incomplete")
        if old_version != recovery.get("invalid_model_version"):
            raise ValueError("extrapolation_recovery_training_parent_mismatch")
        baseline = recovery.get("baseline_diagram") or {}
        if baseline.get("version") != recovery.get("baseline_hull_version"):
            raise ValueError("extrapolation_recovery_baseline_invalid")
        recovery["training_parent_model_version"] = old_version
        recovery["replacement_model_version"] = new_version
        recovery["status"] = "awaiting_new_model_refresh"
        old_version = recovery["baseline_model_version"]
    # A new epoch carries forward each structure's most recent valid MLIP result.
    # Preserve provenance; never rewrite the previously approved refresh plan.
    state["mlip_energy_policy"] = "latest_valid_per_structure_v1"
    rounds = state.setdefault("upload_layout", {}).setdefault("model_rounds", {})
    if old_version and old_version not in rounds:
        rounds[old_version] = max(rounds.values(), default=0) + 1
    if new_version not in rounds:
        rounds[new_version] = max(rounds.values(), default=0) + 1
    layout = state["upload_layout"]
    if scientific_number is not None:
        layout.setdefault("scientific_epochs", {})[new_version] = scientific_number
    if external_update:
        layout.setdefault("extrapolation_rounds", {})[new_version] = recovery.get("extrapolation_round", 1) + 1
    state.setdefault("mlip_feedback_baselines", {}).pop(new_version, None)
    diagrams = state.get("phase_diagrams") or {}
    prior = (
        diagrams.get("mlip")
        if isinstance(diagrams, dict)
        else next(
            (row for row in reversed(diagrams) if row.get("model_version") == old_version), {}
        )
    )
    state["model_refresh"] = {
        **({"extrapolation_round": layout["extrapolation_rounds"][new_version]} if external_update else {}),
        "status": "approval_required",
        "old_model_version": old_version,
        "new_model_version": new_version,
        "wave": 0,
        "old_diagram": deepcopy(
            recovery.get("baseline_diagram") if recovery.get("status") == "awaiting_new_model_refresh"
            else (state.get("phase_diagrams_by_model") or {}).get(old_version) or prior or {}
        ),
    }

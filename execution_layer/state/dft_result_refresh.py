"""Validate metadata-only DFT re-extraction without changing calculation/accounting facts."""
from copy import deepcopy
import json

DFT_STAGES = {"dft_single_point", "dft_relax"}
MAGNETIC_KEYS = ("magnetic_moments", "magnetic_check")


def magnetic_evidence(row):
    outputs = row.get("outputs") or row
    raw = outputs.get("magnetic_moments") or outputs.get("magnetic_check") or {}
    return {key: raw.get(key) for key in ("elements", "moments", "noncollinear", "final_frame_index", "unit")}


def validated_magnetic_refresh(prior, incoming):
    """Return unchanged/refresh/rejected and a restricted merge; never replace E/F labels."""
    if prior.get("stage") not in DFT_STAGES or incoming.get("stage") not in DFT_STAGES:
        return {"status": "unchanged"}
    for key in ("task_id", "task_key", "structure_id", "stage", "status", "model_version", "converged",
                "upload_operation_id", "search_group_index", "parent_relax_round"):
        if prior.get(key) is not None and incoming.get(key) != prior[key]:
            return {"status": "rejected", "reason": f"dft_refresh_{key}_mismatch"}
    old, new = prior.get("outputs") or {}, incoming.get("outputs") or {}
    for key in ("structure", "structure_checksum", "energy", "energy_unit", "training_energy", "forces", "forces_unit",
                "stress", "stress_unit", "stress_convention", "atom_count", "composition", "final_frame_index",
                "training_frame_index", "final_frame_valid", "training_frames", "training_ready"):
        if key in old and json.dumps(old[key], sort_keys=True) != json.dumps(new.get(key), sort_keys=True):
            return {"status": "rejected", "reason": f"dft_refresh_{key}_mismatch"}
    prediction_changed = old.get("remote_mlip_prediction") != new.get("remote_mlip_prediction")
    if prediction_changed and new.get("remote_mlip_prediction") is not None:
        previous = old.get("remote_mlip_prediction") or {}
        if previous.get("status") == "completed":
            return {"status": "rejected", "reason": "dft_refresh_completed_mlip_prediction_changed"}
        from execution_layer.remote.dft_mlip_pair import validate_prediction
        if new["remote_mlip_prediction"].get("status") == "completed":
            try:
                validate_prediction(incoming, new["remote_mlip_prediction"])
            except (ValueError, TypeError, KeyError) as error:
                return {"status": "rejected", "reason": f"dft_refresh_mlip_prediction_invalid:{error}"}
    if magnetic_evidence(prior) == magnetic_evidence(incoming) and not prediction_changed:
        return {"status": "unchanged"}
    if old.get("energy") is None or not (old.get("structure") or old.get("structure_checksum")):
        return {"status": "rejected", "reason": "dft_refresh_final_labels_unverifiable"}
    merged = deepcopy(prior)
    for key in (*MAGNETIC_KEYS, "remote_mlip_prediction", "mlip_result_file", "mlip_result_checksum"):
        if key in new:
            merged.setdefault("outputs", {})[key] = deepcopy(new[key])
    prior_spin = old.get("spin_state_check") or {}
    if prior_spin.get("status") in {"unknown", "rejected"} and incoming.get("checks_passed") is True:
        other_reasons = [r for r in prior.get("quality_rejection_reasons", []) if r != "dft_spin_standard_not_passed"]
        merged["checks_passed_before_spin"] = not other_reasons
    # Preserve local phase evidence/other checks and original cost settlement.
    return {"status": "refresh", "result": merged}

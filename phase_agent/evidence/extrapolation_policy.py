"""Bounded sampling policy for a model-epoch refresh."""

import math


def model_update_index(state, model_version):
    """Count actual activations; use recorded model order for legacy state."""
    parents, ambiguous = {}, set()
    for row in state.get("model_activation_history") or []:
        if not isinstance(row, dict):
            continue
        new, old = row.get("new_model_version"), row.get("old_model_version")
        if not isinstance(new, str) or not new or not isinstance(old, str) or not old:
            continue
        if new in parents and parents[new] != old:
            ambiguous.add(new)
        parents[new] = old
    if model_version in parents:
        current, seen, count = model_version, set(), 0
        while current in parents:
            if current in seen or current in ambiguous:
                return None
            seen.add(current)
            current = parents[current]
            count += 1
        rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
        initial = (state.get("confirmed_config") or {}).get("mlip") or {}
        initial_version = initial.get("version") or initial.get("name")
        if rounds.get(current) != 1 and initial_version != current:
            return None
        return count
    number = ((state.get("upload_layout") or {}).get("model_rounds") or {}).get(model_version)
    return number - 1 if type(number) is int and number >= 1 else None


def assess_policy(
    update_index, failure_fraction, attempt_count, valid_count, switch_to_perturbation, *, legacy=False
):
    """Decide from supplied evidence; calculation and result management remain external."""
    if (
        (legacy and (type(update_index) is not int or update_index < 0))
        or type(attempt_count) is not int
        or attempt_count < 0
        or type(valid_count) is not int
        or valid_count < 0
        or type(switch_to_perturbation) is not bool
        or isinstance(failure_fraction, bool)
        or not isinstance(failure_fraction, (int, float))
        or not math.isfinite(failure_fraction)
        or not 0 <= failure_fraction <= 1
    ):
        return {"status": "unknown", "route": None}
    if failure_fraction < 0.1:
        return {"status": "clear", "route": None}
    if valid_count >= 50:
        return {"status": "complete", "route": None}
    if attempt_count >= 5:
        return {"status": "attempt_limit", "route": None}
    route = (
        "perturbation"
        if not legacy or update_index < 2 or failure_fraction >= 0.2 or switch_to_perturbation
        else "extrapolation"
    )
    return {"status": "sampling_required", "route": route}


def recovery_lineage(previous, model_version, evidence_digest, initial_epoch):
    """Keep supplement attempts together while counting new model refreshes as rounds."""
    import hashlib

    previous = previous or {}
    same_model = previous.get("invalid_model_version") == model_version
    replacement = (
        previous.get("status") == "awaiting_new_model_refresh"
        and previous.get("replacement_model_version") == model_version
    )
    if previous.get("policy_version") == "epoch_refresh_v1" and (same_model or replacement):
        lineage = previous.get("recovery_lineage_id")
        if not lineage:
            lineage = hashlib.sha256(
                (
                    str(previous.get("invalid_model_version"))
                    + ":"
                    + str(previous.get("round_evidence_digest"))
                ).encode()
            ).hexdigest()
        return {
            "recovery_lineage_id": lineage,
            "extrapolation_round": previous.get("extrapolation_round", 1) + int(replacement),
            "origin_epoch": previous.get("origin_epoch") or initial_epoch,
        }
    lineage = hashlib.sha256((str(model_version) + ":" + str(evidence_digest)).encode()).hexdigest()
    return {"recovery_lineage_id": lineage, "extrapolation_round": 1, "origin_epoch": initial_epoch}

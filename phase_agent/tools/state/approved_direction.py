"""Bind a concrete action to the exact approved scientific direction."""

import hashlib
import json

TOOLS = {
    "supplement_dft": {"select_dft_candidates", "adjust_strategy"},
    "other": {"adjust_strategy"},
}


def approved_direction(state):
    rows = []
    for job in state.get("remote_finetune_jobs", {}).values():
        handoff = job.get("training_handoff") or {}
        proposal = handoff.get("direction_proposal") or {}
        if (
            not job.get("activated")
            and job.get("original_model_version") == state.get("active_model_version")
            and handoff.get("direction_status") == "approved"
            and proposal.get("direction") in TOOLS
        ):
            rows.append(proposal)
    if len(rows) > 1:
        raise ValueError("multiple_approved_scientific_directions")
    return rows[0] if rows else None


def direction_hash(proposal):
    return hashlib.sha256(
        json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def direction_action_errors(action, state):
    proposal = approved_direction(state)
    bound = action.get("_approved_direction_hash")
    if not proposal:
        return ["approved_direction_changed"] if bound else []
    errors = []
    if (action.get("tool") or action.get("action_type")) not in TOOLS[proposal["direction"]]:
        errors.append("action_outside_approved_direction")
    if bound != direction_hash(proposal):
        errors.append("action_not_bound_to_current_direction")
    return errors

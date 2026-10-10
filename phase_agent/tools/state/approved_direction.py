"""Bind a concrete action to the exact approved scientific direction."""

import hashlib
import json

TOOLS = {
    "supplement_dft": {"select_dft_candidates", "adjust_strategy"},
    "other": {"adjust_strategy"},
}


def action_direction(state, *, approved_only=False):
    rows = []
    for job in state.get("remote_finetune_jobs", {}).values():
        handoff = job.get("training_handoff") or {}
        proposal = handoff.get("direction_proposal") or {}
        if (
            not job.get("activated")
            and job.get("original_model_version") == state.get("active_model_version")
            and (
                handoff.get("direction_status") == "approved"
                or (
                    not approved_only
                    and handoff.get("stage") == "execution_plan_ready"
                    and handoff.get("direction_status") == "awaiting_approval"
                )
            )
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
    proposal = action_direction(state)
    bound = action.get("_approved_direction_hash")
    if not proposal:
        return ["approved_direction_changed"] if bound else []
    errors = []
    if (action.get("tool") or action.get("action_type")) not in TOOLS[proposal["direction"]]:
        errors.append("action_outside_approved_direction")
    if bound != direction_hash(proposal):
        errors.append("action_not_bound_to_current_direction")
    plan = proposal.get("plan")
    if (
        isinstance(plan, dict)
        and proposal.get("review_policy") == "post_training_single_approval_v3"
    ):
        for key in ("tool", "parameters", "target_ids", "budget"):
            if action.get(key) != plan.get(key):
                errors.append("saved_direction_plan_changed:" + key)
    return errors


def approved_direction(state):
    return action_direction(state, approved_only=True)


def record_action_direction(state, action, state_path, decision):
    """The concrete action approval records its direction in the same user turn."""
    proposal = action_direction(state)
    if not proposal or not action.get("_approved_direction_hash"):
        return
    errors = direction_action_errors(action, state)
    if errors:
        raise ValueError(";".join(errors))
    from phase_agent.graphs.direction_review_graph import direction_checkpoint

    # Evidence changes require a fresh proposal, even when a saved action still exists.
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    for job in state.get("remote_finetune_jobs", {}).values():
        handoff = job.get("training_handoff") or {}
        if handoff.get("direction_proposal") != proposal:
            continue
        if handoff.get("stage") not in {"execution_plan_ready", "execution_plan_approved"}:
            continue  # Historical separate approvals retain their existing contract.
        fresh = inspect_training_results(job)
        if not fresh or fresh.get("fingerprint") != proposal.get("training_fingerprint"):
            raise ValueError("training_evidence_changed")
        checkpoint = direction_checkpoint(state_path, proposal, decision)
        handoff["direction_status"] = checkpoint["status"]
        handoff["stage"] = (
            "execution_plan_approved" if decision == "approve" else "direction_rejected"
        )
        if decision == "reject":
            candidate = state.get("candidate_models", {}).get(
                proposal.get("candidate_model_version"), {}
            )
            review = candidate.get("agent_review") or {}
            review["followup_result"] = {"status": "rejected_by_user"}


def complete_action_direction(state, action, record_id, status):
    if status != "completed" or not action.get("_approved_direction_hash"):
        return
    for job in state.get("remote_finetune_jobs", {}).values():
        handoff = job.get("training_handoff") or {}
        proposal = handoff.get("direction_proposal") or {}
        if direction_hash(proposal) != action["_approved_direction_hash"]:
            continue
        handoff.update(direction_status="completed", stage="followup_completed")
        candidate = state.get("candidate_models", {}).get(
            proposal.get("candidate_model_version"), {}
        )
        review = candidate.get("agent_review") or {}
        review["followup_result"] = {"status": status, "record_id": record_id}

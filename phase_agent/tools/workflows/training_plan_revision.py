"""Explicit edits revise the saved direction before selecting its concrete action."""

from copy import deepcopy


def revise_training_plan(state, stored, feedback, agent_client, state_path=None):
    action = (stored.get("agent_proposal") or {}).get("raw_action") or {}
    bound = action.get("_approved_direction_hash")
    if not bound:
        return state, None
    from phase_agent.tools.state.approved_direction import action_direction, direction_hash
    from phase_agent.tools.local.training_agent_review import review_training_choice

    direction = action_direction(state)
    if direction and direction_hash(direction) != bound:
        return state, None  # The lifecycle already incorporated this feedback.
    updated = deepcopy(state)
    waits = [
        deepcopy(job["training_handoff"])
        for job in updated.get("remote_finetune_jobs", {}).values()
        if (job.get("training_handoff") or {}).get("direction_proposal") == direction
    ]
    if not direction or len(waits) != 1:
        return state, {
            "status": "rejected",
            "state": state,
            "reason": "旧方向已变化，请重新查看当前方案；未执行。",
        }
    updated, waits = review_training_choice(
        updated,
        waits,
        agent_client,
        state_path=state_path,
        user_message=feedback,
        revise_approved=True,
    )
    if waits and waits[0].get("stage") in {"execution_plan_ready", "execution_plan_approved"}:
        review = (
            updated.get("candidate_models", {}).get(waits[0]["candidate_model_version"]) or {}
        ).get("agent_review") or {}
        if action_direction(updated) and review.get("user_feedback") == str(feedback).strip():
            return updated, None
    if waits and waits[0].get("stage") == "awaiting_activation_approval":
        updated.get("pending_execution_policies", {}).pop(
            next(
                (
                    key
                    for key, value in updated.get("pending_execution_policies", {}).items()
                    if value == stored
                ),
                "",
            ),
            None,
        )
        updated.setdefault("superseded_training_proposals", []).append(
            {"record": deepcopy(stored), "reason": "user_revised_direction"}
        )
        return updated, {
            "status": "training_handoff",
            "state": updated,
            "training_handoffs": waits,
            "submitted": False,
            "reason": waits[0]["reason"],
        }
    return state, {
        "status": "rejected",
        "state": state,
        "reason": "新方向未形成有效方案；原方案保留，未执行。",
    }

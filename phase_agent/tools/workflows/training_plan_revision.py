"""Preserve historical training plans when their revision policy is unsupported."""


def revise_training_plan(state, stored, feedback, agent_client, state_path=None):
    action = (stored.get("agent_proposal") or {}).get("raw_action") or {}
    bound = action.get("_approved_direction_hash")
    if not bound:
        return state, None
    from phase_agent.tools.state.approved_direction import action_direction, direction_hash

    direction = action_direction(state)
    if direction and direction_hash(direction) != bound:
        return state, None  # The lifecycle already incorporated this feedback.
    # The current training policy no longer asks an agent to choose a direction.
    # Re-running it would silently replace a historical edit with model activation.
    return state, {
        "status": "rejected",
        "state": state,
        "reason": "当前训练流程不支持修订历史方向；原方案与审批保留，未切换模型或执行任务。",
        "submitted": False,
    }

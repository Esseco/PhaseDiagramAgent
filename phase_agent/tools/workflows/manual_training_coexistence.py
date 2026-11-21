"""Permit a frozen training-input proposal while verified inputs remain unsubmitted."""

from phase_agent.tools.state.prepared_unsubmitted import verified_ids


def allowed(frame):
    state = (frame.get("feedback") or {}).get("state") or {}
    wait = frame.get("manual_wait") or {}
    action = getattr(frame.get("agent_client"), "_frozen_training_action", None)
    human = frame.get("human_feedback")
    if action is None and isinstance(human, dict) and human.get("decision") == "approve":
        record = (state.get("pending_execution_policies") or {}).get(
            frame.get("invocation_id")
        ) or {}
        proposal = record.get("agent_proposal") or {}
        action = proposal.get("raw_action") or proposal
    if not isinstance(action, dict):
        return False
    params = action.get("parameters") or {}
    ids = wait.get("task_ids") or []
    if (
        action.get("tool") != "update_mlip"
        or params.get("action", "RETRAIN_MLIP") != "RETRAIN_MLIP"
        or params.get("prepare_inputs_only") is not True
        or (action.get("_decision_team") or {}).get("backend") != "codex_native"
        or not action.get("_simulation_plan")
        or not ids
        or len(ids) != len(set(ids))
        or wait.get("waiting_task_count") != len(ids)
        or wait.get("waiting_by_stage") != {"dft_single_point": len(ids)}
    ):
        return False
    return set(ids) <= verified_ids(state)


def blocking_tasks(state, action):
    """Keep all live tasks blocking except evidence-bound input-only training."""
    from phase_agent.policies.workflow_policy import active_tasks

    rows = active_tasks(state)
    if not isinstance(action, dict):
        return rows
    params = action.get("parameters") or {}
    if (
        action.get("tool") != "update_mlip"
        or params.get("action", "RETRAIN_MLIP") != "RETRAIN_MLIP"
        or params.get("prepare_inputs_only") is not True
        or (action.get("_decision_team") or {}).get("backend") != "codex_native"
        or not action.get("_simulation_plan")
    ):
        return rows
    verified = verified_ids(state)
    return [row for row in rows if row.get("task_id") not in verified]

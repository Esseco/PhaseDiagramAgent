"""Bounded reviewed memory and observed outcomes for both decision backends."""

from copy import deepcopy
from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge


def decision_memory_context(state, *, relevant=None):
    memory = state.get("decision_memory") or {}
    if relevant is None:
        from phase_agent.evidence.memory_action_scope import memory_action_scope

        action = memory_action_scope(state)["action"]
        system_id = ((state.get("confirmed_config") or {}).get("system") or {}).get(
            "system_id"
        ) or state.get("system_id")
        relevant = retrieve_relevant_knowledge(
            state,
            system_id=system_id,
            action=action,
            model_version=state.get("active_model_version"),
            limit=8,
            query=" ".join([str(action or ""), str(state.get("current_status") or "")]),
            token_budget=2000,
        )
    recent = []
    for row in (state.get("action_records") or [])[-5:]:
        proposal = row.get("agent_proposal") or {}
        action = row.get("final_action") or row.get("action") or proposal.get("raw_action") or {}
        execution = row.get("execution_result") or row.get("execution") or {}
        execution = execution if isinstance(execution, dict) else {}
        recent.append(
            {
                "record_id": row.get("record_id"),
                "status": row.get("status"),
                "tool": action.get("tool") or action.get("action_type"),
                "model_version": row.get("model_version"),
                "reason": str(action.get("reason") or "")[:600],
                "execution_status": execution.get("status"),
                "simulation_report": deepcopy(execution.get("simulation_report")),
                "team_id": (action.get("_decision_team") or {}).get("team_id"),
            }
        )
    return {
        "version": int(memory.get("version") or 0),
        "approved_knowledge": deepcopy(relevant),
        "short_term": deepcopy(memory.get("short_term") or {}),
        "recent_actions": recent,
        "pending_memory_count": sum(
            row.get("status") in {"pending_review", "conflict_review"}
            for row in state.get("memory_review_queue") or []
        ),
        "authority": "Current facts, configuration and approvals govern. Reviewed memory is scoped advice; historical outcomes do not prove unexecuted benefits. Pending knowledge is not active.",
    }

"""Bounded reviewed memory and observed outcomes for both decision backends."""

from copy import deepcopy
from contextvars import ContextVar
from contextlib import contextmanager
import hashlib
import json
from phase_agent.persistence.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge


def decision_memory_context(state, *, relevant=None, constraints=None, diagnostics=None):
    memory = state.get("decision_memory") or {}
    diagnostics = [] if diagnostics is None else diagnostics
    constraints = [] if constraints is None else constraints
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
            query=memory_query(state, action),
            diagnostics=diagnostics,
            required_constraints=constraints,
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
    short_term = deepcopy(memory.get("short_term") or {})
    rewards = short_term.get("recent_rewards")
    if isinstance(rewards, list):
        for row in rewards:
            if (
                isinstance(row, dict)
                and isinstance(row.get("batch_id"), str)
                and len(row["batch_id"]) > 240
            ):
                raw_id = row["batch_id"]
                digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
                parts = raw_id.count("+") + 1
                row["batch_id"] = f"{raw_id[:160]}...(+{parts}_parts,{digest})"
    return {
        "version": int(memory.get("version") or 0),
        "retrieval_filters": diagnostics,
        "reviewed_constraints": constraints,
        "human_constraints": deepcopy(state.get("human_constraints") or []),
        "approved_knowledge": deepcopy(relevant),
        "short_term": short_term,
        "recent_actions": recent,
        "pending_memory_count": sum(
            row.get("status") in {"pending_review", "conflict_review"}
            for row in state.get("memory_review_queue") or []
        ),
        "authority": "Current facts, configuration and approvals govern. Reviewed memory is scoped advice; historical outcomes do not prove unexecuted benefits. Pending knowledge is not active.",
    }


def memory_query(state, action=None):
    """Use explicit scientific scope, never classify intent from user keywords."""
    return (
        " ".join(
            str(state.get(key) or "")
            for key in (
                "scientific_question",
                "role_task",
                "failure_type",
                "current_status",
                "coverage_gaps",
                "user_message",
                "user_instruction",
            )
        )
        + " "
        + str(action or "")
        + " "
        + current_memory_query()
    )


_memory_query = ContextVar("phase_memory_query", default="")


def current_memory_query():
    return _memory_query.get()


@contextmanager
def memory_query_scope(question=None, role_tasks=None):
    token = _memory_query.set(
        str(question or "") + " " + json.dumps(role_tasks or {}, ensure_ascii=False, sort_keys=True)
    )
    try:
        yield
    finally:
        _memory_query.reset(token)

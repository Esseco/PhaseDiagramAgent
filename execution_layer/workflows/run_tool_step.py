"""Confirmed-config tool step with an explicit execution policy gate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from decision_layer.agent.propose_tool_action import propose_agent_tool_action
from decision_layer.agent.revise_tool_proposal import revise_tool_proposal
from data_layer.memory.decision_memory import update_long_term_advice, update_long_term_memory
from execution_layer.dispatch.execute_tool_action import execute_tool_action
from execution_layer.policy.execution_policy import apply_execution_policy, build_agent_proposal
from execution_layer.budget.reserve_action_budget import reserve_action_budget
from execution_layer.budget.record_budget_usage import record_budget_usage
from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.policy.validate_tool_action import validate_tool_action
from execution_layer.state.state_manager import agent_state_summary, update_state_snapshot


def run_tool_step(
    state: dict | None,
    session: dict,
    *,
    registry: dict,
    agent_client=None,
    context=None,
    execute=False,
    invocation_id=None,
    execution_mode: str | None = None,
    human_feedback: str | dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
) -> dict:
    """Propose, gate, validate, and optionally execute one tool action.

    Interactive mode returns ``awaiting_approval`` on the first call. Call it
    again with the same ``invocation_id`` and ``human_feedback`` to continue.
    """
    current = deepcopy(
        state
        or {
            "status": "ready",
            "tasks": [],
            "decisions": [],
            "action_records": [],
            "effective_decisions": {},
            "invocations": {},
            "pending_execution_policies": {},
        }
    )
    current.setdefault("action_records", [])
    current.setdefault("pending_execution_policies", {})
    if invocation_id and invocation_id in current.get("invocations", {}):
        return {**deepcopy(current["invocations"][invocation_id]), "state": current, "idempotent_replay": True}

    explicit_mode = execution_mode is not None
    mode = execution_mode or ("autonomous" if execute else "dry_run")
    config = (
        (session.get("confirmed_snapshot") or {}).get("config")
        if session.get("status") == "confirmed"
        else session.get("config", {})
    )
    if session.get("status") == "confirmed":
        current.setdefault("confirmed_config_version", session["confirmed_snapshot"]["config_version"])
    current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
    decision_state = agent_state_summary(current)

    pending_key = invocation_id or "__single_interactive_action__"
    stored = current["pending_execution_policies"].get(pending_key)
    advice_changed = False
    if mode == "interactive" and stored and isinstance(human_feedback, dict) and human_feedback.get("long_term_advice") is not None:
        updated = update_long_term_advice(current, human_feedback["long_term_advice"], source=f"{pending_key}:revision-{stored.get('revision', 0)}")
        advice_changed = updated.get("decision_memory") != current.get("decision_memory")
        current = updated
        if advice_changed:
            current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
            decision_state = agent_state_summary(current)
    if mode == "interactive" and stored and isinstance(human_feedback, dict) and human_feedback.get("long_term_memory") is not None:
        updated = update_long_term_memory(current, human_feedback["long_term_memory"],
                                          source=f"{pending_key}:revision-{stored.get('revision', 0)}")
        memory_changed = updated.get("decision_memory") != current.get("decision_memory")
        advice_changed = advice_changed or memory_changed
        current = updated
        if memory_changed:
            current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
            decision_state = agent_state_summary(current)
    if mode == "interactive" and stored:
        proposal = deepcopy(stored["agent_proposal"])
        record_id = stored["record_id"]
    elif mode == "replay":
        replay_action = deepcopy((replay_record or {}).get("final_action") or {})
        proposal = build_agent_proposal(replay_action, decision_state)
        record_id = _record_id(current, invocation_id)
    else:
        allowed = list((config.get("agent") or {}).get("allowed_tools") or [])
        action = propose_agent_tool_action(decision_state, agent_client=agent_client, allowed_tools=allowed)
        if action.get("_llm_usage"):
            current = record_budget_usage(
                current, {"llm_usage": action["_llm_usage"], "iteration": current.get("iteration", 0)}
            )
        proposal = build_agent_proposal(action, decision_state)
        record_id = _record_id(current, invocation_id)

    opinion = _extract_opinion(human_feedback) if mode == "interactive" and stored else None
    if advice_changed:
        opinion = str(human_feedback.get("comment") or "") + "\n请根据更新后的人工长期建议重新分析，生成供人工审批的新 proposal。"
    if opinion is not None:
        allowed = list((config.get("agent") or {}).get("allowed_tools") or [])
        revision = revise_tool_proposal(proposal, opinion, state=decision_state, allowed_tools=allowed, agent_client=agent_client)
        proposal = build_agent_proposal(revision["action"], decision_state)
        history = deepcopy(stored.get("feedback_history") or [])
        history.append({"comment": opinion, "revision_status": revision["revision_status"], "analysis": revision["analysis"]})
        revision_number = int(stored.get("revision", 0)) + 1
        current["pending_execution_policies"][pending_key] = {"record_id": record_id, "agent_proposal": deepcopy(proposal), "revision": revision_number, "feedback_history": history}
        record = _audit_record(record_id, mode, proposal, {"decision": "comment", "comment": opinion}, None, None, "awaiting_approval")
        record["feedback_history"] = history
        _upsert_record(current, record)
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        return {"status": "awaiting_approval", "execution_mode": mode, "agent_proposal": proposal, "human_feedback": {"decision": "comment", "comment": opinion}, "feedback_history": history, "revision": revision_number, "final_action": None, "action": proposal["raw_action"], "validation": None, "execution": None, "execution_result": None, "record_id": record_id, "state": current, "idempotent_replay": False}

    policy = apply_execution_policy(
        proposal,
        execution_mode=mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
    )
    if policy["status"] == "awaiting_approval":
        record = _audit_record(record_id, mode, proposal, None, None, None, "awaiting_approval", deepcopy((stored or {}).get("feedback_history") or []))
        _upsert_record(current, record)
        current["pending_execution_policies"][pending_key] = {
            "record_id": record_id,
            "agent_proposal": deepcopy(proposal),
            "revision": int((stored or {}).get("revision", 0)),
            "feedback_history": deepcopy((stored or {}).get("feedback_history") or []),
        }
        response = {
            "status": "awaiting_approval",
            "execution_mode": mode,
            "agent_proposal": proposal,
            "human_feedback": None,
            "final_action": None,
            "action": proposal["raw_action"],
            "validation": None,
            "execution": None,
            "execution_result": None,
            "record_id": record_id,
            "revision": int((stored or {}).get("revision", 0)),
            "feedback_history": deepcopy((stored or {}).get("feedback_history") or []),
        }
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        return {**response, "state": current, "idempotent_replay": False}

    current["pending_execution_policies"].pop(pending_key, None)
    if policy["status"] == "rejected_by_user":
        response = _policy_rejection(record_id, mode, proposal, policy)
        _upsert_record(
            current,
            _audit_record(record_id, mode, proposal, policy["human_feedback"], None, None, response["status"], deepcopy((stored or {}).get("feedback_history") or [])),
        )
        current.setdefault("decisions", []).append(deepcopy(response))
        _store_completed(current, invocation_id, response)
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        return {**response, "state": current, "idempotent_replay": False}

    action = policy["final_action"]
    validation = validate_tool_action(action, current, session, registry)
    if not validation["valid"] and mode != "interactive" and (config.get("agent") or {}).get("rule_fallback", True):
        allowed = list((config.get("agent") or {}).get("allowed_tools") or [])
        action = propose_agent_tool_action(decision_state, agent_client=None, allowed_tools=allowed)
        action["fallback_reason"] = ",".join(validation["errors"])
        validation = validate_tool_action(action, current, session, registry)

    should_execute = policy["execute"] if explicit_mode else bool(execute)
    if not validation["valid"]:
        status, execution = "rejected", None
    elif not should_execute:
        status, execution = "planned_only", None
    else:
        formal = action.get("tool") not in {"check_convergence", "pause_search"}
        owns_action_reservation = formal and registry.get(action.get("tool"), {}).get("reservation_mode") != "children"
        if owns_action_reservation:
            current = reserve_action_budget(current, action, config_version=validation["config_version"])
        execution = execute_tool_action(
            action,
            registry=registry,
            context={
                **(context or {}),
                "confirmed_config": config,
                "config_version": validation["config_version"],
            },
        )
        current, payload_status = _apply_execution_result(
            current, action, execution, record_id=record_id, formal=owns_action_reservation
        )
        status = payload_status or execution["status"]

    response = {
        "status": status,
        "execution_mode": mode,
        "agent_proposal": proposal,
        "human_feedback": deepcopy(policy["human_feedback"]),
        "final_action": deepcopy(action),
        "action": deepcopy(action),
        "validation": validation,
        "execution": execution,
        "execution_result": deepcopy(execution),
        "record_id": record_id,
    }
    _upsert_record(
        current,
        _audit_record(record_id, mode, proposal, policy["human_feedback"], action, execution, status, deepcopy((stored or {}).get("feedback_history") or [])),
    )
    current.setdefault("decisions", []).append(
        {"config_version": validation.get("config_version"), **deepcopy(response)}
    )
    _store_completed(current, invocation_id, response)
    current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
    return {**response, "state": current, "idempotent_replay": False}


def _record_id(state, invocation_id):
    return invocation_id or f"action-{len(state.get('action_records', [])) + 1:06d}"


def _extract_opinion(feedback):
    if isinstance(feedback, str):
        return None if feedback.strip().lower() in {"approve", "同意", "reject"} else feedback.strip()
    if not isinstance(feedback, dict):
        return None
    decision = feedback.get("decision")
    comment = str(feedback.get("comment") or "").strip()
    if comment.lower() in {"approve", "同意"}:
        return None
    if decision in {None, "comment", "revise"} and comment:
        return comment
    return None


def _audit_record(record_id, mode, proposal, feedback, final_action, result, status, feedback_history=None):
    return {
        "record_id": record_id,
        "execution_mode": mode,
        "status": status,
        "agent_proposal": deepcopy(proposal),
        "human_feedback": deepcopy(feedback),
        "final_action": deepcopy(final_action),
        "execution_result": deepcopy(result),
        "feedback_history": deepcopy(feedback_history or []),
    }


def _upsert_record(state, record):
    records = state.setdefault("action_records", [])
    for index, existing in enumerate(records):
        if existing.get("record_id") == record["record_id"]:
            records[index] = deepcopy(record)
            return
    records.append(deepcopy(record))


def _store_completed(state, invocation_id, response):
    if invocation_id:
        state.setdefault("invocations", {})[invocation_id] = deepcopy(response)


def _policy_rejection(record_id, mode, proposal, policy):
    return {
        "status": "rejected_by_user",
        "execution_mode": mode,
        "agent_proposal": proposal,
        "human_feedback": deepcopy(policy["human_feedback"]),
        "final_action": None,
        "action": proposal["raw_action"],
        "validation": None,
        "execution": None,
        "execution_result": None,
        "record_id": record_id,
    }


def _apply_execution_result(current, action, execution, *, record_id, formal):
    if execution.get("status") != "completed" or not isinstance(execution.get("result"), dict):
        return current, None
    payload = deepcopy(execution["result"])
    if isinstance(payload.get("state"), dict):
        merged = deepcopy(current)
        merged.update(deepcopy(payload["state"]))
        current = merged
    payload_status = payload.get("status")
    if not formal:
        return current, payload_status
    task_key = payload.get("task_key") or action.get("task_key")
    task_id = payload.get("task_id") or f"{record_id}:task"
    if payload_status in {"pending", "running", "completed", "failed", "timeout", "cancelled"}:
        task_result = {
            **payload,
            "task_id": task_id,
            "task_key": task_key,
            "status": payload_status,
        }
        reconciled = reconcile_task_results(current, [task_result])
        current = reconciled["state"]
    return current, payload_status

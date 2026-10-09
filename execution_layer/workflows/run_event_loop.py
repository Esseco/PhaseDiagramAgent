"""Resumable multi-step Agent → policy → validation → tool event loop."""

from copy import deepcopy
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from orchestration.state import EventGraphState
from orchestration.event_loop_graph import run_event_loop_graph

from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.workflows.run_tool_step import run_tool_step
from execution_layer.workflows.compact_action_history import compact_action_history
from execution_layer.step_runner.file_protocol import write_json
from execution_layer.policy.file_approval import read_approval_decision, write_approval_request
from data_layer.memory.decision_memory import update_long_term_advice
from execution_layer.state.state_manager import update_state_snapshot


STOP_STATUSES = {
    "awaiting_approval", "rejected_by_user", "rejected", "failed",
    "not_configured", "paused", "converged", "budget_exhausted",
    "pending", "running", "configuration_revision_required", "configuration_version_mismatch",
    "approval_reconciliation_required",
    "execution_reconciliation_required",
}


@dataclass
class ActionLoopSession:
    """Invocation-local loop callbacks, not checkpointable business state."""
    prepare_step: Callable
    execute: Callable
    persist: Callable
    route: Callable
    result: Callable
    max_steps: int


def run_event_loop(state, session, *, registry, agent_client=None, context=None,
                   execution_mode="autonomous", human_feedback=None, replay_record=None,
                   recovered_results=None, max_steps=1, invocation_id=None, state_path=None,
                   approval_directory=None, initial_long_term_advice=None, event_graph=None):
    """Public compatibility entry: execute, persist, and return the business result."""
    loop = prepare_event_loop(state, session, registry=registry, agent_client=agent_client,
        context=context, execution_mode=execution_mode, human_feedback=human_feedback,
        replay_record=replay_record, recovered_results=recovered_results, max_steps=max_steps,
        invocation_id=invocation_id, state_path=state_path, approval_directory=approval_directory,
        initial_long_term_advice=initial_long_term_advice)
    run_event_loop_graph(max_steps=loop.max_steps, execute=loop.execute,
                         persist=loop.persist, route=loop.route, event_graph=event_graph)
    return loop.result()


def prepare_event_loop(
    state,
    session,
    *,
    registry,
    agent_client=None,
    context=None,
    execution_mode="autonomous",
    human_feedback=None,
    replay_record=None,
    recovered_results=None,
    max_steps=1,
    invocation_id=None,
    state_path=None,
    approval_directory=None,
    initial_long_term_advice=None,
):
    """Run bounded Agent actions and persist after every state transition."""
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps 必须是正整数")
    current = _load_state(state)
    if "decision_memory" not in current:
        current = update_long_term_advice(current, initial_long_term_advice, source="initial_user_input")
    current.setdefault("event_index", 0)
    current.setdefault("event_history", [])
    reconciliation = reconcile_task_results(current, recovered_results)
    current = reconciliation["state"]
    _save_state(current, state_path)
    events = []
    final_status = "max_steps_reached"
    starting_index = int(current.get("event_index", 0))
    from langgraph.graph import END

    def prepare_step(graph_state):
        offset = graph_state["offset"]
        step_id = _step_invocation_id(starting_index, invocation_id, offset, max_steps)
        if execution_mode == "interactive" and offset == 0:
            pending_ids = list(current.get("pending_execution_policies", {}))
            if len(pending_ids) == 1 and (invocation_id is None or max_steps > 1):
                step_id = pending_ids[0]
        step_context = {
            **(context or {}),
            "state_path": str(state_path) if state_path else None,
            "event_state": deepcopy(current),
            "search_state": deepcopy(current),
        }
        step_feedback = human_feedback if offset == 0 else None
        if execution_mode == "interactive" and step_feedback is None and approval_directory:
            pending = current.get("pending_execution_policies", {}).get(step_id)
            if pending:
                step_feedback = read_approval_decision(approval_directory, step_id, pending)
        return {"state": current, "session": session, "registry": registry,
                "agent_client": agent_client, "context": step_context, "execute": True,
                "invocation_id": step_id, "execution_mode": execution_mode,
                "human_feedback": step_feedback,
                "replay_record": replay_record if offset == 0 else None}, step_id

    def execute_step(graph_state, *, tool_graph=None):
        arguments, step_id = prepare_step(graph_state)
        response = run_tool_step(**arguments, **({"tool_graph": tool_graph} if tool_graph is not None else {}))
        return {"response": response, "step_id": step_id}

    def persist_step(graph_state):
        nonlocal current, final_status
        response, step_id = graph_state["response"], graph_state["step_id"]
        current = response["state"]
        event = compact_action_history(response)
        current.setdefault("event_history", []).append(event)
        current["event_index"] = int(current.get("event_index", 0)) + (0 if response.get("idempotent_replay") else 1)
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        events.append(event)
        final_status = response["status"]
        _save_state(current, state_path)
        if final_status == "awaiting_approval" and approval_directory:
            events[-1]["approval_files"] = write_approval_request(approval_directory, step_id, response)
        return {"offset": graph_state["offset"] + 1}

    def next_step(graph_state):
        nonlocal final_status
        if final_status in STOP_STATUSES:
            return END
        if current.get("pending_tasks"):
            final_status = "tasks_in_progress"
            return END
        return END if graph_state["offset"] >= max_steps else "execute_validated_action"

    def result():
        return {"status": final_status, "state": current, "events": events,
                "reconciled": reconciliation["reconciled"], "steps_executed": len(events)}

    return ActionLoopSession(prepare_step, execute_step, persist_step, next_step, result, max_steps)


def _step_invocation_id(starting_index, base, offset, max_steps):
    if base and max_steps == 1:
        return base
    prefix = base or "event"
    return f"{prefix}-{starting_index + offset + 1:06d}"


def _load_state(state):
    if isinstance(state, (str, Path)):
        return json.loads(Path(state).read_text(encoding="utf-8"))
    return deepcopy(state or {})


def _save_state(state, path):
    if path is None:
        return
    write_json(path, state)
    from data_layer.memory.publish_memory_views import publish_memory_views
    publish_memory_views(state, path)

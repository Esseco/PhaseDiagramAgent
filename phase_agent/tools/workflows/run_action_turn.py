"""One scientific action per turn, with existing approval and durable bookkeeping."""

from copy import deepcopy
import json
from pathlib import Path

from phase_agent.interface_compatibility import call_with_optimizations
from phase_agent.agents.common.dialogue_contract import DIALOGUE_STATUSES
from phase_agent.policies.file_approval import read_approval_decision, write_approval_request
from phase_agent.persistence.memory.decision_memory import update_long_term_advice
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.state.state_manager import update_state_snapshot
from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.tools.workflows.compact_action_history import compact_action_history
from phase_agent.tools.workflows.run_tool_step import run_tool_step


STOP_STATUSES = set(DIALOGUE_STATUSES) | {
    "awaiting_approval",
    "rejected_by_user",
    "rejected",
    "failed",
    "not_configured",
    "paused",
    "converged",
    "budget_exhausted",
    "pending",
    "running",
    "configuration_revision_required",
    "configuration_version_mismatch",
    "proposal_context_changed",
    "approval_reconciliation_required",
    "execution_reconciliation_required",
}


def run_action_turn(
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
    tool_graph=None,
    _owned=False,
):
    """Validate one action, persist its result, and return to the project graph."""
    if type(max_steps) is not int or max_steps != 1:
        raise ValueError("Scientific workflow permits exactly one action per turn")
    current = _load_state(state, _owned=_owned)
    if "decision_memory" not in current:
        current = update_long_term_advice(
            current, initial_long_term_advice, source="initial_user_input"
        )
    current.setdefault("event_index", 0)
    current.setdefault("event_history", [])
    reconciliation = call_with_optimizations(
        reconcile_task_results, current, recovered_results, _optimizations={"_owned": True}
    )
    current = reconciliation["state"]
    from phase_agent.policies.workflow_policy import normalize_workflow_state

    current = call_with_optimizations(
        normalize_workflow_state, current, _optimizations={"_owned": True}
    )
    _save_state(current, state_path)
    step_id = invocation_id or f"event-{int(current.get('event_index', 0)) + 1:06d}"
    if execution_mode == "interactive" and invocation_id is None:
        pending_ids = list(current.get("pending_execution_policies") or {})
        if len(pending_ids) == 1:
            step_id = pending_ids[0]
    feedback = human_feedback
    if execution_mode == "interactive" and feedback is None and approval_directory:
        pending = (current.get("pending_execution_policies") or {}).get(step_id)
        if pending:
            feedback = read_approval_decision(approval_directory, step_id, pending)
    arguments = {
        "state": current,
        "session": session,
        "registry": registry,
        "agent_client": agent_client,
        "context": {
            **(context or {}),
            "state_path": str(state_path) if state_path else None,
            "_event_state_owned": True,
        },
        "execute": True,
        "invocation_id": step_id,
        "execution_mode": execution_mode,
        "human_feedback": feedback,
        "replay_record": replay_record,
    }
    response = run_tool_step(
        **arguments, **({"tool_graph": tool_graph} if tool_graph is not None else {})
    )
    current = response["state"]
    event = compact_action_history(response)
    current.setdefault("event_history", []).append(event)
    current["event_index"] = int(current.get("event_index", 0)) + (
        0 if response.get("idempotent_replay") else 1
    )
    current = call_with_optimizations(
        update_state_snapshot,
        current,
        config_version=current.get("confirmed_config_version"),
        _optimizations={"_owned": True},
    )
    final_status = response["status"]
    _save_state(current, state_path)
    from phase_agent.runtime.turn_process import process_event
    from phase_agent.tools.state.task_waiting import awaiting_task_result

    process_event(
        "动作后已保存的事实",
        {
            "status": final_status,
            "config_version": current.get("confirmed_config_version"),
            "active_model_version": current.get("active_model_version"),
            "event_index": current.get("event_index"),
            "task_count": len(current.get("tasks") or []),
            "waiting_results": sum(awaiting_task_result(t) for t in current.get("tasks") or []),
            "含义": "动作结果已入账；等待任务时暂停，下一轮重新读取事实，不表示科学收敛。",
        },
    )
    if final_status == "awaiting_approval" and approval_directory:
        event["approval_files"] = write_approval_request(approval_directory, step_id, response)
    from phase_agent.graphs.wait_contract import wait_boundary

    if final_status not in STOP_STATUSES and wait_boundary(final_status) is None:
        if current.get("pending_tasks") or any(
            awaiting_task_result(task) for task in current.get("tasks") or []
        ):
            final_status = "tasks_in_progress"
    return {
        "status": final_status,
        "reason": event.get("reason"),
        "answer": event.get("answer") if event.get("status") in DIALOGUE_STATUSES else None,
        "state": current,
        "events": [event],
        "reconciled": reconciliation["reconciled"],
        "steps_executed": 1,
    }


def _load_state(state, *, _owned=False):
    if isinstance(state, (str, Path)):
        return json.loads(Path(state).read_text(encoding="utf-8"))
    # Lifecycle frames are already private; public callers keep an independent copy.
    return state if _owned and isinstance(state, dict) else deepcopy(state or {})


def _save_state(state, path):
    if path is None:
        return
    write_json(path, state)
    from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

    publish_memory_views(state, path)

"""Resumable multi-step Agent → policy → validation → tool event loop."""

from copy import deepcopy
import json
from pathlib import Path

from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.workflows.run_tool_step import run_tool_step
from execution_layer.policy.file_approval import read_approval_decision, write_approval_request
from data_layer.memory.decision_memory import update_long_term_advice
from execution_layer.state.state_manager import update_state_snapshot


STOP_STATUSES = {
    "awaiting_approval", "rejected_by_user", "rejected", "failed",
    "not_configured", "paused", "converged", "budget_exhausted",
    "pending", "running",
}


def run_event_loop(
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
    if not isinstance(max_steps, int) or max_steps <= 0:
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
    for offset in range(max_steps):
        step_id = _step_invocation_id(starting_index, invocation_id, offset, max_steps)
        if execution_mode == "interactive" and offset == 0:
            pending_ids = list(current.get("pending_execution_policies", {}))
            if len(pending_ids) == 1 and (invocation_id is None or max_steps > 1):
                step_id = pending_ids[0]
        step_context = {
            **(context or {}),
            "event_state": deepcopy(current),
            "search_state": deepcopy(current),
        }
        step_feedback = human_feedback if offset == 0 else None
        if execution_mode == "interactive" and step_feedback is None and approval_directory:
            pending = current.get("pending_execution_policies", {}).get(step_id)
            if pending:
                step_feedback = read_approval_decision(approval_directory, step_id, pending)
        response = run_tool_step(
            current,
            session,
            registry=registry,
            agent_client=agent_client,
            context=step_context,
            execute=True,
            invocation_id=step_id,
            execution_mode=execution_mode,
            human_feedback=step_feedback,
            replay_record=replay_record if offset == 0 else None,
        )
        current = response["state"]
        event = {key: deepcopy(value) for key, value in response.items() if key != "state"}
        current.setdefault("event_history", []).append(event)
        current["event_index"] = int(current.get("event_index", 0)) + (0 if response.get("idempotent_replay") else 1)
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        events.append(event)
        final_status = response["status"]
        _save_state(current, state_path)
        if final_status == "awaiting_approval" and approval_directory:
            events[-1]["approval_files"] = write_approval_request(approval_directory, step_id, response)
        if final_status in STOP_STATUSES:
            break
        if current.get("pending_tasks"):
            final_status = "tasks_in_progress"
            break
    return {
        "status": final_status,
        "state": current,
        "events": events,
        "reconciled": reconciliation["reconciled"],
        "steps_executed": len(events),
    }


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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)

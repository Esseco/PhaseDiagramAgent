"""Tool audit records and scientific result merging; no graph routing."""
from copy import deepcopy
from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.workflows.compact_action_history import compact_action_history


def _audit_record(record_id, mode, proposal, feedback, final_action, result, status, feedback_history=None):
    return {
        "record_id": record_id,
        "execution_mode": mode,
        "status": status,
        "agent_proposal": deepcopy(proposal),
        "human_feedback": deepcopy(feedback),
        "final_action": deepcopy(final_action),
        "execution_result": compact_action_history(result),
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
        state.setdefault("invocations", {})[invocation_id] = compact_action_history(response)


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
        # A handler receives the state from before this approval was consumed.
        # Its returned scientific state must not restore that pending approval.
        merged["pending_execution_policies"] = deepcopy(current.get("pending_execution_policies") or {})
        current = merged
    payload_status = payload.get("status")
    if action.get("tool") in {"generate_branches", "select_dft_candidates"} and payload_status not in {
            "failed", "rejected", "not_configured", "awaiting_approval"}:
        from analysis_layer.state.post_dft_assessment import post_dft_assessment
        assessment = post_dft_assessment(current, current.get("confirmed_config") or {})
        source_scope = action.get("_post_dft_scope_key") or (assessment or {}).get("scope_key")
        successful = payload_status in {"completed", "generated", "ready", "pending", "prepared", "already_prepared"}
        if action.get("tool") == "select_dft_candidates":
            successful = successful and bool(payload.get("tasks"))
        if source_scope and successful:
            decided = current.setdefault("post_dft_decided_rounds", [])
            if source_scope not in decided:
                decided.append(source_scope)
            version = (assessment or {}).get("scope", {}).get("model_version") or current.get("active_model_version")
            current.setdefault("post_dft_decided_mc_tasks", {})[version] = sorted(
                str(row.get("task_id")) for row in current.get("tasks") or []
                if row.get("stage") == "deep_search" and row.get("model_version") == version)
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

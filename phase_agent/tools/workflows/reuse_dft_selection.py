"""Restore a bound plan; failed executions require existing no-effect reconciliation."""

from copy import deepcopy


def reusable_selection(state):
    if state.get("pending_execution_policies") or not state.get("action_records"):
        return None
    record = state["action_records"][-1]
    execution = record.get("execution_result") or {}
    result = execution.get("result") or {}
    outcome = execution.get("operation_outcome") or {}
    proposal = record.get("agent_proposal") or {}
    action = proposal.get("raw_action") or {}
    template_rejected = (
        record.get("status") == "rejected"
        and execution.get("status") == "completed"
        and result.get("status") == "rejected"
        and result.get("reason") in {"DFT 参数模板尚未确认", "DFT 公共模板尚未确认"}
    )
    validation = result.get("validation") or {}
    validation_rejected = (
        record.get("status") == "rejected"
        and execution.get("status") == "completed"
        and result.get("status") == "rejected"
        and validation.get("valid") is False
        and bool(validation.get("errors") or validation.get("rejected"))
        and result.get("tasks") == []
        and outcome.get("task_ids") == []
        and outcome.get("artifacts") == []
        and outcome.get("dispatch_status") == "completed"
        and outcome.get("result_status") == "rejected"
        and outcome.get("tool") == "select_dft_candidates"
        and not execution.get("error")
        and not outcome.get("error")
        and bool(state.get("confirmed_config_version"))
        and bool(state.get("active_model_version"))
    )
    audit = (state.get("execution_reconciliations") or {}).get(record.get("record_id")) or {}
    identity = audit.get("identity") or {}
    from phase_agent.tools.policy.file_approval import proposal_hash

    verified_failure = (
        record.get("status") in {"failed", "reconciled_no_effect"}
        and execution.get("status") == "failed"
        and audit.get("resolution") == "verified_no_effect"
        and bool(audit.get("evidence"))
        and bool(audit.get("reviewed_at"))
        and identity.get("invocation_id") == record.get("record_id")
        and identity.get("config_version") == state.get("confirmed_config_version")
        and identity.get("tool") == action.get("tool")
        and identity.get("action_hash") == proposal_hash(record.get("final_action") or action)
    )
    if (
        not (template_rejected or verified_failure or validation_rejected)
        or (record.get("human_feedback") or {}).get("decision") != "approve"
        or action.get("tool") != "select_dft_candidates"
        or not action.get("task_key")
        or not record.get("record_id")
        or not (action.get("parameters") or {}).get("decisions")
        or result.get("tasks")
        or outcome.get("task_ids")
        or outcome.get("submitted")
        or outcome.get("config_version") != state.get("confirmed_config_version")
        or outcome.get("model_version") != state.get("active_model_version")
        or (
            (state.get("effective_decisions") or {}).get(action.get("task_key"))
            and not (
                verified_failure
                and (state["effective_decisions"][action["task_key"]]).get("verified_no_effect")
            )
        )
        or any(
            t.get("parent_decision_id") == record.get("record_id")
            or t.get("scientific_parent_task_key") == action.get("task_key")
            for t in state.get("tasks") or []
        )
    ):
        return None
    reused = deepcopy(proposal)
    reused["reuse_source_record_id"] = record["record_id"]
    if validation_rejected:
        # This handler rejects validation before reserving costs or creating tasks.
        reused["reuse_requires_preview"] = True
    return reused

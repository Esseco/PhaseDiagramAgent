"""Close verified local training-input delivery without completing remote training."""

from phase_agent.tools.budget.settle_budget import settle_budget


def settle_training_input(state, action, execution, *, record_id):
    key = action.get("task_key")
    reservation = (state.get("budget_reservations") or {}).get(key) or {}
    payload = execution.get("result") or {}
    if (
        action.get("tool") != "update_mlip"
        or (action.get("parameters") or {}).get("prepare_inputs_only") is not True
        or execution.get("status") != "completed"
        or execution.get("error")
        or payload.get("status") != "awaiting_remote_training"
        or payload.get("submitted") is not False
        or reservation.get("status") != "reserved"
        or reservation.get("reconciliation_required")
        or reservation.get("reserved_cost") != 0
        or reservation.get("relative_cost", 0) != 0
        or key in (state.get("pending_execution_policies") or {})
    ):
        return state
    handoffs = payload.get("handoffs") or []
    jobs = (state.get("remote_finetune_jobs") or {}).values()
    if not handoffs or not all(
        handoff.get("folder")
        and handoff.get("import_directory")
        and any(
            job.get("directory") == handoff["import_directory"]
            and (job.get("handoff") or {}).get("folder") == handoff["folder"]
            and (job.get("handoff") or {}).get("package_fingerprint")
            == handoff.get("package_fingerprint")
            for job in jobs
        )
        for handoff in handoffs
    ):
        return state
    current = settle_budget(
        state,
        task_key=key,
        settlement_id="training-input-delivered:" + record_id,
        task_status="completed",
        actual_cost=0,
        cost_source="verified_local_input_delivery",
    )["state"]
    current.setdefault("effective_decisions", {}).setdefault(key, {}).update(
        status="completed",
        record_id=record_id,
    )
    return current


def settle_saved_training_inputs(state):
    current = state
    for record in state.get("action_records") or []:
        current = settle_training_input(
            current,
            record.get("final_action") or {},
            record.get("execution_result") or {},
            record_id=str(record.get("record_id") or ""),
        )
    return current

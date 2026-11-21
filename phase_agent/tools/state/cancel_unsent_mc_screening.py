"""Cancel only proven unsubmitted Relax screens created by an MC allocation."""

from copy import deepcopy


def cancel_unsent_mc_screening(state, events, *, decision_id, allocation_checksum, _owned=False):
    claimed = [e.get("payload") or {} for e in events if e.get("event") == "claimed"]
    returned = [e.get("payload") or {} for e in events if e.get("event") == "tool_returned"]
    if len(claimed) != 1 or len(returned) != 1:
        raise ValueError("screening_receipt_missing_or_ambiguous")
    action = claimed[0].get("action") or {}
    identity = claimed[0].get("identity") or {}
    evidence = returned[0].get("evidence") or {}
    preview = (action.get("parameters") or {}).get("budget_preview") or {}
    targets = action.get("target_ids") or []
    refs = evidence.get("returned_task_refs") or []
    ids = [row.get("task_id") for row in refs]
    if (
        identity.get("invocation_id") != decision_id
        or returned[0].get("identity") != identity
        or action.get("tool") != "allocate_mc_bohb"
        or returned[0].get("outcome") != "completed"
        or evidence.get("result_status") != "completed"
        or preview.get("allocation_checksum") != allocation_checksum
        or not targets
        or len(set(targets)) != len(targets)
        or not ids
        or None in ids
        or len(set(ids)) != len(ids)
        or any(
            row.get("stage") != "relax_and_feature"
            or row.get("status") != "pending"
            or row.get("parent_decision_id") != decision_id
            for row in refs
        )
    ):
        raise ValueError("screening_receipt_binding_invalid")
    tasks = [
        row for row in state.get("tasks") or [] if row.get("parent_decision_id") == decision_id
    ]
    if len(tasks) != len(ids) or {row.get("task_id") for row in tasks} != set(ids):
        raise ValueError("screening_task_scope_changed")
    batch = state.get("branch_batch") or {}
    pending = state.get("pending_branch_screening") or {}
    if (
        batch.get("status") != "relax_pending"
        or set(batch.get("branch_ids") or []) != set(targets)
        or set(pending.get("branch_ids") or []) != set(targets)
    ):
        raise ValueError("screening_branch_scope_changed")
    refs_by_id = {row["task_id"]: row for row in refs}
    binding_fields = (
        "task_key",
        "stage",
        "status",
        "model_version",
        "config_version",
        "parent_decision_id",
    )
    if any(
        any(row.get(key) != refs_by_id[row["task_id"]].get(key) for key in binding_fields)
        for row in tasks
    ):
        raise ValueError("screening_task_receipt_mismatch")
    reservations = state.get("budget_reservations") or {}
    unsafe_fields = (
        "slurm_batch_id",
        "job_id",
        "remote_job_id",
        "upload_operation_id",
        "input_path",
        "remote_path",
    )
    for row in tasks:
        r = reservations.get(row.get("task_key")) or {}
        if (
            row.get("stage") != "relax_and_feature"
            or row.get("status") != "pending"
            or row.get("branch_id") not in targets
            or any(row.get(k) for k in unsafe_fields)
            or r.get("status") != "reserved"
            or r.get("stage") != "relax_and_feature"
        ):
            raise ValueError("screening_task_may_have_been_submitted")
    current = state if _owned else deepcopy(state)
    ids_set = set(ids)
    for row in current.get("tasks") or []:
        if row.get("task_id") not in ids_set:
            continue
        row.update(
            status="cancelled",
            recovery_cancelled=True,
            cancellation_reason="erroneous_mc_screening",
        )
        r = current["budget_reservations"][row["task_key"]]
        r.update(
            status="released",
            released_cost=r["reserved_cost"],
            release_reason="erroneous_mc_screening",
        )
    from phase_agent.tools.state.task_waiting import active_pending_tasks
    from phase_agent.tools.state.branch_batch_state import transition_branch_batch

    current["pending_tasks"] = active_pending_tasks(current.get("tasks") or [])
    current["branch_batch"] = transition_branch_batch(
        current["branch_batch"], "failed", reason="erroneous_mc_screening_cancelled"
    )
    current.pop("pending_branch_screening", None)
    current.setdefault("cancelled_mc_screenings", []).append(
        {
            "decision_id": decision_id,
            "task_ids": sorted(ids),
            "allocation_checksum": allocation_checksum,
            "reason": "erroneous_mc_screening",
            "remote_cancel_sent": False,
        }
    )
    return current, len(ids)

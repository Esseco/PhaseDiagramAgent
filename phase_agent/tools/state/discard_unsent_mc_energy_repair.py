"""Cancel one explicitly discarded, proven-unsubmitted energy-repair handoff."""

from copy import deepcopy
from pathlib import Path


def discard_unsent_mc_energy_repair(
    state, events, *, decision_id, repair_id, user_message, workspace_root, _owned=False
):
    if not str(user_message or "").strip():
        raise ValueError("explicit_discard_instruction_required")
    claimed = [e.get("payload") or {} for e in events if e.get("event") == "claimed"]
    returned = [e.get("payload") or {} for e in events if e.get("event") == "tool_returned"]
    if len(claimed) != 1 or len(returned) != 1:
        raise ValueError("energy_repair_receipt_missing_or_ambiguous")
    action, identity = claimed[0].get("action") or {}, claimed[0].get("identity") or {}
    result = returned[0].get("evidence") or {}
    preview = (action.get("parameters") or {}).get("energy_repair_preview") or {}
    record = (state.get("mc_energy_repairs") or {}).get(repair_id) or {}
    refs = result.get("returned_task_refs") or []
    ids = [row.get("task_id") for row in refs]
    targets = action.get("target_ids") or []
    if (
        identity.get("invocation_id") != decision_id
        or returned[0].get("identity") != identity
        or action.get("tool") != "prepare_local_batch_files"
        or action.get("operation") != "prepare_mc_energy_repair"
        or (action.get("parameters") or {}).get("mode") != "mc_energy_repair_inputs"
        or preview.get("checksum") != repair_id
        or record.get("preview") != preview
        or record.get("approval_record_id") != decision_id
        or record.get("status") != "prepared"
        or returned[0].get("outcome") != "completed"
        or result.get("result_status") != "prepared"
        or not ids
        or None in ids
        or len(set(ids)) != len(ids)
        or set(ids) != set(record.get("task_ids") or [])
        or set(targets) != {x.get("source_task_id") for x in preview.get("frames") or []}
    ):
        raise ValueError("energy_repair_receipt_binding_invalid")
    tasks = [x for x in state.get("tasks") or [] if x.get("energy_basis_repair_id") == repair_id]
    if len(tasks) != len(ids) or {x.get("task_id") for x in tasks} != set(ids):
        raise ValueError("energy_repair_task_scope_changed")
    refs_by_id = {x["task_id"]: x for x in refs}
    bindings = (
        "task_key",
        "stage",
        "status",
        "model_version",
        "config_version",
        "parent_decision_id",
    )
    reservations = state.get("budget_reservations") or {}
    for task in tasks:
        reservation = reservations.get(task.get("task_key")) or {}
        if (
            any(task.get(k) != refs_by_id[task["task_id"]].get(k) for k in bindings)
            or task.get("stage") != "relax_and_feature"
            or task.get("status") != "pending"
            or task.get("parent_decision_id") != decision_id
            or task.get("job_id")
            or task.get("remote_job_id")
            or task.get("submitted_at")
            or (task.get("result_path") and Path(task["result_path"]).is_file())
            or reservation.get("status") != "reserved"
        ):
            raise ValueError("energy_repair_may_have_been_submitted")
    batch_ids = {x.get("slurm_batch_id") for x in tasks}
    batches = [b for b in state.get("slurm_batches") or [] if b.get("batch_id") in batch_ids]
    if (
        None in batch_ids
        or len(batches) != len(batch_ids)
        or any(
            b.get("status") != "prepared"
            or b.get("job_id")
            or b.get("slurm_job_id")
            or set(b.get("task_ids") or []) != set(ids)
            for b in batches
        )
    ):
        raise ValueError("energy_repair_batch_scope_or_status_changed")
    for batch in batches:
        folder = Path((batch.get("handoff") or {}).get("folder") or "")
        if (
            not folder.is_absolute()
            or not folder.resolve().is_relative_to(Path(workspace_root).resolve())
            or (folder / ".submitted_jobs.json").exists()
        ):
            raise ValueError("energy_repair_handoff_submission_unknown")
    current = state if _owned else deepcopy(state)
    task_ids = set(ids)
    for task in current.get("tasks") or []:
        if task.get("task_id") not in task_ids:
            continue
        task.update(
            status="cancelled", recovery_cancelled=True, cancellation_reason=str(user_message)
        )
        reservation = current["budget_reservations"][task["task_key"]]
        reservation.update(
            status="released",
            released_cost=reservation["reserved_cost"],
            release_reason="user_discarded_energy_repair",
        )
    for batch in current.get("slurm_batches") or []:
        if batch.get("batch_id") in batch_ids:
            batch.update(status="cancelled", cancellation_reason=str(user_message))
    current["mc_energy_repairs"][repair_id].update(
        status="cancelled", cancellation_reason=str(user_message)
    )
    from phase_agent.tools.state.task_waiting import active_pending_tasks

    current["pending_tasks"] = active_pending_tasks(current.get("tasks") or [])
    current.pop("current_state_snapshot", None)
    current.pop("agent_state_summary", None)
    current.setdefault("discarded_mc_energy_repairs", []).append(
        {
            "repair_id": repair_id,
            "decision_id": decision_id,
            "task_ids": sorted(ids),
            "user_message": str(user_message),
            "remote_cancel_sent": False,
        }
    )
    return current, len(ids)

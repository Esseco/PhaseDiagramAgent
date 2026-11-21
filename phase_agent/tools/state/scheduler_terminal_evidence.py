"""Validate external scheduler observations without changing task outcomes or costs."""

from pathlib import PurePosixPath, PureWindowsPath


def bind_terminal_observations(state, observations):
    """Attach attested scheduler evidence to the exact already-terminal tasks.

    Callers must obtain the observations from the scheduler; a local decision
    to stop waiting is not itself evidence that the remote job has ended.
    """
    if not isinstance(observations, list) or not observations:
        raise ValueError("scheduler_observations_required")
    updated = {**state, "tasks": [dict(task) for task in state.get("tasks") or []]}
    seen_tasks, seen_jobs = set(), set()
    for row in observations:
        if not isinstance(row, dict) or set(row) != {
            "task_id", "job_id", "state", "workdir", "elapsed", "observed_at", "source"
        }:
            raise ValueError("invalid_scheduler_observation")
        task_id, job_id = row["task_id"], row["job_id"]
        if not isinstance(task_id, str) or not task_id or not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError("invalid_scheduler_identity")
        if task_id in seen_tasks or job_id in seen_jobs:
            raise ValueError("duplicate_scheduler_identity")
        seen_tasks.add(task_id)
        seen_jobs.add(job_id)
        matches = [t for t in updated.get("tasks") or [] if t.get("task_id") == task_id]
        if len(matches) != 1:
            raise ValueError("scheduler_task_not_unique")
        task = matches[0]
        observation = _bind_one(task, row)
        prior = task.get("scheduler_terminal_evidence")
        if prior and prior != observation:
            raise ValueError("scheduler_evidence_conflict")
        task["scheduler_terminal_evidence"] = observation
    return updated


def _bind_one(task, row):
    if (row["source"] != "electerm_sacct" or row["state"] != "CANCELLED"
            or not isinstance(row["elapsed"], str) or not row["elapsed"]
            or not isinstance(row["observed_at"], str) or not row["observed_at"]):
        raise ValueError("scheduler_terminal_proof_required")
    if task.get("stage") != "dft_single_point" or task.get("status") != "cancelled":
        raise ValueError("scheduler_task_not_cancelled_dft")
    workdir = row["workdir"]
    if not isinstance(workdir, str) or not PurePosixPath(workdir).is_absolute():
        raise ValueError("scheduler_workdir_invalid")
    expected_folder = PureWindowsPath(task.get("input_path") or "").parent.name
    batch_id = task.get("batch_id")
    parts = PurePosixPath(workdir).parts
    if (not expected_folder or not batch_id or PurePosixPath(workdir).name != expected_folder
            or not any(part.endswith("_" + batch_id) for part in parts)):
        raise ValueError("scheduler_workdir_task_mismatch")
    observation = {**row, "task_key": task.get("task_key"), "batch_id": batch_id,
                   "task_checksum": task.get("task_checksum")}
    if not observation["task_key"] or not observation["task_checksum"]:
        raise ValueError("scheduler_task_binding_incomplete")
    return observation


def terminal_reservation_is_proven(state, key, reservation):
    """Recognize a terminal DFT task while retaining its unsettled cost."""
    if (reservation.get("stage") != "dft_single_point" or reservation.get("status") != "reserved"
            or reservation.get("reconciliation_required")):
        return False
    tasks = [t for t in state.get("tasks") or [] if t.get("task_key") == key]
    if len(tasks) != 1:
        return False
    task = tasks[0]
    evidence = task.get("scheduler_terminal_evidence") or {}
    try:
        row = {k: evidence[k] for k in (
            "task_id", "job_id", "state", "workdir", "elapsed", "observed_at", "source"
        )}
        checked = _bind_one(task, row)
    except (KeyError, TypeError, ValueError):
        return False
    return (checked == evidence and row["task_id"] == task.get("task_id")
            and isinstance(row["job_id"], str) and row["job_id"].isdigit()
            and task.get("config_version") == reservation.get("config_version"))


def user_waived_dft_reservation(state, key, reservation):
    """Honor an explicit stop-waiting decision without settling or deleting the cost."""
    if (reservation.get("stage") != "dft_single_point" or reservation.get("status") != "reserved"
            or reservation.get("reconciliation_required")):
        return False
    tasks = [t for t in state.get("tasks") or [] if t.get("task_key") == key]
    if len(tasks) != 1:
        return False
    task = tasks[0]
    if (task.get("stage") != "dft_single_point" or task.get("status") != "cancelled"
            or task.get("config_version") != reservation.get("config_version")
            or not task.get("task_id")):
        return False
    tid = task["task_id"]
    return any(
        decision.get("decision") == "close"
        and bool(decision.get("question_id"))
        and bool(str(decision.get("user_message") or "").strip())
        and tid in (decision.get("asked_task_ids") or [])
        and tid in (decision.get("waived_task_ids") or [])
        and tid in (decision.get("cancelled_task_ids") or [])
        and (decision.get("scope") or {}).get("model_version") == task.get("model_version")
        and (decision.get("scope") or {}).get("upload_operation_id") == task.get("upload_operation_id")
        for decision in state.get("dft_recovery_decisions") or []
    )

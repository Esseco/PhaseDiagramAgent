"""Explicit local MC cancellation; remote jobs, results and accounting stay intact."""

from copy import deepcopy
import hashlib
import json

from phase_agent.tools.state.task_waiting import active_pending_tasks

ACTIVE = frozenset({"pending", "running", "submitted", "unknown", "submission_unknown"})
BINDING_FIELDS = (
    "task_key",
    "stage",
    "config_version",
    "model_version",
    "relax_mc_bundle_id",
    "upload_operation_id",
    "task_checksum",
    "slurm_batch_id",
)


def _scopes(state):
    if not state.get("active_model_version") or not state.get("confirmed_config_version"):
        return
    groups = {}
    for task in state.get("tasks") or []:
        if (
            task.get("stage") != "deep_search"
            or task.get("status") not in ACTIVE
            or task.get("model_refresh_id")
            or task.get("model_version") != state.get("active_model_version")
            or task.get("config_version") != state.get("confirmed_config_version")
        ):
            continue
        key = (
            task.get("relax_mc_bundle_id"),
            task.get("slurm_batch_id"),
            task.get("upload_operation_id"),
        )
        groups.setdefault(key, []).append(task)
    for rows in groups.values():
        binding = [
            {
                "task_id": row["task_id"],
                "status": row["status"],
                **{k: row.get(k) for k in BINDING_FIELDS},
            }
            for row in sorted(rows, key=lambda row: row["task_id"])
        ]
        digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:20]
        yield "mc-wait-" + digest, rows


def cancellation_scopes(state):
    """Bounded model context; a scope binds every member, never a truncated subset."""
    return [
        {
            "scope_id": scope_id,
            "stage": "MC",
            "count": len(rows),
            "bundle_id": rows[0].get("relax_mc_bundle_id"),
            "batch_id": rows[0].get("slurm_batch_id"),
            "task_id_examples": [row["task_id"] for row in rows[:6]],
        }
        for scope_id, rows in list(_scopes(state))[-8:]
    ]


def latest_cancellation_facts(state):
    """Carry the user's local cancellation across restarts in a bounded factual view."""
    rows = [
        row
        for row in state.get("task_wait_cancellations") or []
        if (row.get("binding") or {}).get("model_version") == state.get("active_model_version")
        and (row.get("binding") or {}).get("config_version")
        == state.get("confirmed_config_version")
    ]
    if not rows:
        return {}
    latest = rows[-1]
    selected = [row for row in rows if row["scope_id"] == latest["scope_id"]]
    return {
        "scope_id": latest["scope_id"],
        "task_count": len(selected),
        "task_ids": [row["task_id"] for row in selected[:12]],
        "user_message": latest["user_message"][:240],
        "remote_cancel_sent": False,
        "instruction": "这些MC已明确停止本地等待。用已接受证据比较下一步，不要求补齐被取消任务；不标成功、不自动重投、不结算未知费用。新科学动作仍须具体方案审批。",
    }


def cancel_task_waits(state, scope_id, *, user_message):
    if not str(user_message or "").strip():
        raise ValueError("explicit_task_wait_cancellation_required")
    rows = next((rows for key, rows in _scopes(state) if key == scope_id), None)
    if not rows:
        raise ValueError("task_wait_scope_changed; refresh current tasks before cancelling")
    ids = {row["task_id"] for row in rows}
    current = deepcopy(state)
    journal = []
    for task in current.get("tasks") or []:
        if task.get("task_id") not in ids:
            continue
        journal.append(
            {
                "task_id": task["task_id"],
                "binding": {key: task.get(key) for key in BINDING_FIELDS},
                "previous_status": task["status"],
                "scope_id": scope_id,
                "user_message": str(user_message),
                "remote_cancel_sent": False,
            }
        )
        task.update(
            status="cancelled", recovery_cancelled=True, cancellation_reason=str(user_message)
        )
    current.setdefault("task_wait_cancellations", []).extend(journal)
    current["pending_tasks"] = active_pending_tasks(current.get("tasks") or [])
    return current, {row["task_id"]: row for row in journal}


def restore_cancelled_waits(state, cancellations):
    """Replay only administrative tombstones, never results or financial values."""
    rows = {row.get("task_id"): row for row in state.get("tasks") or []}
    for task_id, record in cancellations.items():
        task = rows.get(task_id)
        if (
            not task
            or task.get("status") not in ACTIVE | {"cancelled"}
            or record.get("task_id") != task_id
            or task.get("stage") != "deep_search"
            or task.get("model_refresh_id")
            or record.get("previous_status") not in ACTIVE
            or record.get("remote_cancel_sent") is not False
            or not record.get("user_message")
            or not str(record.get("scope_id") or "").startswith("mc-wait-")
            or set(record.get("binding") or {}) != set(BINDING_FIELDS)
            or any(task.get(key) != record["binding"][key] for key in BINDING_FIELDS)
        ):
            raise ValueError(
                "cancelled_task_wait_binding_changed_requires_reconciliation:" + task_id
            )
        task.update(
            status="cancelled", recovery_cancelled=True, cancellation_reason=record["user_message"]
        )
    if cancellations:
        state["pending_tasks"] = active_pending_tasks(state.get("tasks") or [])


def commit_task_wait_cancellation(state_path, state, scope_id, *, user_message):
    from phase_agent.tools.state.workflow_control import commit_control

    from pathlib import Path

    path = Path(state_path)
    raw = path.read_bytes()
    saved = json.loads(raw)
    # A turn may have read an earlier snapshot. Cancel only current, matching facts.
    if saved.get("workflow_control_revision") != state.get("workflow_control_revision"):
        raise ValueError("task_wait_scope_changed; reload current state")
    updated, cancellations = cancel_task_waits(saved, scope_id, user_message=user_message)
    return commit_control(
        path,
        saved,
        {
            "cancelled_waits": cancellations,
            "append": {"task_wait_cancellations": list(cancellations.values())},
        },
        evidence={
            "operation": "cancel_local_mc_waits",
            "user_message": str(user_message),
            "scope_id": scope_id,
            "task_ids": sorted(cancellations),
            "remote_cancel_sent": False,
        },
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        desired_state=updated,
    )

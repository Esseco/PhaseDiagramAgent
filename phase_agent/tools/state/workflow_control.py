"""Durable administrative revisions survive stale checkpoints without replaying science."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

FIELD = "workflow_control_revision"
VALUE_FIELDS = {"branch_count_policy", "branch_region_review", "workflow_stage", "dedup_gate"}
APPEND_FIELDS = {
    "generation_history",
    "reconciled_proposals",
    "recovery_review_history",
    "action_records",
    "task_wait_cancellations",
}


def control_path(state_path):
    path = Path(state_path)
    return path.with_name(path.name + ".control.json")


def read_control(state_path):
    path = control_path(state_path)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("state_path") != str(Path(state_path).resolve()) or not record.get("revision"):
        raise ValueError("invalid_workflow_control_owner_or_revision")
    return record


def current_revision(state_path):
    record = read_control(state_path) if state_path else None
    return record["revision"] if record else None


def restore_control(state_path, state):
    record = read_control(state_path)
    if (
        record
        and record.get("path_relocations")
        and state.get("path_relocation_revision") != record.get("path_relocation_revision")
    ):
        from phase_agent.configuration.session.relocate_workspace_paths import (
            relocate_workspace_paths,
        )

        state = relocate_workspace_paths(
            state,
            record["path_relocations"],
            Path(state_path).resolve().parent.parent,
            preserve_snapshots=True,
        )
        state["path_relocation_revision"] = record.get("path_relocation_revision")
    if not record or state.get(FIELD) == record["revision"]:
        return state
    if not record.get("restore_allowed", True):
        raise ValueError(
            "stale_business_state_after_result_or_financial_reconciliation; reload current verified state"
        )
    binding = record["binding"]
    if any(
        state.get(key) != binding.get(key)
        for key in ("confirmed_config_version", "active_model_version")
    ):
        raise ValueError("workflow_control_binding_changed_requires_reconciliation")
    updated = deepcopy(state)
    patch = record["patch"]
    values = patch.get("values") or {}
    if set(values) - VALUE_FIELDS or set(patch.get("append") or {}) - APPEND_FIELDS:
        raise ValueError("unsupported_workflow_control_field")
    updated.update(deepcopy(values))
    from phase_agent.tools.state.task_wait_cancellation import restore_cancelled_waits

    restore_cancelled_waits(updated, patch.get("cancelled_waits") or {})
    for name in ("execution_reconciliations", "invocations"):
        updated.setdefault(name, {}).update(deepcopy(patch.get(name) or {}))
    pending = updated.setdefault("pending_execution_policies", {})
    for key in patch.get("pending_remove") or []:
        pending.pop(key, None)
    pending.update(deepcopy(patch.get("pending_add") or {}))
    for key in patch.get("recoveries_remove") or []:
        updated.setdefault("pending_execution_recoveries", {}).pop(key, None)
    for name, rows in (patch.get("append") or {}).items():
        target = updated.setdefault(name, [])
        for row in rows:
            exists = (
                any(item.get("record_id") == row.get("record_id") for item in target)
                if name == "action_records"
                else row in target
            )
            if not exists:
                target.append(deepcopy(row))
    for row in updated.get("action_records") or []:
        if row.get("record_id") in (patch.get("record_statuses") or {}):
            row["status"] = patch["record_statuses"][row["record_id"]]
    for key, status in (patch.get("decision_statuses") or {}).items():
        if key in (updated.get("effective_decisions") or {}):
            updated["effective_decisions"][key]["status"] = status
    for key in patch.get("reservation_reconciled") or []:
        if key in (updated.get("budget_reservations") or {}):
            updated["budget_reservations"][key].pop("reconciliation_required", None)
    updated[FIELD] = record["revision"]
    return updated


def reconciliation_patch(before, after):
    """Only administrative deltas; never copy task results, model settings or costs."""
    return {
        "values": {
            k: deepcopy(after[k]) for k in VALUE_FIELDS if k in after and after[k] != before.get(k)
        },
        "execution_reconciliations": {
            k: v
            for k, v in (after.get("execution_reconciliations") or {}).items()
            if v != (before.get("execution_reconciliations") or {}).get(k)
        },
        "invocations": {
            k: v
            for k, v in (after.get("invocations") or {}).items()
            if v != (before.get("invocations") or {}).get(k)
        },
        "pending_remove": sorted(
            set(before.get("pending_execution_policies") or {})
            - set(after.get("pending_execution_policies") or {})
        ),
        "pending_add": {
            key: value
            for key, value in (after.get("pending_execution_policies") or {}).items()
            if value != (before.get("pending_execution_policies") or {}).get(key)
        },
        "recoveries_remove": sorted(
            set(before.get("pending_execution_recoveries") or {})
            - set(after.get("pending_execution_recoveries") or {})
        ),
        "append": {
            name: [
                v
                for v in after.get(name) or []
                if (
                    v.get("record_id") not in {r.get("record_id") for r in before.get(name) or []}
                    if name == "action_records"
                    else v not in (before.get(name) or [])
                )
            ]
            for name in APPEND_FIELDS
        },
        "record_statuses": {
            row["record_id"]: row["status"]
            for row in after.get("action_records") or []
            if row.get("record_id")
        },
        "decision_statuses": {
            key: row["status"]
            for key, row in (after.get("effective_decisions") or {}).items()
            if row.get("status")
        },
        "reservation_reconciled": [
            key
            for key, row in (before.get("budget_reservations") or {}).items()
            if row.get("reconciliation_required")
            and not (after.get("budget_reservations") or {})
            .get(key, {})
            .get("reconciliation_required")
        ],
    }


@contextmanager
def state_write_guard(state_path, state):
    """Serialize revision checks with administrative commits; stale writers fail closed."""
    if Path(state_path).name != "state.json" and not control_path(state_path).exists():
        yield
        return
    from filelock import FileLock

    control_path(state_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(control_path(state_path)) + ".lock", timeout=10):
        expected = current_revision(state_path)
        if expected and state.get(FIELD) != expected:
            raise ValueError("stale_workflow_control_revision; reload state before saving")
        yield


def commit_control(
    state_path, state, patch, *, evidence, expected_sha256, desired_state=None, restore_allowed=True
):
    """Explicit administrative repair only; caller verifies evidence and user authorization."""
    from filelock import FileLock
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = Path(state_path)
    with FileLock(str(control_path(path)) + ".lock", timeout=10):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("workflow_state_changed_before_control_commit")
        if not evidence:
            raise ValueError("workflow_control_requires_evidence")
        previous = read_control(path)
        if previous:
            patch = _merge_patch(previous["patch"], patch, state)
        record = {
            **(
                {
                    "path_relocations": previous["path_relocations"],
                    "path_relocation_revision": previous.get("path_relocation_revision"),
                }
                if previous and previous.get("path_relocations")
                else {}
            ),
            "schema_version": 1,
            "state_path": str(path.resolve()),
            "revision": uuid4().hex,
            "previous_revision": previous["revision"] if previous else None,
            "binding": {
                key: state.get(key) for key in ("confirmed_config_version", "active_model_version")
            },
            "patch": deepcopy(patch),
            "evidence": deepcopy(evidence),
            "restore_allowed": restore_allowed and (previous or {}).get("restore_allowed", True),
        }
        # Validate the patch before publishing its revision.
        if (
            set((patch.get("values") or {})) - VALUE_FIELDS
            or set(patch.get("append") or {}) - APPEND_FIELDS
        ):
            raise ValueError("unsupported_workflow_control_field")
        from phase_agent.tools.state.task_wait_cancellation import restore_cancelled_waits

        cancelled = patch.get("cancelled_waits") or {}
        restore_cancelled_waits(
            {
                "tasks": [
                    deepcopy(t) for t in state.get("tasks") or [] if t.get("task_id") in cancelled
                ]
            },
            cancelled,
        )
        write_json(control_path(path), record)
    if desired_state is not None:
        updated = deepcopy(desired_state)
        updated[FIELD] = record["revision"]
    else:
        updated = restore_control(path, state)
    write_json(path, updated)
    return updated


def _merge_patch(previous, incoming, state):
    merged = deepcopy(previous)
    for key in (
        "values",
        "execution_reconciliations",
        "invocations",
        "record_statuses",
        "decision_statuses",
        "cancelled_waits",
    ):
        merged.setdefault(key, {}).update(deepcopy(incoming.get(key) or {}))
    merged["values"] = {
        **merged.get("values", {}),
        **{k: deepcopy(state[k]) for k in merged.get("values", {}) if k in state},
        **deepcopy(incoming.get("values") or {}),
    }
    for key in ("pending_remove", "recoveries_remove", "reservation_reconciled"):
        merged[key] = sorted(set(previous.get(key) or []) | set(incoming.get(key) or []))
    pending = state.get("pending_execution_policies") or {}
    merged["pending_add"] = {
        k: deepcopy(pending[k]) for k in (previous.get("pending_add") or {}) if k in pending
    }
    merged["pending_add"].update(deepcopy(incoming.get("pending_add") or {}))
    for key in merged["pending_remove"]:
        merged["pending_add"].pop(key, None)
    for name, rows in (incoming.get("append") or {}).items():
        destination = merged.setdefault("append", {}).setdefault(name, [])
        destination.extend(deepcopy(row) for row in rows if row not in destination)
    return merged

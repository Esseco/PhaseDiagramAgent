"""Version-bound, rebuildable read-only views of the authoritative state.

The view is deliberately not a second source of scientific truth.  It contains
only the small fields needed by status, task-list and workspace-overview reads.
Business decisions and approvals continue to load and validate ``state.json``.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from phase_agent.persistence.state.read_versions import file_identity, read_identity


VIEW_SCHEMA_VERSION = 2
VIEW_NAME = "state.view.json"
_TASK_FIELDS = (
    "task_id",
    "task_key",
    "batch_id",
    "stage",
    "status",
    "model_version",
    "config_version",
)


def state_view_path(state_path):
    return Path(state_path).resolve().with_name(VIEW_NAME)


def _identity_payload(identity):
    if identity is None:
        return None
    return [list(row) if isinstance(row, tuple) else row for row in identity]


def _task_rows(state):
    tasks = state.get("tasks") or []
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    return [{key: row.get(key) for key in _TASK_FIELDS} for row in tasks if isinstance(row, dict)]


def _generation_batches(state):
    history = state.get("generation_history") or []
    retired = set()
    try:
        from phase_agent.tools.state.retired_generations import retired_task_keys

        retired = retired_task_keys(state)
    except (ImportError, TypeError, ValueError):
        pass
    active_model = state.get("active_model_version")
    rows = []
    for row in history:
        if not isinstance(row, dict):
            continue
        if (
            row.get("new_branch_ids")
            and row.get("task_key") not in retired
            and row.get("model_version") == active_model
        ):
            rows.append(
                {
                    "model_version": row.get("model_version"),
                    "task_key": row.get("task_key"),
                    "branch_count": len(row.get("new_branch_ids") or []),
                    "structure_count": len(row.get("registered_ids") or []),
                }
            )
    return rows[-20:]


def build_state_view(state: dict, state_path) -> dict:
    """Build a small projection without copying scientific histories."""
    tasks = _task_rows(state)
    pending = state.get("pending_execution_policies") or {}
    jobs = state.get("slurm_batches") or []
    if isinstance(jobs, dict):
        jobs = list(jobs.values())
    from collections import Counter

    status = {
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "hull_version": state.get("active_hull_version"),
        "task_counts": dict(Counter(row.get("status", "unknown") for row in tasks)),
        "task_stage_counts": dict(Counter(row.get("stage", "unknown") for row in tasks)),
        "job_counts": dict(
            Counter(row.get("status", "unknown") for row in jobs if isinstance(row, dict))
        ),
        "pending_task_ids": [row.get("task_id") for row in tasks if row.get("status") == "pending"],
        "failed_task_ids": [row.get("task_id") for row in tasks if row.get("status") == "failed"],
        "pending_approvals": len(pending),
        "reserved_relative_cost": float(state.get("reserved_relative_cost", 0) or 0),
        "budget_remaining": state.get("budget_remaining"),
        "coverage": deepcopy(state.get("coverage_summary")),
        "phase_diagram": deepcopy(state.get("phase_diagram_summary")),
        "qbc": deepcopy(state.get("qbc_summary")),
        "convergence": deepcopy(state.get("convergence_result")),
    }
    directions = []
    for job in (state.get("remote_finetune_jobs") or {}).values():
        handoff = job.get("training_handoff") or {}
        proposal = handoff.get("direction_proposal")
        if proposal:
            directions.append(
                {
                    "proposal": {key: proposal.get(key) for key in ("proposal_id", "reason")},
                    "status": handoff.get("direction_status"),
                    "waiting_at": handoff.get("stage"),
                    "activated": bool(job.get("activated")),
                }
            )
    try:
        from phase_agent.evidence.model_epoch import model_epoch

        epoch = model_epoch(state, state.get("active_model_version"))
    except (ImportError, TypeError, ValueError):
        epoch = None
    sources = {"state": _identity_payload(read_identity(state_path, strong=True))}
    control = Path(state_path).resolve().with_name(Path(state_path).name + ".control.json")
    sources["control"] = _identity_payload((file_identity(state_path), file_identity(control)))
    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "sources": sources,
        "workflow_control_revision": state.get("workflow_control_revision"),
        "confirmed_config_version": state.get("confirmed_config_version"),
        "active_model_version": state.get("active_model_version"),
        "active_model": deepcopy(state.get("active_model")),
        "epoch": epoch,
        "model_refresh": {
            key: (state.get("model_refresh") or {}).get(key)
            for key in ("status", "new_model_version", "old_model_version")
        },
        "snapshot_locator": {
            "field": "current_state_snapshot",
            "source": str(Path(state_path).resolve()),
        },
        "state_snapshot_index": state.get("state_snapshot_index"),
        "status": status,
        "task_rows": tasks,
        "directions": directions,
        "pending_execution_policies": {
            key: {name: row.get(name) for name in ("record_id", "status", "revision")}
            for key, row in pending.items()
        },
        "generation_batches": _generation_batches(state),
        "history_counts": {
            "tasks": len(tasks),
            **{
                key: None if hasattr(state, "fields") else len(state.get(key) or [])
                for key in ("action_records", "event_history", "invocations")
            },
        },
        "authority": "缓存视图；科学事实、审批和恢复必须读取并校验 state.json/ledger。",
    }


def _view_matches(view, state_path):
    if not isinstance(view, dict) or view.get("schema_version") != VIEW_SCHEMA_VERSION:
        return False
    expected = _identity_payload(read_identity(state_path, strong=True))
    return (view.get("sources") or {}).get("state") == expected


def publish_state_view(state_path, state: dict, *, expected_identity=None) -> str:
    """Publish a view only after the source identity is stable."""
    from phase_agent.tools.step_runner.file_protocol import _write_json

    path = Path(state_path).resolve()
    before = read_identity(path, strong=True)
    if (
        expected_identity is not None
        and (before if len(expected_identity) == 3 else before[:2]) != expected_identity
    ):
        raise ValueError("state_changed_before_view_publish")
    view = build_state_view(state, path)
    if read_identity(path, strong=True) != before:
        raise ValueError("state_changed_during_view_publish")
    target = state_view_path(path)
    _write_json(target, view)
    return str(target)


def read_state_view(state_path, *, rebuild=True, publish=True, state_loader=None) -> dict:
    """Read a fresh view, rebuilding it from full state when necessary."""
    from phase_agent.runtime.turn_process import record_read_metric

    path = Path(state_path).resolve()
    target = state_view_path(path)
    try:
        view = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        view = None
    if _view_matches(view, path):
        record_read_metric("state_view_hit")
        return {**view, "cache_status": "hit"}
    record_read_metric(
        "state_view_miss",
        reason="missing_or_corrupt" if view is None else "schema_or_source_changed",
    )
    if not rebuild:
        raise ValueError("state_view_missing_or_stale")
    if state_loader is None:
        from phase_agent.tools.step_runner.file_protocol import read_json

        state_loader = read_json
    before = read_identity(path, strong=True)
    from phase_agent.persistence.state.field_index import open_indexed_state, IndexUnavailable

    state = open_indexed_state(path)
    view = None
    if state is not None:
        try:
            view = build_state_view(state, path)
            state.verify(strong=True)
        except IndexUnavailable:
            state = None
    if state is None:
        state = state_loader(path, {}) or {}
        view = build_state_view(state, path)
    if read_identity(path, strong=True) != before:
        raise ValueError("state_changed_during_view_build")
    if read_identity(path, strong=True) != before:
        raise ValueError("state_changed_during_view_build")
    if publish:
        try:
            publish_state_view(path, state, expected_identity=before)
            status = "rebuilt"
        except (OSError, ValueError, TypeError):
            status = "computed_read_only"
    else:
        status = "computed_read_only"
    if read_identity(path, strong=True) != before:
        raise ValueError("state_changed_during_view_build")
    return {**view, "cache_status": status}

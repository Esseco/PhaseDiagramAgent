"""Small, disposable project overview bound to the underlying files."""

from contextlib import nullcontext
import json
from pathlib import Path

from filelock import FileLock

from phase_agent.tools.state.workflow_control import control_path
from phase_agent.tools.step_runner.file_protocol import read_json, _write_json


def _identity(path):
    if path is None:
        return None
    try:
        p = Path(path).resolve()
        stat = p.stat()
        return {
            "path": str(p),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
        }
    except FileNotFoundError:
        return {"path": str(Path(path).resolve()), "missing": True}


def source_identities(state_path, config_path, ledger_path):
    return {
        "state": _identity(state_path),
        "control": _identity(control_path(state_path)),
        "config": _identity(config_path),
        "ledger": _identity(ledger_path),
    }


def project_overview(state_path, config_path, ledger_path, *, publish=True):
    """Return a version-checked overview; never use this index as scientific authority."""
    from phase_agent.runtime.turn_process import record_read_metric

    state_path = Path(state_path).resolve()
    index_path = state_path.parent / "context_overview.json"
    lock_path = state_path.parent / "langgraph_lifecycle.sqlite.lock"
    lock = FileLock(str(lock_path), timeout=30) if publish else nullcontext()
    with lock:
        before = source_identities(state_path, config_path, ledger_path)
        try:
            saved = json.loads(index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            saved = None
        if (
            isinstance(saved, dict)
            and saved.get("schema_version") == 3
            and saved.get("sources") == before
        ):
            if source_identities(state_path, config_path, ledger_path) != before:
                raise ValueError("overview_sources_changed_during_read")
            record_read_metric("overview_hit")
            return {**saved, "cache_status": "hit"}
        record_read_metric(
            "overview_miss",
            reason="missing_or_corrupt" if saved is None else "schema_or_sources_changed",
        )
        session = read_json(config_path, {}) if config_path else {}
        session = session or {}
        from phase_agent.persistence.state.state_view import read_state_view

        view = read_state_view(state_path, publish=publish, state_loader=read_json)
        status = view.get("status") or {}

        result = {
            "schema_version": 3,
            "sources": before,
            "epoch": view.get("epoch"),
            "config_version": view.get("confirmed_config_version")
            or (session.get("confirmed_snapshot") or {}).get("config_version"),
            "model_version": view.get("active_model_version"),
            "workflow_control_revision": view.get("workflow_control_revision"),
            "task_counts": status.get("task_counts") or {},
            "pending_approvals": status.get("pending_approvals", 0),
            "generation_batches": view.get("generation_batches") or [],
            "generation_note": "Registered batches for current model; counts may overlap and do not imply completed search.",
            "cache_status": view.get("cache_status"),
            "detail_sources": {
                "state": str(state_path),
                "ledger": str(Path(ledger_path).resolve()) if ledger_path else None,
                "analysis": str(state_path.parent.parent / "analysis_outputs"),
                "tasks": {"transport": "HTTP", "entry": "/phase/tasks", "source_field": "tasks"},
                "pending": {"entry": "phase_pending", "source_field": "pending_execution_policies"},
                "decision": {"entry": "phase_context", "authority": "verified_state_and_ledger"},
                "team_resume": {"entry": "phase_team_read", "include_history": False},
            },
            "authority": "Overview only; decisions and approvals require current verified source records.",
        }
        after = source_identities(state_path, config_path, ledger_path)
        if before != after:
            raise ValueError("overview_sources_changed_during_build")
        if publish:
            _write_json(index_path, result)
        return {**result, "cache_status": "rebuilt" if publish else "computed_read_only"}

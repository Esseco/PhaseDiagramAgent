"""Project-lock boundary for an explicitly approved refresh handoff repair."""

from pathlib import Path
import re
from filelock import FileLock

from phase_agent.persistence.ledger.phase_data_manager import _ledger_identity, refresh_if_changed
from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version


def _saved_context(control, state, runtime_config_path):
    """Load only saved evidence; never compose the relocating scientific runtime."""
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.runtime.project_context import current_config_session
    from phase_agent.runtime.runtime_config_io import (
        _load_json_object,
        _resolve_path,
        runtime_base_directory,
    )
    from phase_agent.tools.remote.integrity import file_checksum

    runtime = Path(runtime_config_path).resolve()
    runtime_digest = file_checksum(runtime)
    session_path = Path(control.chat_handler.config_session_path)
    session_digest = file_checksum(session_path)
    settings = _load_json_object(runtime, "runtime config")
    base = runtime_base_directory(runtime, settings)
    state_path = _resolve_path(settings["state_path"], base)
    ledger = _resolve_path(settings["ledger_path"], base)
    if state_path.resolve() != Path(control.state_path).resolve() or not ledger.is_file():
        raise ValueError("refresh_repair_saved_project_binding_invalid")
    session = current_config_session(control.chat_handler)
    snapshot = session.get("confirmed_snapshot") or {}
    if session.get("status") != "confirmed" or snapshot.get("config_version") != state.get(
        "confirmed_config_version"
    ):
        raise ValueError("refresh_repair_confirmed_config_changed")
    config = snapshot.get("config") or {}
    if settings.get("local_path_relocations"):
        config = relocate_workspace_paths(config, settings["local_path_relocations"], base)
    reference_file = settings.get("phase_references_path")
    reference_file = _resolve_path(reference_file, base) if reference_file else None
    reference_digest = (
        file_checksum(reference_file) if reference_file and reference_file.is_file() else None
    )
    references = (
        _load_json_object(reference_file, "phase references")
        if reference_digest
        else (config.get("system") or {}).get("phase_references") or {}
    )
    if not isinstance(references, dict):
        raise ValueError("refresh_repair_phase_references_invalid")
    references = {
        key: str(_resolve_path(value, base)) if isinstance(value, str) else value
        for key, value in references.items()
    }
    paths = [runtime, session_path]
    if reference_file and reference_file.is_file():
        paths.append(reference_file)
    paths.extend(Path(v) for v in references.values() if isinstance(v, str))
    watched = {str(p): file_checksum(p) for p in paths}
    if (
        watched[str(runtime)] != runtime_digest
        or watched[str(session_path)] != session_digest
        or (reference_digest and watched[str(reference_file)] != reference_digest)
    ):
        raise ValueError("refresh_repair_config_changed_during_read")
    ledger_version = _ledger_identity(ledger)
    manager = PhaseDataManager.load(ledger)
    if _ledger_identity(ledger) != ledger_version:
        raise ValueError("refresh_repair_ledger_changed_during_read")
    expected = (config.get("system") or {}).get("boundary")
    if expected is not None:
        from phase_agent.science.structures.boundary_utils import json_safe

        if json_safe(manager.boundary) != json_safe(expected):
            raise ValueError("refresh_repair_boundary_changed")
    return manager, ledger, references, watched, ledger_version


def repair(control, source_sha256, task_count, *, runtime_config_path=None):
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("refresh_repair_sha256_required")
    if type(task_count) is not int or not 1 <= task_count <= 100000:
        raise ValueError("refresh_repair_invalid_task_count")
    from phase_agent.configuration.defaults.default_slurm_cluster_config import (
        default_slurm_cluster_config,
    )
    from phase_agent.tools.remote.integrity import file_checksum
    from phase_agent.tools.remote.refresh_handoff_repair import rebuild_refresh_handoff
    from phase_agent.tools.state.execution_receipts import recovery_report

    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        version = read_identity(path, strong=True)
        state = read_json(path, {}) or {}
        if state.get("pending_execution_policies") or recovery_report(path, state)["unsettled"]:
            raise ValueError("refresh_repair_pending_or_unsettled_action")
        kwargs = control.chat_handler.workflow_kwargs
        watched = {}
        if runtime_config_path:
            manager, ledger, references, watched, ledger_version = _saved_context(
                control, state, runtime_config_path
            )
        else:
            manager = kwargs.get("manager")
            ledger = getattr(manager, "_ledger_path", None)
            if manager is None or not ledger:
                raise ValueError("refresh_repair_ledger_unavailable")
            refresh_if_changed(manager, ledger)
            references = kwargs.get("phase_references")
            ledger_version = _ledger_identity(ledger)
        result = rebuild_refresh_handoff(
            state,
            path.parent.parent,
            source_sha256,
            task_count,
            manager,
            references,
            default_slurm_cluster_config(),
        )
        if _ledger_identity(ledger) != ledger_version:
            raise ValueError("refresh_repair_ledger_changed_during_preparation")
        if any(file_checksum(name) != digest for name, digest in watched.items()):
            raise ValueError("refresh_repair_config_or_reference_changed_during_preparation")
        if result["status"] == "prepared":
            write_json_if_version(path, state, version)
            control._codex_context_cache = None
        elif read_identity(path, strong=True) != version:
            raise ValueError("stale_state_file_version; reload state before reporting")
        return result

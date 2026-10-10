"""Resolve the user-confirmed local workspace and its generated-data paths."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from phase_agent.configuration.session.validate_workspace_root import validate_workspace_root


DEFAULT_WORKSPACE_PATHS = {
    "config_snapshots": "parameters/snapshots",
    "state": "workflow_state/state.json",
    "ledger": "workflow_state/ledgers/phase_data.json",
    "branch_energy_pool_ledger": "workflow_state/ledgers/branch_energy_pools.json",
    "structures": "structures/candidate_structures",
    "phase_diagrams": "analysis_outputs",
    "qbc_results": "workflow_state/cache/qbc/qbc_results.json",
    "work": "workflow_state/work",
    "approvals": "workflow_state/approvals",
    "approved_batches": "workflow_state/approved_batches",
    "upload_batches": "submissions",
    "new_runs": "open_webui_runs",
}


def default_workspace_storage(workspace_root, *, path_overrides=None, base_directory=None) -> dict:
    """Return the editable storage section; paths are relative to one root."""
    root = Path(validate_workspace_root(str(workspace_root))).resolve()
    paths = deepcopy(DEFAULT_WORKSPACE_PATHS)
    for key, configured in (path_overrides or {}).items():
        if key not in paths or not configured:
            continue
        target = Path(configured).expanduser()
        if not target.is_absolute() and base_directory is not None:
            target = Path(base_directory) / target
        target = target.resolve()
        try:
            paths[key] = target.relative_to(root).as_posix() or "."
        except ValueError as error:
            raise ValueError(f"默认路径 {key} 位于统一工作区之外：{target}") from error
    return {
        "workspace_root": str(root),
        "paths": paths,
    }


def resolve_workspace_paths(config: dict, *, base_directory) -> dict[str, Path]:
    """Resolve every output path, rejecting malformed or escaping relative paths."""
    storage = config.get("storage") or {}
    if not isinstance(storage, dict):
        raise ValueError("storage 必须是 JSON 对象")
    unknown_storage = set(storage) - {"workspace_root", "paths"}
    if unknown_storage:
        raise ValueError("storage 中包含未知字段：" + ", ".join(sorted(unknown_storage)))
    root_value = storage.get("workspace_root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("storage.workspace_root 必须是非空路径")
    root = Path(validate_workspace_root(root_value)).expanduser()
    if not root.is_absolute():
        root = Path(base_directory) / root
    root = root.resolve()

    configured_paths = storage.get("paths") or {}
    if not isinstance(configured_paths, dict):
        raise ValueError("storage.paths 必须是 JSON 对象")
    unknown = set(configured_paths) - set(DEFAULT_WORKSPACE_PATHS)
    missing = set(DEFAULT_WORKSPACE_PATHS) - set(configured_paths)
    if unknown or missing:
        details = []
        if missing:
            details.append("缺少 " + ", ".join(sorted(missing)))
        if unknown:
            details.append("未知 " + ", ".join(sorted(unknown)))
        raise ValueError("storage.paths 字段不匹配：" + "；".join(details))

    resolved = {}
    for key, value in configured_paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"storage.paths.{key} 必须是非空相对路径")
        path = Path(value).expanduser()
        if path.is_absolute():
            raise ValueError(f"storage.paths.{key} 必须相对于 workspace_root")
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(f"storage.paths.{key} 不能越出 workspace_root") from error
        resolved[key] = target
    resolved["workspace_root"] = root
    return resolved

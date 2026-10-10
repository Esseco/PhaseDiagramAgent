"""Rename local transport directories with byte checks and rollback, not analysis."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil

from phase_agent.configuration.session.workspace_directory_names import DIRECTORY_RENAMES
from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage
from phase_agent.configuration.session.workspace_active_files import workspace_active_files
from phase_agent.tools.local.migrate_workspace_layout import (
    require_agent_offline,
    rewrite_active_paths,
    _read,
    _file_hash,
)
from phase_agent.tools.local.workspace_csv_metadata import rewrite_csv_metadata
from phase_agent.tools.remote.migrate_legacy_upload_layout import _replace_paths, _atomic_write


def rename_workspace_directories(workspace, *, apply=False, agent_port=8765):
    root = Path(workspace).resolve()
    moves = [
        (root / old, root / new) for old, new in DIRECTORY_RENAMES.items() if (root / old).exists()
    ]
    if not moves:
        return {"status": "unchanged"}
    for source, target in moves:
        if source.is_symlink() or target.exists() or not target.resolve().is_relative_to(root):
            raise ValueError(f"目标存在或路径不安全：{source} -> {target}")
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError(f"源目录含链接：{source}")
    state_path = workspace_active_files(root)["state"]
    initial_state = state_path.read_bytes()
    replacements = [(str(a), str(b)) for a, b in moves]
    # Relative replacements are used only for explicit descriptor bindings.
    relative = [(str(a.relative_to(root)), str(b.relative_to(root))) for a, b in moves]
    storage = default_workspace_storage(root)
    edits = []
    paths = [root / "agent_runtime.json"]
    for folder in (
        "config",
        "runtime",
        "memory",
        "inputs",
        "InitFile",
        "outputs",
        "upload_batches",
    ):
        base = root / folder
        paths.extend(
            p
            for p in base.rglob("*.json")
            if not {"snapshots", "approvals"}.intersection(p.relative_to(base).parts)
        )
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        before = _read(path)
        after = rewrite_active_paths(before, replacements)
        if path.name == "config_session.json" and before.get("status") == "confirmed":
            after["config"] = deepcopy(before.get("config"))
        if path.name == "search_config.project.json":
            after.setdefault("config", {})["storage"] = storage
        if path.name == "agent_runtime.json":
            for key, value in after.items():
                if isinstance(value, str) and (key.endswith("_path") or key.endswith("_directory")):
                    after[key] = _replace_paths(value, relative)
            after["runtime_storage_override"] = storage
            after["local_path_relocations"] = [{"from": a, "to": b} for a, b in replacements]
        if path == state_path:
            after["workspace_layout"] = {
                "version": 3,
                "workspace_root": str(root),
                "state_path": str(root / "workflow_state/state.json"),
                "memory_directory": str(root / "agent_memory"),
                "submission_plan_path": str(
                    root / "workflow_state/submission_plans/RELAX_UPLOAD_PLAN.json"
                ),
            }
        if before != after:
            edits.append((path, after, path.read_bytes()))
    for path in (root / "outputs").rglob("*.csv"):
        original = path.read_bytes()
        updated = rewrite_csv_metadata(path, replacements)
        if updated != original:
            edits.append((path, updated, original))
    result = {
        "status": "planned",
        "moves": [{"from": str(a), "to": str(b)} for a, b in moves],
        "updated_metadata": len(edits),
    }
    if not apply:
        return result
    require_agent_offline(agent_port)
    backup = root / "history_backups/directory_names" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    # Do not create a destination of a pending move before that move completes.
    hashes = {
        p: _file_hash(p)
        for source, _ in moves
        for p in source.rglob("*")
        if p.is_file() and source.name != "backups"
    }
    if state_path.read_bytes() != initial_state or any(
        p.read_bytes() != data for p, _, data in edits
    ):
        raise RuntimeError("元数据已变化，未移动目录")
    require_agent_offline(agent_port)
    moved, written = [], []
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
        for source, digest in hashes.items():
            if _file_hash(Path(_replace_paths(str(source), replacements))) != digest:
                raise RuntimeError(f"文件内容校验失败：{source}")
        backup.mkdir(parents=True)
        for source, payload, original in edits:
            saved = backup / source.relative_to(root)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(original)
            target = Path(_replace_paths(str(source), replacements))
            written.append((target, original))
            if isinstance(payload, bytes):
                from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

                _publish_bytes(target, payload)
            else:
                _atomic_write(target, payload)
        new_state_path = root / "workflow_state/state.json"
        state = _read(new_state_path)
        from phase_agent.analysis.feedback.export_dft_products import publish_output_catalog
        from phase_agent.analysis.feedback.workspace_guide import publish_workspace_guide
        from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

        publish_output_catalog(state, root / "analysis_outputs")
        publish_memory_views(state, new_state_path)
        publish_workspace_guide(root)
        _atomic_write(new_state_path, state)
    except Exception:
        for path, original in reversed(written):
            path.write_bytes(original)
        for source, target in reversed(moved):
            shutil.move(str(target), str(source))
        raise
    result.update(status="renamed", verified_files=len(hashes), backup=str(backup))
    _atomic_write(backup / "manifest.json", result)
    return result

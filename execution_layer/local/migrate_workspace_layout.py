"""Recoverable directory-only migration with frozen scientific configuration intact."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket

from config_layer.session.resolve_workspace_paths import default_workspace_storage
from execution_layer.local.workspace_layout_plan import workspace_layout_plan
from execution_layer.remote.migrate_legacy_upload_layout import _replace_paths, _atomic_write


FROZEN = {"confirmed_config", "confirmed_snapshot", "source_config_snapshot"}


def rewrite_active_paths(value, replacements):
    forms = [(old_form, old_form.lower(), new_form)
             for old, new in replacements
             for old_form, new_form in ((old, new), (Path(old).as_posix(), Path(new).as_posix()))]

    def rewrite(item):
        if isinstance(item, dict):
            return {key: deepcopy(child) if key in FROZEN else rewrite(child) for key, child in item.items()}
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        if not isinstance(item, str):
            return item
        lowered = item.lower()
        separator = "/" if "/" in item else "\\"
        for old, normalized, new in forms:
            if lowered == normalized:
                return new
            prefix = old.rstrip("/\\")
            if lowered.startswith(prefix.lower() + separator):
                return new.rstrip("/\\") + item[len(prefix):]
        return item
    return rewrite(value)


def _read(path):
    from config_layer.session.load_editable_config_json import _strip_jsonc_comments
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_strip_jsonc_comments(text))


def _metadata_paths(root, moves):
    paths = [root / "current/state.json", root / "runtime/state.json", root / "config_session.json",
             root / "config/config_session.json", root / "search_config.project.json",
             root / "config/search_config.project.json", root / "agent_runtime.json"]
    for source, destination in moves:
        if destination.is_relative_to(root / "backups"):
            continue
        if source.is_file() and source.suffix == ".json" and "backups" not in source.parts:
            paths.append(source)
        elif source.is_dir() and "config_snapshots" not in source.parts and "approvals" not in source.parts:
            paths.extend(source.rglob("*.json"))
    # Never edit VASP/MLIP returns, upload manifests or historical approvals.
    return list(dict.fromkeys(path for path in paths if path.is_file()
        and "backups" not in path.parts and not path.name.startswith("state.before")
        and not path.name.endswith(".bak")))


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_agent_offline(port):
    if port is None:
        return
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex(("127.0.0.1", int(port))) == 0:
            raise RuntimeError(f"本地{port}端口服务仍在运行；请先关闭Agent启动窗口，再迁移目录。未移动文件。")


def migrate_workspace_layout(workspace, *, dry_run=False, agent_port=8765):
    if not dry_run:
        require_agent_offline(agent_port)
    root = Path(workspace).resolve()
    state_path = root / "runtime/state.json"
    if not state_path.is_file():
        state_path = root / "current/state.json"
    original_state = state_path.read_bytes()
    state = json.loads(original_state)
    if any(row.get("status") in {"running", "submitted", "submitting"}
           for row in state.get("slurm_batches") or []):
        raise ValueError("有登记中的活动提交作业；先停止本地目录变更并核对作业")
    moves = workspace_layout_plan(root, state)
    result = {"moves": [{"source": str(a), "target": str(b)} for a, b in moves],
              "missing_training_inputs": [job["directory"] for job in (state.get("remote_finetune_jobs") or {}).values()
                                          if not Path(job["directory"]).is_dir()]}
    if dry_run:
        return {"status": "planned", **result}
    replacements = sorted([(str(a), str(b)) for a, b in moves], key=lambda row: -len(row[0]))
    replacements += [(str(a.relative_to(root)), str(b.relative_to(root))) for a, b in moves]
    edits = []
    storage = default_workspace_storage(root)
    # This legacy migrator deliberately produces layout v2. The separate named
    # directory migration upgrades v2 without mixing two move transactions.
    from config_layer.session.workspace_directory_names import DIRECTORY_RENAMES
    inverse = [(new, old) for old, new in DIRECTORY_RENAMES.items() if old != "InitFile"]
    storage["paths"] = {key: _replace_paths(value, inverse) for key, value in storage["paths"].items()}
    settings_path = root / "agent_runtime.json"
    for path in _metadata_paths(root, moves):
        original = _read(path)
        changed = rewrite_active_paths(original, replacements)
        if path.name == "config_session.json" and original.get("status") == "confirmed":
            changed["config"] = deepcopy(original.get("config"))
        if path == settings_path:
            changed.update(config_session_path="config/config_session.json",
                           editable_config_draft_path="config/search_config.project.json",
                           state_path="runtime/state.json", ledger_path="runtime/ledgers/phase_data.json",
                           branch_energy_pool_ledger_path="runtime/ledgers/branch_energy_pools.json",
                           structure_directory="inputs/candidate_structures",
                           phase_diagram_directory="outputs", approval_directory="runtime/approvals",
                           local_action_directory="runtime/approved_batches",
                           runtime_storage_override=storage)
        if path.name == "search_config.project.json":
            changed.setdefault("config", {})["storage"] = storage
        if path == state_path:
            changed["workspace_layout"] = {"version": 2, "workspace_root": str(root),
                "state_path": str(root / "runtime/state.json"), "memory_directory": str(root / "memory"),
                "submission_plan_path": str(root / "runtime/submission_plans/RELAX_UPLOAD_PLAN.json")}
        if changed != original:
            edits.append((path, changed, path.read_bytes()))
    from analysis_layer.phase.phase_snapshot_paths import model_output_directory
    from analysis_layer.state.model_epoch import model_epoch
    from execution_layer.local.workspace_csv_metadata import rewrite_csv_metadata
    epoch_roots = [(model_output_directory(root / "outputs", version, state=state), model_epoch(state, version))
                   for version in (state.get("upload_layout") or {}).get("model_rounds") or {}]
    for source, destination in moves:
        if not source.is_relative_to(root / "outputs") or destination.is_relative_to(root / "backups"):
            continue
        for path in ([source] if source.is_file() else source.rglob("*.csv")):
            if path.suffix != ".csv":
                continue
            target = Path(_replace_paths(str(path), replacements))
            epoch = next((label for base, label in epoch_roots if target.is_relative_to(base)), None)
            updated = rewrite_csv_metadata(path, replacements, epoch)
            if updated != path.read_bytes():
                edits.append((path, updated, path.read_bytes()))
    # Hash only moved files; scientific values are never inferred here.
    files = {path: _file_hash(path) for source, _ in moves
             for path in ([source] if source.is_file() else source.rglob("*")) if path.is_file()}
    backup = root / "backups/workspace_layout" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup.mkdir(parents=True)
    for path, _, _ in edits:
        destination = backup / "metadata" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    moved = []
    written = []
    try:
        require_agent_offline(agent_port)
        if state_path.read_bytes() != original_state:
            raise RuntimeError("Agent正在修改state，请关闭Agent后重试")
        if any(path.read_bytes() != original for path, _, original in edits):
            raise RuntimeError("待迁移元数据已被修改；未移动文件，请停止并发编辑后重试")
        for source, target in moves:
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
        for source, digest in files.items():
            target = Path(_replace_paths(str(source), replacements))
            if _file_hash(target) != digest:
                raise RuntimeError(f"移动后的文件内容不一致：{target}")
        for source, payload, _ in edits:
            written.append(source)
            target = Path(_replace_paths(str(source), replacements))
            if isinstance(payload, bytes):
                from analysis_layer.feedback.export_dft_products import _publish_bytes
                _publish_bytes(target, payload)
            else:
                _atomic_write(target, payload)
    except Exception:
        for source, target in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(source))
        for path, _, original in edits:
            if path in written:
                path.write_bytes(original)
        raise
    # Remove empty shells only; no recursive deletion or removal of file data.
    for base in (root / "current", root / "outputs"):
        if base.is_dir():
            for folder in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            if base.name == "current" and not any(base.iterdir()):
                base.rmdir()
    current = _read(root / "runtime/state.json")
    from analysis_layer.feedback.export_dft_products import publish_output_catalog
    from data_layer.memory.publish_memory_views import publish_memory_views
    publish_output_catalog(current, root / "outputs")
    publish_memory_views(current, root / "runtime/state.json")
    from analysis_layer.feedback.workspace_guide import publish_workspace_guide
    publish_workspace_guide(root)
    _atomic_write(root / "runtime/state.json", current)
    result.update(status="organized", backup=str(backup), verified_moved_files=len(files),
                  updated_metadata_files=len(edits))
    _atomic_write(backup / "migration.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent-port", type=int, default=8765)
    args = parser.parse_args()
    print(json.dumps(migrate_workspace_layout(args.workspace, dry_run=args.dry_run,
        agent_port=args.agent_port), ensure_ascii=False, indent=2))

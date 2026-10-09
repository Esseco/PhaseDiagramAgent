"""Recoverable local-only epoch migration; never rewrite remote paths or history."""
import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from execution_layer.remote.migrate_legacy_upload_layout import _replace_paths, _atomic_write
from config_layer.session.workspace_active_files import workspace_active_files


def migrate(workspace, *, dry_run=False):
    workspace = Path(workspace).resolve()
    root = workspace / "upload_batches"
    active_files = workspace_active_files(workspace)
    state_path = active_files["state"]
    initial_state = state_path.read_bytes()
    state = json.loads(initial_state.decode("utf-8"))
    if any(b.get("status") in {"running", "submitting", "submitted"} for b in state.get("slurm_batches", [])):
        raise ValueError("有已登记活动远端批次，不能直接迁移")
    replacements, moves = [], []
    for source in sorted(root.glob("MLIP-round-*")):
        match = re.fullmatch(r"MLIP-round-(\d+)_(.+)", source.name)
        if not match:
            continue
        target = root / f"epoch{int(match[1])-1}_{match[2]}"
        replacements.append((str(source), str(target)))
        for child in source.iterdir():
            moves.append((child, target / child.name))
    counts = {}
    for key, job in sorted((state.get("remote_finetune_jobs") or {}).items()):
        version = job["original_model_version"]
        epoch = int(state["upload_layout"]["model_rounds"][version])-1
        counts[version] = counts.get(version, 0)+1
        source = Path(job["directory"])
        target = root / f"epoch{epoch}_{version}" / f"MLIP-finetune-round-{counts[version]:04d}"
        if source == target:
            continue
        replacements.append((str(source), str(target)))
        if source.exists():
            moves.append((source, target))
    for source, target in moves:
        if not source.resolve().is_relative_to(root.resolve()) or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("移动路径超出upload_batches")
        if target.exists():
            raise FileExistsError(target)
    result = {"moves": [{"source": str(s), "target": str(t)} for s, t in moves],
              "missing_training": [job["directory"] for job in (state.get("remote_finetune_jobs") or {}).values()
                                   if not Path(job["directory"]).exists()]}
    if dry_run:
        return result
    backup = workspace / "backups" / ("epoch-layout-"+datetime.now().strftime("%Y%m%dT%H%M%S%f"))
    backup.mkdir(parents=True)
    files = [state_path, root / "RELAX_UPLOAD_PLAN.json", active_files["ledger"], active_files["branch_pool"]]
    files += list((workspace / "config").glob("*.json"))
    files += list(workspace.glob("*.json"))
    files += [p for source, target in moves for p in source.rglob("*.json") if source.is_dir()]
    edits = []
    for path in dict.fromkeys(files):
        if not path.is_file():
            continue
        original = json.loads(path.read_text(encoding="utf-8"))
        updated = _replace_paths(original, replacements)
        if original != updated:
            edits.append((path, updated))
            saved = backup / path.relative_to(workspace)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    moved = []
    try:
        if state_path.read_bytes() != initial_state:
            raise RuntimeError("Agent正在修改state，请空闲后重试迁移")
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
        for path, payload in edits:
            new_path = Path(_replace_paths(str(path), replacements))
            _atomic_write(new_path, payload)
    except Exception:
        for source, target in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(source))
        for saved in backup.rglob("*.json"):
            shutil.copy2(saved, workspace / saved.relative_to(backup))
        raise
    for old in root.glob("MLIP-round-*"):
        if old.is_dir() and not any(old.iterdir()):
            old.rmdir()
    result["backup"] = str(backup)
    _atomic_write(backup / "migration.json", result)
    return result


def repair_active_metadata(workspace, migration_path):
    workspace = Path(workspace).resolve()
    migration = json.loads(Path(migration_path).read_text(encoding="utf-8"))
    replacements = [(row["source"], row["target"]) for row in migration["moves"]]
    # Include model-root prefixes, not just moved children.
    replacements += [(str(Path(old).parent), str(Path(new).parent)) for old, new in replacements
                     if Path(old).parent.name.startswith("MLIP-round-")]
    backup = Path(migration["backup"])
    changed = []
    active_files = workspace_active_files(workspace)
    for key in ("ledger", "branch_pool"):
        path = active_files[key]
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = _replace_paths(payload, replacements)
        if updated != payload:
            saved = backup / path.relative_to(workspace)
            saved.parent.mkdir(parents=True, exist_ok=True)
            if not saved.exists():
                shutil.copy2(path, saved)
            _atomic_write(path, updated)
            changed.append(path.name)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair-metadata", default=None)
    args = parser.parse_args()
    result = repair_active_metadata(args.workspace, args.repair_metadata) if args.repair_metadata else migrate(args.workspace, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))

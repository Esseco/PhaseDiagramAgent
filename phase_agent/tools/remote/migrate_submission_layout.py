"""Reversible local layout migration without rewriting scientific payloads."""

from datetime import datetime
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil

from phase_agent.configuration.session.relocate_workspace_paths import (
    relocate_workspace_paths,
)
from phase_agent.tools.local.organize_epoch_folders import (
    repair_archive_index,
    relocate_archive_index,
)
from phase_agent.tools.remote.submission_layout import (
    submission_bucket_moves,
    submission_cleanup_moves,
    submission_moves,
    write_submission_index,
)

OPERATIONAL_KEYS = {
    "tasks",
    "slurm_batches",
    "phase_records",
    "phase_identification_cache",
    "branch_candidates",
    "remote_finetune_jobs",
    "relax_mc_batches",
    "result_collection",
    "result_recoveries",
    "filesystem_migrations",
    "model_registry",
    "candidate_models",
    "dft_dataset_records",
    "dft_training_records",
    "new_dft_records",
    "qbc_candidates",
    "phase_diagrams",
    "phase_diagrams_by_model",
    "current_branch_hull_by_model",
}
IMMUTABLE_NAMES = {
    "task.json",
    "manifest.json",
    "batch.snapshot.json",
    "result.json",
    "task.finished.json",
}


def inventory(directory):
    root = Path(directory)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and (p.name in IMMUTABLE_NAMES or p.suffix == ".zip")
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".layout.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def migrate_submission_layout(project, backup_directory):
    project = Path(project).resolve()
    root = project / "submissions"
    state_path = project / "workflow_state/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    moves = submission_moves(root, state)
    moves.extend(submission_bucket_moves(root, state))
    moves.extend(submission_cleanup_moves(root, state))
    if not moves:
        repair_archive_index(project)
        write_submission_index(root, state)
        return {"project": project.name, "status": "already_compact", "moves": []}
    backup = Path(backup_directory) / project.name
    if backup.exists():
        backup = backup / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup.mkdir(parents=True, exist_ok=False)
    ledger_path = project / "workflow_state/ledgers/phase_data.json"
    files = [p for p in [state_path, ledger_path, root / "RELAX_UPLOAD_PLAN.json"] if p.is_file()]
    saved = {}
    for path in files:
        relative = path.relative_to(project)
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        saved[path] = destination
    archive_index = root / ".archive_index.json"
    original_archive_index = archive_index.read_bytes() if archive_index.is_file() else None
    digests = {row["from"]: inventory(row["from"]) for row in moves}
    moved, links, aliases = [], [], []
    try:
        for row in moves:
            old, new = Path(row["from"]), Path(row["to"])
            assert old.is_relative_to(root) and new.is_relative_to(root)
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            moved.append((old, new))
            assert inventory(new) == digests[str(old)], str(new)
        # `root` is the submissions directory; the archive helper accepts either
        # that path or the project root and rewrites both moved and stale entries.
        relocate_archive_index(root, moves)
        for key in OPERATIONAL_KEYS:
            if key in state:
                state[key] = relocate_workspace_paths(state[key], moves, project)
        state.setdefault("upload_layout", {})["version"] = 2
        atomic_json(state_path, state)
        for path in files:
            if path == state_path:
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            atomic_json(path, relocate_workspace_paths(value, moves, project))
        write_submission_index(root, state)
        # Hide only obsolete wrapper directories, never a canonical calculation.
        if os.name == "nt":
            wrappers = set()
            for row in moves:
                parts = Path(row["from"]).relative_to(root).parts
                if len(parts) >= 2:
                    wrapper = root / parts[0] / parts[1]
                    if wrapper.is_dir():
                        wrappers.add(wrapper)
            for wrapper in wrappers:
                attrs = ctypes.windll.kernel32.GetFileAttributesW(str(wrapper))
                if attrs == 0xFFFFFFFF or not ctypes.windll.kernel32.SetFileAttributesW(
                    str(wrapper), attrs | 2
                ):
                    raise ctypes.WinError()
        report = {
            "project": project.name,
            "status": "migrated",
            "date": datetime.now().isoformat(),
            "moves": moves,
            "canonical_junctions": aliases,
            "backup": str(backup),
            "immutable_files_verified": sum(map(len, digests.values())),
        }
        atomic_json(root / ".layout-migration.json", report)
        return report
    except Exception:
        if original_archive_index is not None:
            archive_index.write_bytes(original_archive_index)
        for link in reversed(links):
            if os.name == "nt":
                os.rmdir(link)  # Remove this verified junction only; never its target.
            else:
                link.unlink()
        for old, new in reversed(moved):
            if new.exists() and not old.exists():
                new.rename(old)
        for path, original in saved.items():
            shutil.copy2(original, path)
        raise

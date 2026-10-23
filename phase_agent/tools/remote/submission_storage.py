"""Complete checked local archives with lazy restoration of scientific paths."""

import ctypes
import json
import os
import re
from pathlib import Path
import zipfile

from phase_agent.tools.remote.transfer_bundle import (
    MANIFEST,
    _safe_name,
    checksum,
    create_bundle,
    import_bundle,
)


def storage_archive(root, kind):
    root = Path(root)
    for base in root.resolve().parents:
        if base.name == "submissions":
            index = base / ".archive_index.json"
            if index.is_file():
                mappings = json.loads(index.read_text(encoding="utf-8"))
                key = root.resolve().relative_to(base).as_posix()
                if key in mappings:
                    target = base / mappings[key]
                    relative = target.resolve().relative_to(base.resolve())
                    if not (
                        relative.parts[0] == "archives"
                        or (
                            len(relative.parts) >= 3
                            and re.fullmatch(r"epoch\d+", relative.parts[0])
                            and relative.parts[1] == "storage_archives"
                        )
                    ):
                        raise ValueError("Storage archive outside its registered epoch")
                    if target.is_file():
                        return target
            break
    return root / f"all.{kind}.zip"


def is_storage_archive(archive):
    try:
        with zipfile.ZipFile(archive) as zipped:
            return json.loads(zipped.read(MANIFEST)).get("kind") in {
                "storage_inputs",
                "storage_results",
            }
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def _derived(relative):
    return any(
        part in {"__pycache__", ".transfer_receipts"} for part in relative.parts
    ) or relative.name in {".transfer_archive.lock", MANIFEST}


def _files(root):
    root = Path(root).resolve()
    files = []
    for path in root.rglob("*"):
        if getattr(path, "is_junction", lambda: False)() or path.is_symlink():
            raise ValueError(f"Nested submission link cannot be archived: {path}")
        if path.is_file() and not _derived(path.relative_to(root)):
            path.resolve().relative_to(root)
            files.append(path)
    return files


def _duplicate_zip(path, root, digests):
    try:
        with zipfile.ZipFile(path) as zipped:
            manifest = json.loads(zipped.read(MANIFEST))
            rows = manifest["files"]
            if {row["path"] for row in rows} | {MANIFEST} != set(zipped.namelist()):
                return False
            import hashlib

            return all(
                row["path"] in digests
                and digests[row["path"]] == row["sha256"]
                and hashlib.sha256(zipped.read(row["path"])).hexdigest() == row["sha256"]
                for row in rows
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def pack_storage_directory(root, kind):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = storage_archive(root, kind)
    restore_storage_directory(root, kind)
    files = [path for path in _files(root) if path != archive]
    digests = {
        path.relative_to(root).as_posix(): checksum(path)
        for path in files
        if path.suffix.lower() != ".zip"
    }
    duplicates = [
        path
        for path in files
        if path.suffix.lower() == ".zip" and _duplicate_zip(path, root, digests)
    ]
    selected = [path for path in files if path not in duplicates]
    if not selected:
        return None
    create_bundle(root, selected, archive, kind=f"storage_{kind}")
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        rows = json.loads(zipped.read(MANIFEST))["files"]
        import hashlib

        for row in rows:
            assert hashlib.sha256(zipped.read(row["path"])).hexdigest() == row["sha256"]
            assert checksum(root / row["path"]) == row["sha256"]
    return {"archive": str(archive), "files": len(rows), "duplicates": [str(p) for p in duplicates]}


def prune_storage_directory(root, kind):
    root = Path(root).resolve()
    archive = storage_archive(root, kind)
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read(MANIFEST))
        if manifest["kind"] != f"storage_{kind}":
            raise ValueError("Cannot prune a partial remote return")
        rows = manifest["files"]
        digests = {row["path"]: row["sha256"] for row in rows}
        import hashlib

        for row in rows:
            if hashlib.sha256(zipped.read(row["path"])).hexdigest() != row["sha256"]:
                raise ValueError("Archive checksum mismatch; nothing deleted")
    removable = []
    for path in root.rglob("*"):
        if not path.is_file() or path == archive:
            continue
        path.resolve().relative_to(root)
        relative = path.relative_to(root)
        if _derived(relative):
            removable.append(path)
        elif relative.as_posix() in digests:
            if checksum(path) != digests[relative.as_posix()]:
                raise ValueError(f"File changed since packaging; nothing deleted: {path}")
            removable.append(path)
        elif path.suffix.lower() == ".zip" and _duplicate_zip(path, root, digests):
            removable.append(path)
        else:
            raise ValueError(f"Unarchived file; nothing deleted: {path}")
    for path in removable:
        path.unlink()
    for path in sorted(
        (p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        path.resolve().relative_to(root)
        try:
            path.rmdir()  # Empty directories only; never a recursive deletion.
        except OSError:
            _hide(path)
    return len(removable)


def _hide(path):
    if os.name == "nt":
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs != 0xFFFFFFFF:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs | 2)


def restore_storage_directory(root, kind):
    root = Path(root)
    archive = storage_archive(root, kind)
    if not archive.is_file():
        return 0
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read(MANIFEST))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid storage manifest object")
        if manifest.get("kind") != f"storage_{kind}":
            return 0  # Ordinary remote returns stay with the existing collector.
        rows = manifest.get("files")
        if manifest.get("schema") != 1 or not isinstance(rows, list) or len(rows) > 20000:
            raise ValueError("Invalid storage manifest")
        resolved = root.resolve()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("Invalid storage manifest file entry")
            relative = _safe_name(row["path"])
            (root.joinpath(*relative.parts)).resolve().relative_to(resolved)
        missing = any(not (root / row["path"]).is_file() for row in rows)
    if missing:
        import_bundle(archive, root, storage_kind=f"storage_{kind}")
    for row in rows:
        relative = Path(row["path"])
        _hide(root / relative.parts[0])
    return len(rows) if missing else 0


def calculation_directories(state):
    directories = {
        Path(batch["upload_directory"]).parent.parent
        for batch in state.get("slurm_batches", [])
        if batch.get("upload_directory")
    }
    directories.update(
        Path(job["directory"])
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("directory")
    )
    return directories


def restore_state_archives(state):
    restored = 0
    for directory in sorted(calculation_directories(state)):
        for kind in ("inputs", "results"):
            restored += restore_storage_directory(directory / kind, kind)
    return restored

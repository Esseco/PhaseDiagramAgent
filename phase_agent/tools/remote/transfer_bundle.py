"""Portable, checksummed handoff archives; scientific collectors remain authoritative."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
import time

MANIFEST = "transfer_manifest.json"
RESULT_NAMES = {
    "mc_occupations.bin",
    "result.json",
    "task.finished.json",
    "final.vasp",
    "CONTCAR",
    "checkpoint.json",
    "checkpoint.json.gz",
    "training.json",
    "mlip_result.json",
    "remote_artifacts.json",
    "initial_relaxed.vasp",
    "log_index.json",
    "status_summary.json",
    "training.finished.json",
    "models.json",
    "kfold_metrics.csv",
    "energy_comparison.csv",
    "force_comparison.csv",
}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_bundle(root, files, destination, *, kind, prefix=""):
    root = Path(root).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = []
    for source in sorted({Path(p).resolve() for p in files}):
        relative = source.relative_to(root).as_posix()
        name = f"{prefix}/{relative}" if prefix else relative
        _safe_name(name)
        if source == destination.resolve():
            raise ValueError("Archive cannot include itself")
        rows.append({"path": name, "size": source.stat().st_size, "sha256": checksum(source)})
        sources.append(source)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST, json.dumps({"schema": 1, "kind": kind, "files": rows}))
            for source, row in zip(sources, rows):
                archive.write(source, row["path"])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def input_files(directory):
    root = Path(directory)
    excluded = {"__pycache__", "results", "logs", ".git"}
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not excluded.intersection(p.relative_to(root).parts)
        and p.name not in {"UPLOAD_AND_SUBMIT.md", "SHA256SUMS"}
        and not p.name.endswith((".bak", ".part", ".log", ".pyc", ".zip"))
    ]
    return files


def input_bundle(directory):
    root = Path(directory)
    return create_bundle(
        root,
        input_files(root),
        root.parent / f"{root.name}.upload.zip",
        kind="inputs",
        prefix=root.name,
    )


def result_bundle(results_directory, task_directories, name):
    root = Path(results_directory)
    files = []
    for directory in task_directories:
        files.extend(_result_files(Path(directory)))
    runtime = root / f"{name}.runtime.json"
    if runtime.is_file():
        files.append(runtime)
    if not files:
        return None
    archive = create_bundle(root, files, root / f"{name}.results.zip", kind="results")
    consolidate_results(root)
    return archive


def _result_files(directory):
    result = directory / "result.json"
    extra_names = set()
    if result.is_file():
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("checkpoint"):
            extra_names.add(Path(payload["checkpoint"]).name)
    return [
        p
        for p in directory.iterdir()
        if p.is_file()
        and (
            p.name in RESULT_NAMES
            or p.name in extra_names
            or p.suffix.lower() in {".vasp", ".poscar"}
        )
    ]


@contextmanager
def _archive_lock(root):
    """Serialize parallel finishers without leaving locks after process death."""
    with (root / ".transfer_archive.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            deadline = time.monotonic() + 60
            while True:
                try:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Result archive consolidation is busy")
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def consolidate_results(results_directory):
    """Publish one atomically replaced round archive in a clean download folder."""
    root = Path(results_directory)
    with _archive_lock(root):
        files = _result_files(root)
        for directory in root.iterdir():
            if (
                directory.is_dir()
                and (directory / "task.finished.json").is_file()
                and (directory / "result.json").is_file()
            ):
                files.extend(_result_files(directory))
        files.extend(root.glob("*.runtime.json"))
        if not files:
            return None
        return create_bundle(root, files, root / "transfer/all.results.zip", kind="results")


def _safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or name != path.as_posix()
        or any(p in {".", "..", ""} for p in name.split("/"))
        or any(p.endswith((".", " ")) for p in path.parts)
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def import_bundle(
    archive_path, directory, *, max_bytes=8 * 1024**3, storage_kind=None, extra_result_suffixes=()
):
    """Validate the whole archive before publishing; refuse differing existing files."""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > 20000 or len({i.filename.casefold() for i in infos}) != len(infos):
            raise ValueError("Duplicate paths or too many files")
        if sum(i.file_size for i in infos) > max_bytes:
            raise ValueError("Result archive exceeds size limit")
        for info in infos:
            _safe_name(info.filename)
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive directories/symlinks are not supported")
        if archive.getinfo(MANIFEST).file_size > 4 * 1024**2:
            raise ValueError("Transfer manifest exceeds size limit")
        manifest = json.loads(archive.read(MANIFEST))
        allowed = {"results"} if storage_kind is None else {storage_kind}
        if storage_kind not in {None, "storage_inputs", "storage_results"}:
            raise ValueError("Unsupported local storage kind")
        if manifest.get("schema") != 1 or manifest.get("kind") not in allowed:
            raise ValueError("Not a supported result archive")
        rows = manifest["files"]
        names = [r["path"] for r in rows]
        if len(names) != len(set(names)) or set(names) | {MANIFEST} != {i.filename for i in infos}:
            raise ValueError("Transfer manifest file list mismatch")
        with tempfile.TemporaryDirectory(prefix=".transfer-", dir=root) as temp:
            staged = Path(temp)
            targets = []
            for row in rows:
                relative = _safe_name(row["path"])
                if (
                    storage_kind is None
                    and relative.name not in RESULT_NAMES
                    and relative.suffix.lower()
                    not in {
                        ".vasp",
                        ".poscar",
                        ".json",
                        ".gz",
                        *extra_result_suffixes,
                    }
                ):
                    raise ValueError(f"Unexpected result file: {relative}")
                target = root.joinpath(*relative.parts)
                target.resolve().relative_to(root)
                candidate = staged.joinpath(*relative.parts)
                candidate.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.open(row["path"]) as source,
                    candidate.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if candidate.stat().st_size != row["size"] or checksum(candidate) != row["sha256"]:
                    raise ValueError(f"Result checksum mismatch: {relative}")
                if target.exists() and (not target.is_file() or checksum(target) != row["sha256"]):
                    raise ValueError(f"Existing result differs: {target}")
                targets.append((candidate, target))
            for candidate, target in sorted(
                targets, key=lambda pair: pair[1].name.endswith("finished.json")
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.replace(candidate, target)
    return len(rows)


def import_registered_bundles(directories):
    errors = []
    for root in sorted({Path(p) for p in directories if p}):
        if not root.is_dir():
            continue
        for archive in _registered_archives(root):
            receipt = _archive_receipt(root, archive)
            digest = checksum(archive)
            if receipt.is_file() and receipt.read_text(encoding="utf-8") == digest:
                continue
            try:
                import_bundle(archive, root)
                receipt.parent.mkdir(exist_ok=True)
                receipt.write_text(digest, encoding="utf-8")
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                zipfile.BadZipFile,
            ) as error:
                errors.append({"archive": str(archive), "reason": str(error)})
    return errors


def registered_result_directories(state):
    directories = set()
    for row in state.get("slurm_batches", []):
        handoff = row.get("handoff") or {}
        if handoff.get("import_directory"):
            directories.add(handoff["import_directory"])
        if row.get("results_directory"):
            directories.add(str(row["results_directory"]))
        elif row.get("upload_directory"):
            upload = Path(row["upload_directory"])
            directories.add(
                str(
                    (upload.parent.parent if upload.parent.name == "inputs" else upload.parent)
                    / "results"
                )
            )
    return directories


def _registered_archives(root):
    root = Path(root)
    archives = sorted(
        [
            *root.glob("*.results.zip"),
            *(root / "transfer").glob("*.results.zip"),
            *(root / "return-results").glob("*.results.zip"),
        ]
    )
    from phase_agent.tools.remote.submission_storage import is_storage_archive

    return [archive for archive in archives if not is_storage_archive(archive)]


def _archive_receipt(root, archive):
    root = Path(root)
    name = archive.relative_to(root).as_posix().replace("/", "__")
    return root / ".transfer_receipts" / (name + ".sha256")


def has_registered_return(state):
    for batch in (state.get("relax_mc_batches") or {}).values():
        archive = Path(batch.get("expected_results_archive") or "")
        if archive.is_file() and batch.get("import_checksum") != checksum(archive):
            return True
    directories = registered_result_directories(state)
    directories.update(
        str(Path(job["directory"]))
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("directory")
        and not job.get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
    )
    directories.update(
        str(Path(job["directory"]) / "results")
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("directory")
        and not job.get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
    )
    return any(
        not _archive_receipt(root, archive).is_file()
        or _archive_receipt(root, archive).read_text(encoding="utf-8") != checksum(archive)
        for root in directories
        for archive in _registered_archives(root)
    )


def publish_return(handoff_directory, results_relative):
    """Called by each job's exit hook; publish one ZIP per result operation."""
    handoff = Path(handoff_directory).resolve()
    relative = _safe_name(results_relative)
    results = handoff.joinpath(*relative.parts)
    results.resolve().relative_to(handoff)
    if not results.is_dir():
        return None
    # The operation lock covers collection and publishing, including concurrent jobs.
    with _archive_lock(results):
        files = _result_files(results)
        for directory in results.iterdir():
            if (
                directory.is_dir()
                and (directory / "task.finished.json").is_file()
                and (directory / "result.json").is_file()
            ):
                files.extend(_result_files(directory))
        files.extend(results.glob("*.runtime.json"))
        if not files:
            return None
        name = "__".join(relative.parts) + ".results.zip"
        return create_bundle(handoff, files, handoff / "return-results" / name, kind="results")


if __name__ == "__main__":
    import sys

    publish_return(sys.argv[1], sys.argv[2])

"""Compute-node dispatcher. Requires Python 3.10 or newer; standard library only."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

if "preflight_environment" not in globals():
    from phase_agent.tools.remote.environment_preflight import preflight_environment


def safe_path(root, name):
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(p in {".", ".."} for p in name.split("/"))
    ):
        raise ValueError("Unsafe archive path: " + name)
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes upload directory")
    return target


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def submission_order(scripts, afterany):
    """Validate local script dependencies before any scheduler side effect."""
    from graphlib import CycleError, TopologicalSorter

    if not isinstance(afterany, dict) or set(afterany) - set(scripts):
        raise ValueError("Invalid afterany dependency targets")
    for parents in afterany.values():
        if (
            not isinstance(parents, list)
            or not parents
            or any(not isinstance(parent, str) or parent not in scripts for parent in parents)
            or len(set(parents)) != len(parents)
        ):
            raise ValueError("Invalid afterany dependency scripts")
    try:
        return list(
            TopologicalSorter({job: afterany.get(job, []) for job in scripts}).static_order()
        )
    except CycleError as error:
        raise ValueError("Cyclic submission dependencies") from error


def _task_files_for_job(root, job):
    """Include validated sibling task references in regrouped handoffs."""
    root = Path(root).resolve()
    directory = safe_path(root, job).parent
    manifest = directory / "manifest.json"
    if manifest.is_file():
        rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if isinstance(rows, list) and any(
            isinstance(row, dict)
            and ".." in str(row.get("input_path") or "").replace("\\", "/").split("/")
            for row in rows
        ):
            sources = []
            for row in rows:
                name = row.get("input_path") if isinstance(row, dict) else None
                if not isinstance(name, str) or not name or ":" in name:
                    raise ValueError("invalid regrouped task path")
                relative = PurePosixPath(name.replace("\\", "/"))
                source = (directory / str(relative)).resolve()
                if relative.is_absolute() or not source.is_relative_to(root):
                    raise ValueError("regrouped task path outside handoff")
                if source.name != "task.json" or not source.is_file():
                    raise ValueError("missing regrouped task input")
                task = json.loads(source.read_text(encoding="utf-8-sig"))
                if not row.get("task_id") or task.get("task_id") != row["task_id"]:
                    raise ValueError("regrouped task identity mismatch")
                sources.append(source)
            if len(set(sources)) != len(sources) or len({row["task_id"] for row in rows}) != len(
                rows
            ):
                raise ValueError("duplicate regrouped task input or identity")
            return sources
    return directory.rglob("task.json")


def preflight_models(root, jobs):
    """Check all selected MLIP dependencies before submitting any child job."""
    required = {}
    for job in jobs:
        safe_path(root, job)
        for source in _task_files_for_job(root, job):
            task = json.loads(source.read_text(encoding="utf-8"))
            worker = task.get("worker_job") or {}
            evidence = {r.get("model_path"): r for r in worker.get("model_file_evidence") or []}
            paths = worker.get("model_paths") or (
                [worker["model_path"]] if worker.get("model_path") else []
            )
            if (
                worker
                and task.get("stage")
                in {"relax_and_feature", "deep_search", "parent_first_sampling"}
                and not paths
            ):
                raise ValueError("MLIP model path missing; no child jobs submitted: " + str(source))
            for name in paths:
                path = Path(name)
                if not path.is_absolute():
                    path = source.parent / path
                row = evidence.get(name) or {}
                key = str(path.resolve())
                old = required.get(key)
                if (
                    old
                    and old.get("sha256")
                    and row.get("sha256")
                    and old["sha256"] != row["sha256"]
                ):
                    raise ValueError("Conflicting model identity: " + key)
                required[key] = row if row.get("sha256") or not old else old
    missing = [name for name in required if not Path(name).is_file()]
    if missing:
        raise FileNotFoundError(
            "remote_model_not_found: HPC model files missing; no child jobs submitted. Correct the remote path or restore weights on HPC, then retry. Local weights are not required: "
            + "; ".join(missing)
        )
    for name, row in required.items():
        path = Path(name)
        if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
            raise ValueError("Model size mismatch; no child jobs submitted: " + name)
        if row.get("sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("Model checksum mismatch; no child jobs submitted: " + name)
    return len(required)


def _input_matches(source, row, output=None):
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        size += len(chunk)
        if size > row["size"]:
            return False
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    return size == row["size"] and digest.hexdigest() == row["sha256"]


def _extract_input(archive, row, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".dispatch-", suffix=".part", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            with archive.open(row["path"]) as source:
                if not _input_matches(source, row, output):
                    raise ValueError("Input checksum mismatch: " + row["path"])
        if target.exists():
            raise ValueError("Input appeared during extraction: " + row["path"])
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _inputs_verified(path, archive_hash):
    try:
        return json.loads(path.read_text(encoding="utf-8")) == {"archive_sha256": archive_hash}
    except (OSError, ValueError):
        return False


def _run(root):
    import sys

    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required on the compute node")
    root = Path(root).resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Submit with sbatch submit_all.sh; do not run this Python program on the login node"
        )
    # Linux advisory lock prevents two simultaneous dispatchers.
    import fcntl

    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "inputs.zip").open("rb") as stream:
            _dispatch_package(root, stream)


_QOS_LIMIT = "slurm_qos_submit_limit"


class QosSubmissionRejected(RuntimeError):
    """A synchronous refusal with no accepted job ID."""


def _qos_limit_rejection(result):
    lines = result.stderr.splitlines()
    return (
        result.returncode != 0
        and not result.stdout.strip()
        and "sbatch: error: QOSMaxSubmitJobPerUserLimit" in lines
        and any(
            line.startswith("sbatch: error: Batch job submission failed: Job violates accounting/QOS policy")
            for line in lines
        )
    )


def _dispatch_package(root, stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    archive_hash = digest.hexdigest()
    receipt_path = root / ".submitted_jobs.json"
    receipt = (
        json.loads(receipt_path.read_text())
        if receipt_path.exists()
        else {"archive_sha256": archive_hash, "jobs": {}}
    )
    if receipt["archive_sha256"] != archive_hash:
        raise RuntimeError(
            "Different package in an already submitted directory; use a fresh directory"
        )
    validation_path = root / ".dispatch-validated-inputs.json"
    with zipfile.ZipFile(stream) as archive:
        manifest = json.loads(archive.read("transfer_manifest.json"))
        rows = manifest["files"]
        if (
            manifest.get("kind") != "inputs"
            or len(rows) > 20000
            or sum(r["size"] for r in rows) > 8 * 1024**3
        ):
            raise ValueError("Invalid or oversized input package")
        names = [r["path"] for r in rows]
        if len(names) != len(set(names)) or sorted(archive.namelist()) != sorted(
            names + ["transfer_manifest.json"]
        ):
            raise ValueError("Archive file list mismatch")
        plan = json.loads(archive.read("submission_plan.json"))
        jobs = plan["scripts"]
        if (
            not jobs
            or len(jobs) != len(set(jobs))
            or any(j not in names or Path(j).suffix not in {".sh", ".sbatch"} for j in jobs)
        ):
            raise ValueError("Invalid job script list")
        dependencies = plan.get("afterany", {})
        jobs = submission_order(jobs, dependencies)
        remaining = []
        for job in jobs:
            saved = receipt["jobs"].get(job)
            if saved is None or (
                saved.get("status") == "rejected" and saved.get("code") == _QOS_LIMIT
            ):
                remaining.append(job)
            elif saved.get("status") == "submitting":
                raise RuntimeError(
                    "Previous submission interrupted; check squeue/sacct before resolving receipt: "
                    + job
                )
            elif (
                saved.get("status") != "submitted"
                or not isinstance(saved.get("job_id"), str)
                or not saved["job_id"].isdigit()
            ):
                raise ValueError("Invalid submitted job receipt: " + job)
        if not remaining:
            if root.name.endswith("_input"):
                (root.with_name(root.name[:-6] + "_output") / "diagnostics.zip").unlink(
                    missing_ok=True
                )
            print(f"Already submitted: {len(jobs)} jobs; see {receipt_path.name}")
            print("New jobs: 0")
            return
        if not shutil.which("sbatch"):
            raise RuntimeError("sbatch unavailable on this compute node; no child jobs were submitted")
        reserved = {validation_path.name, validation_path.with_suffix(".tmp").name}
        if reserved.intersection(names):
            raise ValueError("Input package contains a reserved dispatcher path")
        verified = _inputs_verified(validation_path, archive_hash)
        missing = []
        # A cold package is fully verified before publishing any member.
        for row in rows:
            target = safe_path(root, row["path"])
            entry = archive.getinfo(row["path"])
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink in archive")
            if not verified:
                with archive.open(entry) as source:
                    if not _input_matches(source, row):
                        raise ValueError("Input checksum mismatch: " + row["path"])
            if target.exists():
                with target.open("rb") as existing:
                    if not _input_matches(existing, row):
                        raise ValueError("Existing input changed: " + row["path"])
            else:
                missing.append((row, target))
        if not verified:
            save(validation_path, {"archive_sha256": archive_hash})
        for row, target in missing:
            _extract_input(archive, row, target)
    try:
        modules = set()
        for job in remaining:
            for source in _task_files_for_job(root, job):
                task = json.loads(source.read_text(encoding="utf-8-sig"))
                if task.get("worker_job") and task.get("stage") in {
                    "relax_and_feature",
                    "deep_search",
                    "parent_first_sampling",
                }:
                    modules.update(("numpy", "scipy", "pymatgen", "ase", "torch", "mace"))
                    if task.get("stage") == "parent_first_sampling":
                        modules.update(
                            (
                                "maml",
                                "tensorflow",
                                "h5py",
                                "Process_MLIP",
                                "Process_AL_MC",
                                "phase_agent",
                            )
                        )
        preflight_models(root, remaining)
        preflight_environment(root, remaining, required_modules=modules)
    except (FileNotFoundError, ValueError, PermissionError, RuntimeError) as error:
        receipt["model_preflight"] = getattr(error, "report", None) or {
            "status": "failed",
            "error": str(error),
            "location": "remote",
            "remedy": "Correct the HPC model reference or restore the recorded weights on HPC; local model download is not required. Already submitted jobs must not be submitted again.",
        }
        save(receipt_path, receipt)
        if root.name.endswith("_input"):
            output = root.with_name(root.name[:-6] + "_output")
            output.mkdir(exist_ok=True)
            temporary = output / "diagnostics.zip.part"
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("dispatch-error.json", json.dumps(receipt["model_preflight"]))
            temporary.replace(output / "diagnostics.zip")
        else:
            save(root / "return-results/dispatch-error.json", receipt["model_preflight"])
        raise
    if root.name.endswith("_input"):
        (root.with_name(root.name[:-6] + "_output") / "diagnostics.zip").unlink(missing_ok=True)
    receipt.pop("model_preflight", None)
    submitted = 0
    for job in remaining:
        target = safe_path(root, job)
        parent_ids = [receipt["jobs"][parent]["job_id"] for parent in dependencies.get(job, [])]
        if any(not isinstance(job_id, str) or not job_id.isdigit() for job_id in parent_ids):
            raise ValueError("Invalid submitted dependency job ID")
        command = ["sbatch", "--parsable"]
        if parent_ids:
            command.append("--dependency=afterany:" + ":".join(parent_ids))
        command.append(target.name)
        # Persist intent immediately before the side effect, after validation.
        receipt["jobs"][job] = {"status": "submitting"}
        save(receipt_path, receipt)
        result = subprocess.run(
            command,
            cwd=target.parent,
            capture_output=True,
            text=True,
        )
        if _qos_limit_rejection(result):
            receipt["jobs"][job] = {"status": "rejected", "code": _QOS_LIMIT}
            save(receipt_path, receipt)
            raise QosSubmissionRejected(
                f"{job}: QOSMaxSubmitJobPerUserLimit; retry this dispatcher when quota allows"
            )
        if result.returncode:
            raise RuntimeError(
                f"sbatch failed for {job}; check Slurm and receipt before retry: "
                + result.stderr.strip()[:1024]
            )
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(
                f"Unexpected sbatch reply for {job}; check Slurm before retry: "
                + result.stdout.strip()[:1024]
            )
        receipt["jobs"][job] = {"status": "submitted", "job_id": job_id}
        if parent_ids:
            receipt["jobs"][job]["afterany_job_ids"] = parent_ids
        save(receipt_path, receipt)
        submitted += 1
    print(f"New jobs: {submitted}; see {receipt_path.name} for job IDs")
    print(
        "Download the sibling <batch>_output folder after all jobs finish; keep its ZIP files compressed (legacy packages retain return-results)."
    )


def run(root):
    try:
        return _run(root)
    except Exception as error:
        root = Path(root).resolve()
        if os.environ.get("SLURM_JOB_ID") and root.name.endswith("_input"):
            output = root.with_name(root.name[:-6] + "_output")
            output.mkdir(exist_ok=True)
            destination = output / "diagnostics.zip"
            # Preserve the model preflight's more specific remedy when present.
            if not destination.exists():
                temporary = output / "diagnostics.zip.part"
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(
                        "dispatch-error.json",
                        json.dumps(
                            {
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}"[:2048],
                                "location": "remote",
                                **(
                                    {
                                        "code": _QOS_LIMIT,
                                        "remedy": "Retry this dispatcher when quota allows. Existing job receipts remain authoritative.",
                                    }
                                    if isinstance(error, QosSubmissionRejected)
                                    else {
                                        "remedy": "Inspect dispatch receipt and Slurm before retry; do not duplicate jobs."
                                    }
                                ),
                            }
                        ),
                    )
                temporary.replace(destination)
        raise


if __name__ == "__main__":
    import sys

    run(sys.argv[1])

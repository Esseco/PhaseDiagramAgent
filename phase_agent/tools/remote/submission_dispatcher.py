"""Compute-node dispatcher. Requires Python 3.10 or newer; standard library only."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
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


def preflight_models(root, jobs):
    """Check all selected MLIP dependencies before submitting any child job."""
    required = {}
    for job in jobs:
        directory = safe_path(root, job).parent
        for source in directory.rglob("task.json"):
            task = json.loads(source.read_text(encoding="utf-8"))
            worker = task.get("worker_job") or {}
            evidence = {r.get("model_path"): r for r in worker.get("model_file_evidence") or []}
            paths = worker.get("model_paths") or (
                [worker["model_path"]] if worker.get("model_path") else []
            )
            if worker and task.get("stage") in {"relax_and_feature", "deep_search"} and not paths:
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


def _run(root):
    import sys

    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required on the compute node")
    root = Path(root).resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Submit with sbatch submit_all.sh; do not run this Python program on the login node"
        )
    if not shutil.which("sbatch"):
        raise RuntimeError("sbatch unavailable on this compute node; no child jobs were submitted")
    # Linux advisory lock prevents two simultaneous dispatchers.
    import fcntl

    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = hashlib.sha256()
        with (root / "inputs.zip").open("rb") as stream:
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
        with zipfile.ZipFile(root / "inputs.zip") as archive:
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
            # Validate every file and every destination before extracting/submitting anything.
            for row in rows:
                target = safe_path(root, row["path"])
                entry = archive.getinfo(row["path"])
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Symlink in archive")
                digest, size = hashlib.sha256(), 0
                with archive.open(entry) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if size != row["size"] or digest.hexdigest() != row["sha256"]:
                    raise ValueError("Input checksum mismatch: " + row["path"])
                if (
                    target.exists()
                    and hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("Existing input changed: " + row["path"])
            plan = json.loads(archive.read("submission_plan.json"))
            jobs = plan["scripts"]
            if (
                not jobs
                or len(jobs) != len(set(jobs))
                or any(j not in names or Path(j).suffix not in {".sh", ".sbatch"} for j in jobs)
            ):
                raise ValueError("Invalid job script list")
            for job in jobs:
                if receipt["jobs"].get(job, {}).get("status") == "submitting":
                    raise RuntimeError(
                        "Previous submission interrupted; check squeue/sacct before resolving receipt: "
                        + job
                    )
            for row in rows:
                target = safe_path(root, row["path"])
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(row["path"]) as stream, target.open("wb") as output:
                        shutil.copyfileobj(stream, output, 1024 * 1024)
        try:
            remaining = [
                job for job in jobs if receipt["jobs"].get(job, {}).get("status") != "submitted"
            ]
            modules = set()
            for job in remaining:
                for source in safe_path(root, job).parent.rglob("task.json"):
                    task = json.loads(source.read_text(encoding="utf-8-sig"))
                    if task.get("worker_job") and task.get("stage") in {
                        "relax_and_feature",
                        "deep_search",
                    }:
                        modules.update(("numpy", "scipy", "pymatgen", "ase", "torch", "mace"))
            preflight_models(
                root,
                [job for job in jobs if receipt["jobs"].get(job, {}).get("status") != "submitted"],
            )
            if remaining:
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
        for job in jobs:
            if receipt["jobs"].get(job, {}).get("status") == "submitted":
                print("Already submitted:", job, receipt["jobs"][job]["job_id"])
                continue
            # Persist intent first: interruption cannot cause a blind duplicate submission.
            receipt["jobs"][job] = {"status": "submitting"}
            save(receipt_path, receipt)
            target = safe_path(root, job)
            result = subprocess.run(
                ["sbatch", "--parsable", target.name],
                cwd=target.parent,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(
                    "sbatch failed; check Slurm and receipt before retry: " + result.stderr.strip()
                )
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(
                    "Unexpected sbatch reply; check Slurm before retry: " + result.stdout
                )
            receipt["jobs"][job] = {"status": "submitted", "job_id": job_id}
            save(receipt_path, receipt)
            submitted += 1
            print("Submitted:", job, job_id)
        print("New jobs:", submitted)
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
                                "error": f"{type(error).__name__}: {error}",
                                "location": "remote",
                                "remedy": "Inspect dispatch receipt and Slurm before retry; do not duplicate jobs.",
                            }
                        ),
                    )
                temporary.replace(destination)
        raise


if __name__ == "__main__":
    import sys

    run(sys.argv[1])

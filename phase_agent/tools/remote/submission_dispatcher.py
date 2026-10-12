"""Compute-node dispatcher. Requires Python 3.10 or newer; standard library only."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile


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
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(root):
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
        archive_hash = hashlib.sha256((root / "inputs.zip").read_bytes()).hexdigest()
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
                content = archive.read(entry)
                if (
                    len(content) != row["size"]
                    or hashlib.sha256(content).hexdigest() != row["sha256"]
                ):
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
                    target.write_bytes(archive.read(row["path"]))
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
            "Download the return-results folder after all jobs finish; keep its ZIP files compressed."
        )


if __name__ == "__main__":
    import sys

    run(sys.argv[1])

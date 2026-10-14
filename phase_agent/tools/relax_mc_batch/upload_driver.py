"""Standalone compute-node entry: verify inputs, run coordinator, publish output."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import traceback
import zipfile


def prepare_work(root):
    archive = root / "inputs.zip"
    fingerprint = hashlib.sha256(archive.read_bytes()).hexdigest()
    work = root / "work"
    receipt = work / ".input_fingerprint"
    with zipfile.ZipFile(archive) as zipped:
        manifest_bytes = zipped.read("transfer_manifest.json")
        manifest = json.loads(manifest_bytes)
        rows = manifest["files"]
        names = [row["path"] for row in rows]
        if manifest.get("schema") != 1 or manifest.get("kind") != "inputs" or len(rows) > 20000:
            raise ValueError("Invalid input manifest")
        if (
            len(set(names)) != len(names)
            or set(names) | {"transfer_manifest.json"} != set(zipped.namelist())
            or len(zipped.namelist()) != len(names) + 1
        ):
            raise ValueError("Input inventory mismatch")
        if sum(zipped.getinfo(name).file_size for name in names) > 8 * 1024**3:
            raise ValueError("Input package too large")
        for row in rows:
            name = row["path"]
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or "\\" in name
                or ":" in name
                or any(p in {"", ".", ".."} for p in name.split("/"))
            ):
                raise ValueError("Unsafe input path")
            payload = zipped.read(name)
            if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("Input checksum mismatch")
        if receipt.exists():
            if receipt.read_text() != fingerprint:
                raise RuntimeError("Input package changed; use a fresh input directory")
        elif work.exists():
            raise RuntimeError("Incomplete extraction; inspect work before retry")
        else:
            work.mkdir()
            for name in names:
                target = work.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read(name))
            # The coordinator needs the archive's manifest as well as its payload.
            (work / "transfer_manifest.json").write_bytes(manifest_bytes)
            receipt.write_text(fingerprint)
        target = work / "transfer_manifest.json"
        if target.exists() and target.read_bytes() != manifest_bytes:
            raise ValueError("Extracted input manifest changed")
        if not target.exists():
            # Repair packages unpacked by the old driver without touching progress.
            target.write_bytes(manifest_bytes)
    return work


def main(root=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use sbatch submit_all.sh; Python must run on a compute node")
    root = Path(root or Path.cwd()).resolve()
    label = root.name.removesuffix("_input")
    output = root.with_name(label + "_output")
    output.mkdir(exist_ok=True)
    code, error = 1, None
    with (output / "run.log").open("a", encoding="utf-8") as log:
        try:
            work = prepare_work(root)
            env = dict(os.environ, SLURM_SUBMIT_DIR=str(work))
            code = subprocess.call(
                ["bash", "submit_all.sh"],
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if (work / "results.zip").is_file():
                temporary = output / "results.zip.part"
                shutil.copy2(work / "results.zip", temporary)
                temporary.replace(output / "results.zip")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=log)
    (output / "run_status.json").write_text(
        json.dumps(
            {
                "batch": label,
                "returncode": code,
                "error": error,
                "results_available": (output / "results.zip").is_file(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Return this complete output folder: {output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

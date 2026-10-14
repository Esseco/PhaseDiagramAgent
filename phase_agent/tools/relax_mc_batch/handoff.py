"""Submit script, compute-node Python driver and one approved input archive."""

from pathlib import Path
import shutil


BOOTSTRAP = """import hashlib, json, os, subprocess, zipfile
from pathlib import Path, PurePosixPath
if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Use sbatch submit_all.sh; Python must run on a compute node")
root = Path.cwd().resolve()
archive = root / "inputs.zip"
fingerprint = hashlib.sha256(archive.read_bytes()).hexdigest()
work = root / "work"
receipt = work / ".input_fingerprint"
if receipt.exists():
    if receipt.read_text() != fingerprint:
        raise RuntimeError("Input package changed; use a fresh upload directory")
elif work.exists():
    raise RuntimeError("Incomplete extraction; inspect work before retry")
else:
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("transfer_manifest.json"))
        rows = manifest["files"]
        names = [row["path"] for row in rows]
        if manifest.get("schema") != 1 or manifest.get("kind") != "inputs" or len(rows) > 20000:
            raise ValueError("Invalid input manifest")
        if len(set(names)) != len(names) or set(names) | {"transfer_manifest.json"} != set(zipped.namelist()):
            raise ValueError("Input inventory mismatch")
        if sum(zipped.getinfo(name).file_size for name in names) > 8 * 1024**3:
            raise ValueError("Input package too large")
        for row in rows:
            name = row["path"]
            path = PurePosixPath(name)
            if not name or path.is_absolute() or "\\\\" in name or ":" in name or any(p in {"", ".", ".."} for p in name.split("/")):
                raise ValueError("Unsafe input path")
            payload = zipped.read(name)
            if len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("Input checksum mismatch")
        work.mkdir()
        for name in names:
            target = work.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(name))
        receipt.write_text(fingerprint)
env = dict(os.environ, SLURM_SUBMIT_DIR=str(work))
raise SystemExit(subprocess.call(["bash", "submit_all.sh"], cwd=work, env=env))
"""


def compact_handoff(directory):
    """Publish a compact view without altering archived inputs or task records."""
    root = Path(directory).resolve()
    archive = root / "inputs.zip"
    original = (root / "submit_all.sh").read_text(encoding="utf-8")
    folder = root / "upload"
    folder.mkdir(exist_ok=True)
    target = folder / "inputs.zip"
    if target.exists():
        from phase_agent.tools.remote.integrity import file_checksum

        if file_checksum(target) != file_checksum(archive):
            raise ValueError("Existing upload package differs; inspect before retry")
    else:
        shutil.copy2(archive, target)
    # Keep the coordinator's allocation and activation; replace only its command.
    prefix = original.split('cd -- "${SLURM_SUBMIT_DIR', 1)
    if len(prefix) != 2:
        raise ValueError("Unsupported coordinator script")
    guard = ': "${SLURM_JOB_ID:?Use sbatch submit_all.sh; Python runs on the compute node}"\n'
    header = prefix[0].replace("set -euo pipefail\n", "set -euo pipefail\n" + guard, 1)
    if guard not in header:
        raise ValueError("Coordinator script lacks strict shell setup")
    command = (
        'cd -- "${SLURM_SUBMIT_DIR:?Submit from the upload directory}"\n'
        "python3 run_upload.py\n"
    )
    (folder / "run_upload.py").write_text(BOOTSTRAP, encoding="utf-8", newline="\n")
    (folder / "submit_all.sh").write_text(
        header + command, encoding="utf-8", newline="\n"
    )
    return folder

"""One portable ZIP plus a login-node dispatcher per calculation handoff."""

import hashlib
import json
from pathlib import Path
from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files
from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script

DRIVER = "\"\"\"Embedded in submit_all.sh. Only Python standard library is needed remotely.\"\"\"\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path, PurePosixPath\nimport shutil\nimport subprocess\nimport zipfile\n\n\ndef safe_path(root, name):\n    path = PurePosixPath(name)\n    if not name or '\\\\' in name or ':' in name or path.is_absolute() or any(p in {'.', '..'} for p in name.split('/')):\n        raise ValueError('Unsafe archive path: ' + name)\n    target = root.joinpath(*path.parts)\n    if not target.resolve().is_relative_to(root.resolve()):\n        raise ValueError('Path escapes upload directory')\n    return target\n\n\ndef save(path, payload):\n    temporary = path.with_suffix('.tmp')\n    with temporary.open('w', encoding='utf-8') as stream:\n        json.dump(payload, stream, indent=2)\n        stream.flush()\n        os.fsync(stream.fileno())\n    os.replace(temporary, path)\n\n\ndef run(root):\n    root = Path(root).resolve()\n    if os.environ.get('SLURM_JOB_ID'):\n        raise RuntimeError('Run bash submit_all.sh on the login node, not sbatch submit_all.sh')\n    if not shutil.which('sbatch'):\n        raise RuntimeError('sbatch not found; run this on the supercomputer login node')\n    # Linux advisory lock prevents two simultaneous dispatchers.\n    import fcntl\n    with (root / '.submit.lock').open('a') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        archive_hash = hashlib.sha256((root / 'inputs.zip').read_bytes()).hexdigest()\n        receipt_path = root / '.submitted_jobs.json'\n        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'archive_sha256': archive_hash, 'jobs': {}}\n        if receipt['archive_sha256'] != archive_hash:\n            raise RuntimeError('Different package in an already submitted directory; use a fresh directory')\n        with zipfile.ZipFile(root / 'inputs.zip') as archive:\n            manifest = json.loads(archive.read('transfer_manifest.json'))\n            rows = manifest['files']\n            if manifest.get('kind') != 'inputs' or len(rows) > 20000 or sum(r['size'] for r in rows) > 8 * 1024**3:\n                raise ValueError('Invalid or oversized input package')\n            names = [r['path'] for r in rows]\n            if len(names) != len(set(names)) or sorted(archive.namelist()) != sorted(names + ['transfer_manifest.json']):\n                raise ValueError('Archive file list mismatch')\n            # Validate every file and every destination before extracting/submitting anything.\n            for row in rows:\n                target = safe_path(root, row['path'])\n                entry = archive.getinfo(row['path'])\n                if (entry.external_attr >> 16) & 0o170000 == 0o120000:\n                    raise ValueError('Symlink in archive')\n                content = archive.read(entry)\n                if len(content) != row['size'] or hashlib.sha256(content).hexdigest() != row['sha256']:\n                    raise ValueError('Input checksum mismatch: ' + row['path'])\n                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:\n                    raise ValueError('Existing input changed: ' + row['path'])\n            plan = json.loads(archive.read('submission_plan.json'))\n            jobs = plan['scripts']\n            if not jobs or len(jobs) != len(set(jobs)) or any(j not in names or Path(j).suffix not in {'.sh', '.sbatch'} for j in jobs):\n                raise ValueError('Invalid job script list')\n            for job in jobs:\n                if receipt['jobs'].get(job, {}).get('status') == 'submitting':\n                    raise RuntimeError('Previous submission interrupted; check squeue/sacct before resolving receipt: ' + job)\n            for row in rows:\n                target = safe_path(root, row['path'])\n                if not target.exists():\n                    target.parent.mkdir(parents=True, exist_ok=True)\n                    target.write_bytes(archive.read(row['path']))\n        submitted = 0\n        for job in jobs:\n            if receipt['jobs'].get(job, {}).get('status') == 'submitted':\n                print('Already submitted:', job, receipt['jobs'][job]['job_id'])\n                continue\n            # Persist intent first: interruption cannot cause a blind duplicate submission.\n            receipt['jobs'][job] = {'status': 'submitting'}\n            save(receipt_path, receipt)\n            target = safe_path(root, job)\n            result = subprocess.run(['sbatch', '--parsable', target.name], cwd=target.parent, capture_output=True, text=True)\n            if result.returncode:\n                raise RuntimeError('sbatch failed; check Slurm and receipt before retry: ' + result.stderr.strip())\n            job_id = result.stdout.strip().split(';')[0]\n            if not job_id.isdigit():\n                raise RuntimeError('Unexpected sbatch reply; check Slurm before retry: ' + result.stdout)\n            receipt['jobs'][job] = {'status': 'submitted', 'job_id': job_id}\n            save(receipt_path, receipt)\n            submitted += 1\n            print('Submitted:', job, job_id)\n        print('New jobs:', submitted)\n        print('Download the return-results folder after all jobs finish; keep its ZIP files compressed.')\n\n\nif __name__ == '__main__':\n    import sys\n    run(sys.argv[1])\n"


def handoff_root(directory):
    directory = Path(directory)
    return next(
        (
            p
            for p in [directory, *directory.parents]
            if p.name.startswith("epoch") and "_" in p.name
        ),
        directory,
    )


def prepare_handoff(directory, scripts):
    """Build a portable dispatch package without changing scientific inputs."""
    import tempfile
    import shutil

    directory = Path(directory).resolve()
    scripts = sorted({Path(p).resolve().relative_to(directory).as_posix() for p in scripts})
    if not scripts:
        return None
    files = set()
    result_roots = {}
    for script in scripts:
        target = directory / script
        if not target.is_file():
            raise ValueError("提交脚本不存在：" + script)
        inputs = next((p for p in target.parents if p.name == "inputs"), None)
        operation = inputs.parent if inputs else target.parent.parent
        payload = target.parent
        if inputs:
            parts = target.relative_to(inputs).parts
            payload = inputs / parts[0] if len(parts) > 1 else inputs
        files.update(input_files(payload))
        result_roots[script] = (operation / "results").relative_to(directory).as_posix()
    with tempfile.TemporaryDirectory(prefix="phase-handoff-") as temp:
        portable = Path(temp)
        for source in files:
            destination = portable / source.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        shutil.copy2(
            Path(__file__).with_name("transfer_bundle.py"), portable / "_round_transfer.py"
        )
        (portable / "submission_plan.json").write_text(
            json.dumps({"schema": 1, "scripts": scripts}), encoding="utf-8"
        )
        for script in scripts:
            target = portable / script
            content = target.read_text(encoding="utf-8")
            lines = content.splitlines(keepends=True)
            index = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if line.strip() and not line.lstrip().startswith("#")
                ),
                len(lines),
            )
            upward = "/".join([".."] * len(Path(script).parent.parts)) or "."
            hook = (
                'PHASE_HANDOFF_ROOT="$(cd -- "' + upward + '" && pwd)"\n'
                'trap \'phase_status=$?; python3 "$PHASE_HANDOFF_ROOT/_round_transfer.py" "$PHASE_HANDOFF_ROOT" '
                + '"'
                + result_roots[script]
                + '"'
                + ' || phase_status=1; exit "$phase_status"\' EXIT\n'
            )
            lines.insert(index, hook)
            target.write_text("".join(lines), encoding="utf-8", newline="\n")
        portable_files = input_files(portable)
        identity = hashlib.sha256()
        for source in sorted(portable_files):
            identity.update(source.relative_to(portable).as_posix().encode())
            identity.update(hashlib.sha256(source.read_bytes()).digest())
        key = identity.hexdigest()[:16]
        base = next((p for p in directory.parents if p.name == "submissions"), directory.parent)
        label = (
            directory.name.split("_", 1)[0]
            if directory.name.startswith("epoch")
            else directory.name
        )
        folder = base / "archives" / "01-upload" / (label + "__" + key)
        archive = folder / "inputs.zip"
        if not archive.exists():
            create_bundle(portable, portable_files, archive, kind="inputs")
    script = folder / "submit_all.sh"
    write_unix_shell_script(
        script,
        '#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\npython3 - "$PWD" <<"PHASE_SUBMIT_PY"\n'
        + DRIVER
        + "\nPHASE_SUBMIT_PY\n",
    )
    return {
        "folder": str(folder),
        "upload_archive": str(archive),
        "submit_script": str(script),
        "command": "bash submit_all.sh",
        "results_directory": str(directory / "results"),
        "return_directory": str(directory / "return-results"),
        "import_directory": str(directory),
        "job_count": len(scripts),
        "layout_version": 2,
    }


def prepare_batch_handoff(state, batch):
    directory = handoff_root(Path(batch["upload_directory"]).parent.parent)
    tasks = {r["task_id"]: r for r in state.get("tasks", [])}
    selected = []
    for row in state.get("slurm_batches", []):
        if not row.get("script_path") or row.get("job_id") or row.get("status") != "prepared":
            continue
        if handoff_root(Path(row["upload_directory"]).parent.parent) != directory:
            continue
        members = [tasks.get(t, {}) for t in row.get("task_ids", [])]
        if not members or any(t.get("status") != "pending" for t in members):
            continue
        selected.append(row)
    if not selected:
        return None
    unpublished = [
        r
        for r in selected
        if not r.get("handoff") or not Path(r["handoff"]["upload_archive"]).is_file()
    ]
    if not unpublished:
        return selected[0]["handoff"]
    # Later incremental preparation must not republish already handed-off jobs
    # in a new directory: remote receipts belong to their original package.
    selected = unpublished
    ids = sorted(r["batch_id"] for r in selected)
    from phase_agent.tools.remote.submission_storage import restore_storage_directory

    for row in selected:
        restore_storage_directory(Path(row["upload_directory"]).parent, "inputs")
    handoff = prepare_handoff(directory, [r["script_path"] for r in selected])
    if handoff:
        handoff["batch_ids"] = ids
        for row in selected:
            row["handoff"] = handoff
    return handoff


def ensure_waiting_handoffs(state):
    # Called once at the workflow handoff, after all batches have been prepared.
    seen = set()
    for row in state.get("slurm_batches", []):
        if not row.get("upload_directory") or not row.get("script_path"):
            continue
        directory = handoff_root(Path(row["upload_directory"]).parent.parent)
        if directory in seen:
            continue
        seen.add(directory)
        prior = row.get("handoff")
        current = prepare_batch_handoff(state, row)
        if current and current != prior:
            base = next((p for p in directory.parents if p.name == "submissions"), None)
            if base:
                from phase_agent.tools.remote.submission_layout import write_submission_index

                write_submission_index(base, state)

"""Compressed inputs, a compute-node Python dispatcher and one Slurm entry script."""

import hashlib
import json
from pathlib import Path
from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files
from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script

DRIVER = Path(__file__).with_name("submission_dispatcher.py").read_text(encoding="utf-8")


def write_dispatch_entry(folder, first_script):
    """Write only the outer compute-job entry; input ZIP and receipts stay intact."""
    folder = Path(folder)
    script = folder / "submit_all.sh"
    dispatcher = folder / "dispatch_tasks.py"
    dispatcher.write_text(DRIVER, encoding="utf-8", newline="\n")
    partition = next(
        (line for line in first_script.splitlines() if line.startswith("#SBATCH --partition")), ""
    )
    activation = next(
        (
            line
            for line in first_script.splitlines()
            if line.startswith("source ") and "activate" in line
        ),
        "",
    )
    write_unix_shell_script(
        script,
        "#!/usr/bin/env bash\n#SBATCH --job-name=phase-dispatch\n#SBATCH -N 1\n#SBATCH -n 1\n#SBATCH --mem=2G\n#SBATCH --time=00:20:00\n#SBATCH --output=dispatch-%j.out\n#SBATCH --error=dispatch-%j.err\n"
        + (partition + "\n" if partition else "")
        + 'set -euo pipefail\n: "${SLURM_JOB_ID:?Use sbatch submit_all.sh}"\ncd -- "${SLURM_SUBMIT_DIR:?Missing Slurm submit directory}"\n'
        + (activation + "\n" if activation else "")
        + 'python3 dispatch_tasks.py "$PWD"\n',
    )
    return script, dispatcher


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
        from phase_agent.tools.remote.handoff_naming import handoff_label, package_folder

        label = handoff_label(directory, scripts)
        folder = package_folder(base / "archives" / "01-upload", label, key)
        archive = folder / "inputs.zip"
        if not archive.exists():
            create_bundle(portable, portable_files, archive, kind="inputs")
    script, dispatcher = write_dispatch_entry(
        folder, (directory / scripts[0]).read_text(encoding="utf-8")
    )
    return {
        "folder": str(folder),
        "upload_archive": str(archive),
        "submit_script": str(script),
        "command": "sbatch submit_all.sh",
        "dispatcher": str(dispatcher),
        "results_directory": str(directory / "results"),
        "return_directory": str(directory / "return-results"),
        "import_directory": str(directory),
        "job_count": len(scripts),
        "layout_version": 3,
        "package_fingerprint": key,
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

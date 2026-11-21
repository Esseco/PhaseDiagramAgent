"""Stage CPU-only recovery for one frozen, previously dispatched refresh."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
from tempfile import TemporaryDirectory
import zipfile

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.prepare_phase_analysis import copy_phase_runtime
from phase_agent.tools.remote.refresh_handoff_repair import _strip_publisher


def stage_phase_cpu_recovery(
    archive, archive_sha256, spec_sha256, task_checksums, remote_root, destination
):
    """Prepare a new CPU-only directory without touching state or remote files.

    The caller must verify scheduler terminal evidence and hold the project lock
    before publishing or registering this directory. No job is submitted here.
    """
    from phase_agent.tools.remote.input_validation import validate_handoff
    from phase_agent.tools.remote.phase_analysis_wave import GROUP_SIZE

    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    root = PurePosixPath(remote_root)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or not root.name.endswith("_input")
        or "\n" in remote_root
        or "\r" in remote_root
    ):
        raise ValueError("phase_recovery_invalid_remote_root")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", archive_sha256)
        or file_checksum(archive) != archive_sha256
    ):
        raise ValueError("phase_recovery_archive_changed")
    if not re.fullmatch(r"[0-9a-f]{64}", spec_sha256):
        raise ValueError("phase_recovery_invalid_spec_digest")
    if not task_checksums or any(not k or not v for k, v in task_checksums.items()):
        raise ValueError("phase_recovery_invalid_task_scope")
    validate_handoff(archive.parent)
    if destination.exists():
        raise ValueError("phase_recovery_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("phase_recovery_duplicate_archive_path")
        specs = [
            name
            for name in names
            if re.fullmatch(
                r"calculations/[^/]+/inputs/zz-phase-analysis-[^/]+/phase_wave.json", name
            )
        ]
        if len(specs) != 1:
            raise ValueError("phase_recovery_wave_ambiguous")
        spec_name = specs[0]
        spec_bytes = bundle.read(spec_name)
        if hashlib.sha256(spec_bytes).hexdigest() != spec_sha256:
            raise ValueError("phase_recovery_wave_changed")
        spec = json.loads(spec_bytes)
        rows = spec.get("tasks") or []
        if len(rows) != len(task_checksums) or len({r.get("task_id") for r in rows}) != len(rows):
            raise ValueError("phase_recovery_task_scope_changed")
        if {r.get("task_id"): r.get("task_checksum") for r in rows} != task_checksums:
            raise ValueError("phase_recovery_task_identity_changed")
        roots = set()
        for row in rows:
            path = PurePosixPath(row.get("result_path") or "")
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 5
                or path.parts[0] != "calculations"
                or path.parts[2] != "results"
                or path.name != "result.json"
            ):
                raise ValueError("phase_recovery_result_path_invalid")
            roots.add(PurePosixPath(*path.parts[:3]).as_posix())
        phase_dir = spec_name.rsplit("/", 1)[0]
        cpu = bundle.read(phase_dir + "/CPU.sh").decode("utf8")
        finalizer = bundle.read(phase_dir + "/CPU_finalize.sh").decode("utf8")
    groups = (len(rows) + GROUP_SIZE - 1) // GROUP_SIZE
    if groups <= 1 or f"#SBATCH --array=0-{groups - 1}" not in cpu:
        raise ValueError("phase_recovery_group_scope_changed")
    remote = shlex.quote(str(root))

    def rewrite(content):
        old = 'cd -- "$(dirname "$0")"'
        if (
            content.count(old) != 1
            or content.count("phase_wave.json ../../../.. --spec-checksum " + spec_sha256) != 1
        ):
            raise ValueError("phase_recovery_script_scope_changed")
        return content.replace(
            old, 'cd -- "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"'
        ).replace(
            "phase_wave.json ../../../.. --spec-checksum " + spec_sha256,
            "phase_wave.json " + remote + " --spec-checksum " + spec_sha256,
        )

    cpu = rewrite(cpu)
    finalizer = rewrite(_strip_publisher(finalizer))
    preflight = (
        "printf '%s  %s\\n' "
        + shlex.quote(archive_sha256)
        + " "
        + shlex.quote(str(root / "inputs.zip"))
        + " | sha256sum -c -\n"
    )
    cwd_line = 'cd -- "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"\n'
    cpu = cpu.replace(cwd_line, cwd_line + preflight, 1)
    finalizer = finalizer.replace(cwd_line, cwd_line + preflight, 1)
    if "--group-index" not in cpu or "--finalize" not in finalizer:
        raise ValueError("phase_recovery_invalid_script")
    finalizer += (
        "\n"
        + "\n".join(
            'python3 "$SLURM_SUBMIT_DIR/_round_transfer.py" '
            + remote
            + " "
            + shlex.quote(root_path)
            for root_path in sorted(roots)
        )
        + "\n"
    )
    with TemporaryDirectory(prefix=".phase-cpu-recovery-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        (stage / "phase_wave.json").write_bytes(spec_bytes)
        (stage / "CPU.sh").write_text(cpu, encoding="utf8", newline="\n")
        (stage / "CPU_finalize.sh").write_text(finalizer, encoding="utf8", newline="\n")
        copy_phase_runtime(stage / "runtime")
        shutil.copy2(Path(__file__).with_name("transfer_bundle.py"), stage / "_round_transfer.py")
        inventory = {
            p.relative_to(stage).as_posix(): file_checksum(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        if file_checksum(archive) != archive_sha256:
            raise ValueError("phase_recovery_archive_changed_during_stage")
        (stage / "recovery_manifest.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "source_archive_sha256": archive_sha256,
                    "spec_sha256": spec_sha256,
                    "task_count": len(rows),
                    "group_count": groups,
                    "remote_result_root": str(root),
                    "result_roots": sorted(roots),
                    "files": inventory,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf8",
        )
        from phase_agent.tools.remote.transfer_bundle import create_bundle

        files = [p for p in stage.rglob("*") if p.is_file()]
        create_bundle(stage, files, stage / "inputs.zip", kind="inputs")
        archive_digest = file_checksum(stage / "inputs.zip")
        stage.rename(destination)
    return {
        "directory": str(destination),
        "upload_archive": str(destination / "inputs.zip"),
        "upload_archive_sha256": archive_digest,
        "task_count": len(rows),
        "group_count": groups,
        "source_archive_sha256": archive_sha256,
        "spec_sha256": spec_sha256,
        "remote_result_root": str(root),
        "result_roots": sorted(roots),
    }

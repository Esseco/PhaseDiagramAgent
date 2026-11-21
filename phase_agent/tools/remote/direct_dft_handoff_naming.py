"""Name an unsubmitted, verified DIRECT DFT handoff without changing its inputs."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from tempfile import TemporaryDirectory
import zipfile

from phase_agent.tools.remote.input_validation import validate_handoff, verification_summary


def name_direct_dft_handoff(state, project_root, record_id, archive_sha256, remote_root):
    root = Path(project_root).resolve()
    remote = PurePosixPath(remote_root)
    if not remote.is_absolute() or ".." in remote.parts or str(remote) == "/":
        raise ValueError("invalid_remote_project_root")
    records = [r for r in state.get("action_records", []) if r.get("record_id") == record_id]
    if len(records) != 1 or records[0].get("status") != "prepared":
        raise ValueError("direct_dft_not_prepared")
    tasks = [t for t in state.get("tasks", []) if t.get("parent_decision_id") == record_id]
    receipt = (state.get("direct_sampling_results") or {}).get("epoch_refresh") or {}
    from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt

    selected = {
        r.get("candidate_id"): r.get("structure_sha256") for r in receipt.get("selected") or []
    }
    if (
        not valid_direct_receipt(receipt)
        or not selected
        or len(selected) != receipt.get("selected_count")
        or len(tasks) != len(selected)
        or {t.get("object_id") for t in tasks} != set(selected)
        or any(
            t.get("stage") != "dft_single_point"
            or t.get("status") != "pending"
            or t.get("job_id")
            or t.get("input_structure_sha256") != selected[t.get("object_id")]
            for t in tasks
        )
    ):
        raise ValueError("direct_dft_task_scope_mismatch")
    task_ids = {t["task_id"] for t in tasks}
    batches = [
        b for b in state.get("slurm_batches", []) if task_ids.intersection(b.get("task_ids") or [])
    ]
    if (
        not batches
        or any(b.get("status") != "prepared" or b.get("job_id") for b in batches)
        or set().union(*(set(b.get("task_ids") or []) for b in batches)) != task_ids
    ):
        raise ValueError("direct_dft_batch_not_unsubmitted")
    old = batches[0].get("handoff") or {}
    archive = Path(old.get("upload_archive") or "").resolve()
    if (
        not archive.is_relative_to(root / "submissions")
        or archive.name != "inputs.zip"
        or any((b.get("handoff") or {}).get("upload_archive") != str(archive) for b in batches)
        or sha256(archive.read_bytes()).hexdigest() != archive_sha256
    ):
        raise ValueError("direct_dft_handoff_binding_mismatch")
    validate_handoff(archive.parent)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith("/task.json")]
        packed = [json.loads(z.read(n)) for n in names]
        if (
            len(packed) != len(tasks)
            or {t.get("task_id") for t in packed} != task_ids
            or any(
                t.get("input_structure_sha256") != selected.get(t.get("object_id")) for t in packed
            )
        ):
            raise ValueError("direct_dft_archive_task_mismatch")
    epoch = archive.parent.parent.parent.parent.name
    if not re.fullmatch(r"epoch[0-9]+", epoch):
        raise ValueError("direct_dft_epoch_unknown")
    folder = archive.parent.parent / "ExtrapolationRecovery_DFTSinglePoint_Batch01_input"
    output = folder.with_name(folder.name[:-6] + "_output")
    remote_input = str(remote / epoch / folder.name)
    remote_output = str(remote / epoch / output.name)
    readme = (
        "# Extrapolation recovery: DFT single-point calculations\n\n"
        f"This batch contains {len(tasks)} DIRECT-selected recovery frames. "
        "It prepares static DFT energy and force labels; it does not submit jobs.\n\n"
        f"Input archive SHA256: `{archive_sha256}`\n\n"
        f"Remote input directory: `{remote_input}`\n\n"
        f"Remote result directory: `{remote_output}`\n\n"
        "Upload and HPC submission require separate user approval.\n"
    )
    names = ("inputs.zip", "submit_all.sh", "dispatch_tasks.py")
    if output.exists():
        raise ValueError("direct_dft_named_handoff_already_exists")
    if folder.exists():
        # Reuse only an exact, still-unregistered result of an interrupted naming.
        if (
            folder.is_symlink()
            or archive.parent.resolve() == folder.resolve()
            or not folder.is_dir()
            or {p.name for p in folder.iterdir()} != {*names, "README.md"}
            or any(
                (folder / name).is_symlink() or not (folder / name).is_file()
                for name in (*names, "README.md")
            )
            or any(
                (folder / name).read_bytes() != (archive.parent / name).read_bytes()
                for name in names
            )
            or (folder / "README.md").read_text(encoding="utf8") != readme
        ):
            raise ValueError("direct_dft_named_handoff_already_exists_or_mismatch")
        verification = verification_summary(validate_handoff(folder))
    else:
        with TemporaryDirectory(prefix=".direct-name-", dir=folder.parent) as tmp:
            staged = Path(tmp) / folder.name
            staged.mkdir()
            for name in names:
                shutil.copyfile(archive.parent / name, staged / name)
            if sha256((staged / "inputs.zip").read_bytes()).hexdigest() != archive_sha256:
                raise ValueError("direct_dft_copied_archive_changed")
            verification = verification_summary(validate_handoff(staged))
            (staged / "README.md").write_text(readme, encoding="utf8")
            staged.rename(folder)
    handoff = deepcopy(old)
    handoff.update(
        folder=str(folder),
        upload_archive=str(folder / "inputs.zip"),
        submit_script=str(folder / "submit_all.sh"),
        dispatcher=str(folder / "dispatch_tasks.py"),
        return_directory=str(output),
        verification=verification,
        remote_input_directory=remote_input,
        remote_output_directory=remote_output,
        description="Extrapolation recovery DIRECT structures: DFT single-point",
    )
    for batch in batches:
        batch.setdefault("superseded_handoffs", []).append(deepcopy(batch.get("handoff")))
        batch["handoff"] = deepcopy(handoff)
        batch["upload_archive"] = handoff["upload_archive"]
    state.setdefault("handoff_naming_history", []).append(
        {
            "record_id": record_id,
            "task_count": len(tasks),
            "previous_folder": str(archive.parent),
            "handoff": deepcopy(handoff),
            "submitted": False,
        }
    )
    return {
        "status": "prepared",
        "task_count": len(tasks),
        "handoff": handoff,
        "previous_folder": str(archive.parent),
        "submitted": False,
    }

"""Repair an unsubmitted DFT handoff without changing tasks or scientific inputs."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import zipfile

from phase_agent.tools.remote.handoff_naming import package_folder
from phase_agent.tools.remote.input_validation import validate_handoff, verification_summary
from phase_agent.tools.remote.transfer_bundle import MANIFEST, _safe_name, create_bundle


def _one_gpu_script(data):
    text = data.decode("utf-8")
    if not re.search(r"(?m)^#SBATCH --gres=gpu:1[ \t]*$", text):
        raise ValueError("dft_repair_requires_exactly_one_gpu")
    text, total = re.subn(
        r"(?m)^#SBATCH (?:-n[ \t]+|--ntasks[= ])[12][ \t]*$", "#SBATCH -n 1", text
    )
    text, per_node = re.subn(
        r"(?m)^#SBATCH --ntasks-per-node[= ][12][ \t]*$", "#SBATCH --ntasks-per-node=1", text
    )
    if total != 1 or per_node != 1:
        raise ValueError("dft_repair_ambiguous_mpi_directives")
    return text.encode("utf-8")


def repair_dft_handoff(
    state, project_root, record_id, expected_archive_sha256, expected_task_count, partition_counts=None
):
    """Stage and validate a new handoff; update only existing batch handoff bindings.

    The caller owns state and persists it under the project lock. Old inputs and
    handoff are retained. No task, model, result, cost, or remote job is changed.
    """
    if partition_counts is not None:
        if (not isinstance(partition_counts, dict) or not partition_counts
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                       or type(count) is not int or count <= 0
                       for name, count in partition_counts.items())
                or sum(partition_counts.values()) != expected_task_count):
            raise ValueError("dft_repair_invalid_partition_counts")
    root = Path(project_root).resolve()
    matches = [r for r in state.get("action_records", []) if r.get("record_id") == record_id]
    if len(matches) != 1 or matches[0].get("status") != "prepared":
        raise ValueError("dft_repair_requires_one_prepared_decision")
    record = matches[0]
    action = record.get("final_action") or {}
    if (
        action.get("tool") != "select_dft_candidates"
        or (record.get("human_feedback") or {}).get("decision") != "approve"
    ):
        raise ValueError("dft_repair_requires_approved_dft_decision")
    selected = [
        d
        for d in (action.get("parameters") or {}).get("decisions", [])
        if d.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    ]
    chosen = [d.get("candidate_id") for d in selected]
    stages = {
        d["candidate_id"]: {"DFT_SINGLE_POINT": "dft_single_point", "DFT_RELAX": "dft_relax"}[
            d["action"]
        ]
        for d in selected
    }
    tasks = [t for t in state.get("tasks", []) if t.get("parent_decision_id") == record_id]
    if (
        len(chosen) != expected_task_count
        or len(set(chosen)) != len(chosen)
        or len(tasks) != expected_task_count
        or len({t.get("task_id") for t in tasks}) != len(tasks)
        or {t.get("object_id") or t.get("structure_id") for t in tasks} != set(chosen)
    ):
        raise ValueError("dft_repair_task_scope_mismatch")
    if any(
        t.get("status") != "pending"
        or t.get("stage") != stages.get(t.get("object_id") or t.get("structure_id"))
        or t.get("job_id")
        for t in tasks
    ):
        raise ValueError("dft_repair_task_not_unsubmitted")
    if any(
        t.get("model_version") != state.get("active_model_version")
        or t.get("config_version") != state.get("confirmed_config_version")
        for t in tasks
    ):
        raise ValueError("dft_repair_model_or_config_changed")
    task_ids = {t["task_id"] for t in tasks}
    batches = [
        b for b in state.get("slurm_batches", []) if task_ids.intersection(b.get("task_ids") or [])
    ]
    if (
        not batches
        or any(b.get("status") != "prepared" or b.get("job_id") for b in batches)
        or set().union(*(set(b.get("task_ids") or []) for b in batches)) != task_ids
    ):
        raise ValueError("dft_repair_batch_not_unsubmitted_or_mixed")
    old = batches[0].get("handoff") or {}
    archive = Path(old.get("upload_archive") or "").resolve()
    if not archive.is_relative_to(root / "submissions") or archive.name != "inputs.zip":
        raise ValueError("dft_repair_archive_outside_project")
    if any((b.get("handoff") or {}).get("upload_archive") != str(archive) for b in batches):
        raise ValueError("dft_repair_multiple_handoffs")
    if old.get("repaired_from_archive_sha256") == expected_archive_sha256:
        if old.get("partition_counts") != partition_counts:
            raise ValueError("dft_repair_partition_counts_changed")
        validate_handoff(archive.parent)
        return {
            "status": "reused",
            "task_count": len(tasks),
            "handoff": deepcopy(old),
            "submitted": False,
        }
    if sha256(archive.read_bytes()).hexdigest() != expected_archive_sha256:
        raise ValueError("dft_repair_source_archive_changed")
    if validate_handoff(archive.parent)["status"] != "passed":
        raise ValueError("dft_repair_source_invalid")
    parent = archive.parent.parent
    match = re.fullmatch(r"(.+)_Batch\d+_input", archive.parent.name)
    if not match:
        raise ValueError("dft_repair_invalid_handoff_name")
    by_id = {t["task_id"]: t for t in tasks}
    with TemporaryDirectory(prefix="dft-gpu-repair-") as temp:
        staging = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            plan = json.loads(bundle.read("submission_plan.json"))
            scripts = plan.get("scripts") or []
            if len(scripts) != expected_task_count or len(set(scripts)) != len(scripts):
                raise ValueError("dft_repair_submission_scope_mismatch")
            partitions = {}
            if partition_counts:
                ordered = [name for name, count in partition_counts.items() for _ in range(count)]
                partitions = dict(zip(scripts, ordered, strict=True))
            observed = set()
            for script in scripts:
                path = str(_safe_name(script))
                if not path.endswith("/GPU.sh"):
                    raise ValueError("dft_repair_invalid_script")
                base = path.rsplit("/", 1)[0]
                if base + "/submit_gpu.sh" not in names:
                    raise ValueError("dft_repair_missing_submit_script")
                task = json.loads(bundle.read(base + "/task.json"))
                task_id = task.get("task_id")
                source = by_id.get(task_id) or {}
                if (
                    not source
                    or task_id in observed
                    or task.get("object_id") != source.get("object_id")
                    or task.get("stage") != source.get("stage")
                    or task.get("task_key") != source.get("task_key")
                    or task.get("model_version") != source.get("model_version")
                    or task.get("input_structure_sha256") != source.get("input_structure_sha256")
                    or sha256(bundle.read(base + "/initial_structure.vasp")).hexdigest()
                    != task.get("input_structure_sha256")
                ):
                    raise ValueError("dft_repair_task_binding_mismatch")
                observed.add(task_id)
            if observed != task_ids:
                raise ValueError("dft_repair_archive_task_mismatch")
            for name in names:
                if name == MANIFEST:
                    continue
                safe = _safe_name(name)
                data = bundle.read(name)
                if name in scripts or any(
                    name == s.rsplit("/", 1)[0] + "/submit_gpu.sh" for s in scripts
                ):
                    data = _one_gpu_script(data)
                    script = name if name in partitions else name.rsplit("/", 1)[0] + "/GPU.sh"
                    if script in partitions:
                        text, count = re.subn(
                            r"(?m)^#SBATCH --partition(?:=|\s+)[A-Za-z0-9_-]+[ \t]*$",
                            "#SBATCH --partition=" + partitions[script], data.decode("utf-8"),
                        )
                        if count != 1:
                            raise ValueError("dft_repair_ambiguous_partition")
                        data = text.encode("utf-8")
                dest = staging.joinpath(*safe.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        files = sorted(p for p in staging.rglob("*") if p.is_file())
        identity = sha256()
        for source in files:
            identity.update(source.relative_to(staging).as_posix().encode())
            identity.update(sha256(source.read_bytes()).digest())
        if sha256(archive.read_bytes()).hexdigest() != expected_archive_sha256:
            raise ValueError("dft_repair_source_archive_changed_during_read")
        key = identity.hexdigest()[:16]
        folder = package_folder(parent, match.group(1), key)
        if not folder.exists():
            with TemporaryDirectory(prefix=".dft-repair-", dir=parent) as tmp:
                new = Path(tmp) / folder.name
                new.mkdir()
                create_bundle(staging, files, new / "inputs.zip", kind="inputs")
                for name in ("submit_all.sh", "dispatch_tasks.py"):
                    shutil.copyfile(archive.parent / name, new / name)
                validate_handoff(new)
                new.rename(folder)
        verification = verification_summary(validate_handoff(folder))
    handoff = deepcopy(old)
    handoff.update(
        folder=str(folder),
        upload_archive=str(folder / "inputs.zip"),
        submit_script=str(folder / "submit_all.sh"),
        dispatcher=str(folder / "dispatch_tasks.py"),
        return_directory=str(folder.with_name(folder.name[:-6] + "_output")),
        package_fingerprint=key,
        verification=verification,
        repaired_from_archive_sha256=expected_archive_sha256,
        repair_profile="one_mpi_rank_per_gpu",
        partition_counts=deepcopy(partition_counts),
    )
    for batch in batches:
        batch.setdefault("superseded_handoffs", []).append(deepcopy(batch.get("handoff")))
        batch["handoff"] = deepcopy(handoff)
        batch["upload_archive"] = handoff["upload_archive"]
    return {
        "status": "prepared",
        "task_count": len(tasks),
        "handoff": handoff,
        "previous_folder": str(archive.parent),
        "submitted": False,
    }

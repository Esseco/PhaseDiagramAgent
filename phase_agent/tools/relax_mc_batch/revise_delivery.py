"""Revise only the GPU delivery settings of a frozen, unsent Relax–MC batch."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import zipfile

from phase_agent.tools.relax_mc_batch.files import balanced_batch_size, gpu_profile
from phase_agent.tools.remote.input_validation import validate_handoff, verification_summary
from phase_agent.tools.remote.transfer_bundle import (
    MANIFEST,
    _read_manifest,
    checksum,
    create_bundle,
)


def _revise_manifest(manifest, partitions):
    revised = deepcopy(manifest)
    profiles = revised["stage_profiles"]
    for stage in ("relax_and_feature", "deep_search"):
        gpu_profile(profiles, stage)["partition_pool"] = list(partitions)
    workload = revised["workload"]
    sizes = revised["config"]["supercomputer"]["batch_sizes"]
    counts = [workload["relax_structure_tasks"], *workload["mc_task_upper"]]
    limits = [sizes["relax_and_feature"], *([sizes["deep_search"]] * 3)]
    workload["gpu_jobs_upper_by_wave"] = [
        math.ceil(count / balanced_batch_size(count, limit))
        for count, limit in zip(counts, limits, strict=True)
    ]
    return revised


def _pending_scope(state, bundle_id, task_ids):
    tasks = [
        row
        for row in state.get("tasks", [])
        if row.get("relax_mc_bundle_id") == bundle_id or row.get("task_id") in task_ids
    ]
    if (
        len(tasks) != len(task_ids)
        or {row["task_id"] for row in tasks} != task_ids
        or any(
            row.get("relax_mc_bundle_id") != bundle_id
            or row.get("status") != "pending"
            or row.get("stage") != "relax_and_feature"
            or row.get("job_id")
            or row.get("submitted")
            or row.get("recovery_cancelled")
            or row.get("slurm_batch_id") != bundle_id
            for row in tasks
        )
    ):
        raise ValueError("relax_mc_revision_tasks_are_not_original_pending_scope")


def revise_delivery(state, project_root, *, bundle_id, source_sha256, confirmed_unsubmitted):
    if confirmed_unsubmitted is not True:
        raise ValueError("relax_mc_revision_requires_remote_unsubmitted_confirmation")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("relax_mc_revision_invalid_source_sha256")
    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    if (
        batch.get("status") != "prepared"
        or batch.get("submitted", False) is not False
        or batch.get("job_id")
        or batch.get("import_checksum")
        or batch.get("model_version") != state.get("active_model_version")
        or batch.get("config_version") != state.get("confirmed_config_version")
    ):
        raise ValueError("relax_mc_revision_requires_current_unsent_prepared_batch")
    previous = batch.get("delivery_revision")
    if previous:
        if (
            previous["source_sha256"] != source_sha256
            or checksum(batch["upload_archive"]) != previous["archive_sha256"]
            or validate_handoff(batch["upload_folder"], kind="relax_mc")["fingerprint"]
            != batch["verification"]["fingerprint"]
        ):
            raise ValueError("relax_mc_revision_existing_delivery_conflict")
        _pending_scope(state, bundle_id, set(previous["task_ids"]))
        return batch

    root = Path(batch["directory"]).resolve(strict=True)
    source = Path(batch["upload_archive"]).resolve(strict=True)
    from phase_agent.tools.relax_mc_batch.handoff import input_directory

    if (
        not root.is_relative_to(Path(project_root).resolve() / "submissions")
        or source != input_directory(root) / "inputs.zip"
        or checksum(source) != source_sha256
        or Path(batch["expected_results_archive"]).exists()
    ):
        raise ValueError("relax_mc_revision_original_delivery_conflict")
    outer = validate_handoff(source.parent, kind="relax_mc")
    if outer["fingerprint"] != batch["verification"]["fingerprint"]:
        raise ValueError("relax_mc_revision_original_verification_changed")

    with zipfile.ZipFile(source) as archive:
        inventory = _read_manifest(archive)["files"]
        manifest = json.loads(archive.read("batch.json"))
    for key in ("bundle_id", "model_version", "config_version", "plan_checksum"):
        if manifest[key] != batch[key]:
            raise ValueError("relax_mc_revision_original_plan_binding_changed")
    tasks = manifest["initial_state"]["tasks"]
    ids = {row["task_id"] for row in tasks}
    if not ids or len(ids) != len(tasks) or len(ids) != batch["workload"]["relax_structure_tasks"]:
        raise ValueError("relax_mc_revision_invalid_original_task_count")
    _pending_scope(state, bundle_id, ids)

    files = []
    for entry in inventory:
        file = (root / entry["path"]).resolve(strict=True)
        if not file.is_relative_to(root) or checksum(file) != entry["sha256"]:
            raise ValueError("relax_mc_revision_original_payload_changed")
        files.append(file)

    partitions = ("v100m3", "v100g32")
    if gpu_profile(manifest["stage_profiles"])["slurm_options"]["partition"] != partitions[0]:
        raise ValueError("relax_mc_revision_requires_bj_hpc_profile")
    revised = _revise_manifest(manifest, partitions)
    replacements = {}
    for name in ("run.py", "files.py"):
        code = Path(__file__).with_name(name)
        replacements[f"runtime/phase_agent/tools/relax_mc_batch/{name}"] = code.read_bytes()
        if name in revised["runtime_source_checksums"]:
            revised["runtime_source_checksums"][name] = checksum(code)
    replacements["batch.json"] = json.dumps(revised, ensure_ascii=False, sort_keys=True).encode()

    target = source.parent.with_name(root.name + "_GPUv2_input")
    if target.exists():
        raise FileExistsError("relax_mc_revision_destination_exists; inspect before retry")
    with TemporaryDirectory(prefix=".gpu-", dir=target.parent) as temporary:
        staging = Path(temporary) / target.name
        staging.mkdir()
        for name in ("run_upload.py", "submit_all.sh"):
            shutil.copyfile(source.parent / name, staging / name)
        converted = staging / "inputs.zip"
        create_bundle(root, files, converted, kind="inputs", payloads=replacements)
        verification = verification_summary(validate_handoff(staging, kind="relax_mc"))
        with zipfile.ZipFile(converted) as archive:
            actual = {row["path"]: row["sha256"] for row in _read_manifest(archive)["files"]}
        expected = {entry["path"]: entry["sha256"] for entry in inventory}
        if actual.keys() != expected.keys() or any(
            actual[name] != digest for name, digest in expected.items() if name not in replacements
        ):
            raise ValueError("relax_mc_revision_unexpected_payload_change")
        if checksum(source) != source_sha256:
            raise ValueError("relax_mc_revision_original_archive_changed")
        archive_sha256 = checksum(converted)
        staging.rename(target)

    batch.update(
        upload_folder=str(target),
        upload_archive=str(target / "inputs.zip"),
        script_path=str(target / "submit_all.sh"),
        verification=verification,
        workload=revised["workload"],
        delivery_revision={
            "source_sha256": source_sha256,
            "source_upload_folder": str(source.parent),
            "archive_sha256": archive_sha256,
            "partitions": list(partitions),
            "task_ids": sorted(ids),
            "remote_output_folder_name": target.name.removesuffix("_input") + "_output",
        },
    )
    return batch


def revise_registered_delivery(control, **arguments):
    from filelock import FileLock

    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.tools.remote.submission_layout import write_submission_index
    from phase_agent.tools.state.execution_receipts import recovery_report
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version

    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        before = read_identity(path, strong=True)
        state = read_json(path)
        if state.get("pending_execution_policies") or recovery_report(path, state)["unsettled"]:
            raise ValueError("relax_mc_revision_pending_or_unsettled_action")
        current = (state.get("relax_mc_batches") or {}).get(arguments["bundle_id"]) or {}
        existing = current.get("delivery_revision")
        root = path.parent.parent
        batch = revise_delivery(state, root, **arguments)
        if not existing:
            write_json_if_version(path, state, before)
            control._codex_context_cache = None
        warning = None
        try:
            write_submission_index(root / "submissions", state)
        except (OSError, ValueError) as error:
            warning = str(error)
        return {
            "status": "already_revised" if existing else "revised",
            "bundle_id": batch["bundle_id"],
            "upload_folder": batch["upload_folder"],
            "archive_sha256": batch["delivery_revision"]["archive_sha256"],
            "remote_output_folder_name": batch["delivery_revision"]["remote_output_folder_name"],
            "workload": batch["workload"],
            "index_warning": warning,
            "submitted": False,
        }


def minimize_delivery(state, project_root, *, bundle_id, source_sha256, confirmed_unsubmitted):
    """Conservatively revise a frozen unsent package without altering task identities."""
    from phase_agent.tools.relax_mc_batch.minimal_payload import reduced_payload
    from phase_agent.tools.relax_mc_batch.handoff import input_directory

    if confirmed_unsubmitted is not True:
        raise ValueError("relax_mc_revision_requires_remote_unsubmitted_confirmation")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("relax_mc_revision_invalid_source_sha256")
    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    if (
        batch.get("status") != "prepared"
        or batch.get("submitted", False) is not False
        or batch.get("job_id")
        or batch.get("import_checksum")
        or batch.get("model_version") != state.get("active_model_version")
        or batch.get("config_version") != state.get("confirmed_config_version")
    ):
        raise ValueError("relax_mc_revision_requires_current_unsent_prepared_batch")
    previous = batch.get("minimal_delivery_revision")
    if previous:
        if (
            previous["source_sha256"] != source_sha256
            or checksum(batch["upload_archive"]) != previous["archive_sha256"]
            or validate_handoff(batch["upload_folder"], kind="relax_mc")["fingerprint"]
            != batch["verification"]["fingerprint"]
        ):
            raise ValueError("relax_mc_minimal_revision_conflict")
        _pending_scope(state, bundle_id, set(previous["task_ids"]))
        return batch
    root = Path(batch["directory"]).resolve(strict=True)
    source = Path(batch["upload_archive"]).resolve(strict=True)
    if (
        not root.is_relative_to(Path(project_root).resolve() / "submissions")
        or source != input_directory(root) / "inputs.zip"
        or checksum(source) != source_sha256
        or Path(batch["expected_results_archive"]).exists()
    ):
        raise ValueError("relax_mc_revision_original_delivery_conflict")
    if (
        validate_handoff(source.parent, kind="relax_mc")["fingerprint"]
        != batch["verification"]["fingerprint"]
    ):
        raise ValueError("relax_mc_revision_original_verification_changed")
    with zipfile.ZipFile(source) as archive:
        inventory = _read_manifest(archive)["files"]
        manifest = json.loads(archive.read("batch.json"))
        sources = {
            entry["path"].removeprefix("runtime/"): archive.read(entry["path"])
            for entry in inventory
            if entry["path"].startswith("runtime/phase_agent/") and entry["path"].endswith(".py")
        }
    for key in ("bundle_id", "model_version", "config_version", "plan_checksum"):
        if manifest[key] != batch[key]:
            raise ValueError("relax_mc_revision_original_plan_binding_changed")
    tasks = manifest["initial_state"]["tasks"]
    ids = {row["task_id"] for row in tasks}
    if not ids or len(ids) != len(tasks) or len(ids) != batch["workload"]["relax_structure_tasks"]:
        raise ValueError("relax_mc_revision_invalid_original_task_count")
    _pending_scope(state, bundle_id, ids)
    for entry in inventory:
        file = (root / entry["path"]).resolve(strict=True)
        if not file.is_relative_to(root) or checksum(file) != entry["sha256"]:
            raise ValueError("relax_mc_revision_original_payload_changed")
    from phase_agent.tools.relax_mc_batch.runtime_inventory import (
        configured_worker_roots,
        runtime_python_inventory,
    )

    worker = manifest.get("config", {}).get("supercomputer", {}).get("worker") or {}
    runtime_paths = runtime_python_inventory(sources, extra_roots=configured_worker_roots(worker))
    kept, revised, aliases = reduced_payload(
        inventory, manifest, runtime_python_paths=runtime_paths
    )
    target = source.parent.with_name(root.name + "_Minimal_input")
    if target.exists():
        raise FileExistsError("relax_mc_revision_destination_exists; inspect before retry")
    with TemporaryDirectory(prefix=".minimal-", dir=target.parent) as temporary:
        staging = Path(temporary) / target.name
        staging.mkdir()
        for name in ("run_upload.py", "submit_all.sh"):
            shutil.copyfile(source.parent / name, staging / name)
        files = [root / entry["path"] for entry in kept]
        replacement = json.dumps(revised, ensure_ascii=False, sort_keys=True).encode()
        create_bundle(
            root, files, staging / "inputs.zip", kind="inputs", payloads={"batch.json": replacement}
        )
        verification = verification_summary(validate_handoff(staging, kind="relax_mc"))
        if checksum(source) != source_sha256:
            raise ValueError("relax_mc_revision_original_archive_changed")
        archive_sha256 = checksum(staging / "inputs.zip")
        staging.rename(target)
    batch.update(
        upload_folder=str(target),
        upload_archive=str(target / "inputs.zip"),
        script_path=str(target / "submit_all.sh"),
        verification=verification,
        minimal_delivery_revision={
            "source_sha256": source_sha256,
            "source_upload_folder": str(source.parent),
            "archive_sha256": archive_sha256,
            "task_ids": sorted(ids),
            "deduplicated_inputs": len(aliases),
            "remote_output_folder_name": target.name.removesuffix("_input") + "_output",
        },
    )
    return batch


def minimize_registered_delivery(control, **arguments):
    """Register one conservative revision under the existing project transaction lock."""
    from filelock import FileLock
    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.tools.state.execution_receipts import recovery_report
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version

    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        before = read_identity(path, strong=True)
        state = read_json(path)
        if state.get("pending_execution_policies") or recovery_report(path, state)["unsettled"]:
            raise ValueError("relax_mc_revision_pending_or_unsettled_action")
        existing = ((state.get("relax_mc_batches") or {}).get(arguments["bundle_id"]) or {}).get(
            "minimal_delivery_revision"
        )
        batch = minimize_delivery(state, path.parent.parent, **arguments)
        if not existing:
            try:
                write_json_if_version(path, state, before)
            except Exception:
                target = Path(batch["upload_folder"]).resolve(strict=True)
                source = Path(batch["directory"]).resolve(strict=True)
                previous_folder = Path(batch["minimal_delivery_revision"]["source_upload_folder"])
                if (
                    target != source.with_name(source.name + "_Minimal_input")
                    or {p.name for p in target.iterdir()}
                    != {"inputs.zip", "run_upload.py", "submit_all.sh"}
                    or checksum(target / "inputs.zip")
                    != batch["minimal_delivery_revision"]["archive_sha256"]
                    or any(
                        checksum(target / name) != checksum(previous_folder / name)
                        for name in ("run_upload.py", "submit_all.sh")
                    )
                ):
                    raise RuntimeError("relax_mc_minimal_revision_cleanup_requires_manual_review")
                shutil.rmtree(target)
                raise
            control._codex_context_cache = None
        return {
            "status": "already_revised" if existing else "revised",
            "bundle_id": batch["bundle_id"],
            "upload_folder": batch["upload_folder"],
            "archive_sha256": batch["minimal_delivery_revision"]["archive_sha256"],
            "deduplicated_inputs": batch["minimal_delivery_revision"]["deduplicated_inputs"],
            "submitted": False,
        }

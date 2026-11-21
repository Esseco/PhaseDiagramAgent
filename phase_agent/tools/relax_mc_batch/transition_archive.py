"""Verify the frozen and returned Relax phase for an explicit policy transition."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from phase_agent.tools.relax_mc_batch.threshold_transition import reassess_verified_task
from phase_agent.tools.remote.integrity import verify_result
from phase_agent.tools.remote.transfer_bundle import checksum, validate_bundle


def _member(path):
    """Accept only portable file names, even when a zip stays unextracted."""
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("transition_member_invalid")
    p = PurePosixPath(path)
    if p.is_absolute() or any(part in (".", "..") for part in p.parts):
        raise ValueError("transition_member_invalid")
    return p.as_posix()


def _json(archive, name):
    data = archive.read(_member(name))
    return json.loads(data.decode("utf-8-sig"))


def verify_transition(state, bundle_id, raw_path, raw_sha256, receipts_path, receipts_sha256):
    """Return a bound review; never modify state, artifacts, tasks or jobs.

    Scheduler states in the receipt are external observations, not an assertion
    that this function queried Slurm. Actual work remains behind approval.
    """
    record = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    if (
        record.get("bundle_id") != bundle_id
        or record.get("status") in {"cancelled", "superseded"}
        or state.get("active_model_version") != record.get("model_version")
        or state.get("confirmed_config_version") != record.get("config_version")
    ):
        raise ValueError("transition_batch_or_current_version_mismatch")
    frozen = Path(record["upload_archive"])
    raw = Path(raw_path)
    receipts_file = Path(receipts_path)
    if checksum(raw) != raw_sha256 or checksum(receipts_file) != receipts_sha256:
        raise ValueError("transition_evidence_checksum_mismatch")
    validate_bundle(raw, extra_result_suffixes=(".csv", ".bin"))
    receipts = json.loads(receipts_file.read_text(encoding="utf-8-sig"))
    if receipts.get("source_sha256") != raw_sha256:
        raise ValueError("transition_receipt_source_mismatch")
    recorded = [t for t in state.get("tasks") or [] if t.get("relax_mc_bundle_id") == bundle_id]
    if not recorded or len({t.get("task_id") for t in recorded}) != len(recorded):
        raise ValueError("transition_registered_tasks_invalid")
    project_tasks = {t["task_id"]: t for t in recorded}
    if any(t.get("stage") != "relax_and_feature" for t in recorded):
        raise ValueError("transition_requires_relax_only_registered_scope")
    with zipfile.ZipFile(raw) as returned, zipfile.ZipFile(frozen) as inputs:
        raw_batch = returned.read("batch.json")
        if inputs.read("batch.json") != raw_batch:
            raise ValueError("transition_frozen_batch_mismatch")
        inventory = {r["path"]: r for r in _json(inputs, "transfer_manifest.json")["files"]}
        if hashlib.sha256(raw_batch).hexdigest() != inventory["batch.json"]["sha256"]:
            raise ValueError("transition_frozen_manifest_invalid")
        batch = json.loads(raw_batch.decode("utf-8-sig"))
        for key in ("bundle_id", "plan_checksum", "model_version", "config_version"):
            if batch.get(key) != (bundle_id if key == "bundle_id" else record.get(key)):
                raise ValueError("transition_batch_binding_mismatch:" + key)
        initial = batch["initial_state"]["tasks"]
        if len(initial) != len(project_tasks) or {t.get("task_id") for t in initial} != set(
            project_tasks
        ):
            raise ValueError("transition_initial_task_scope_mismatch")
        source_state = _json(returned, "workflow_state.json")
        tasks = source_state.get("tasks") or []
        if len(tasks) != len(project_tasks) or {t.get("task_id") for t in tasks} != set(
            project_tasks
        ):
            raise ValueError("transition_return_task_scope_mismatch")
        batches = source_state.get("slurm_batches") or []
        job_map = {str(row["job_id"]): set(row["task_ids"]) for row in batches}
        observed = {
            str(row["job_evidence"]["job_id"]): row["job_evidence"] for row in receipts["jobs"]
        }
        if (
            len(job_map) != len(batches)
            or len(observed) != len(receipts["jobs"])
            or set(job_map) != set(observed)
            or any(
                item.get("state") != "COMPLETED" or set(item.get("task_ids") or []) != job_map[job]
                for job, item in observed.items()
            )
            or sum(len(x) for x in job_map.values()) != len(tasks)
            or set().union(*job_map.values()) != set(project_tasks)
        ):
            raise ValueError("transition_scheduler_evidence_mismatch")
        rows = []
        identity_mismatches = []
        with tempfile.TemporaryDirectory(prefix="phase-transition-") as temporary:
            tmp = Path(temporary)
            for task in tasks:
                registered = project_tasks[task["task_id"]]
                for key in (
                    "task_id",
                    "task_key",
                    "stage",
                    "structure_id",
                    "branch_id",
                    "model_version",
                    "config_version",
                    "relax_mc_bundle_id",
                ):
                    if task.get(key) != registered.get(key):
                        raise ValueError("transition_task_binding_mismatch:" + key)
                worker = _json(returned, task["input_path"])
                for key in (
                    "task_id",
                    "task_key",
                    "task_checksum",
                    "model_version",
                    "config_version",
                    "structure_id",
                ):
                    if worker.get(key) != task.get(key):
                        raise ValueError("transition_worker_binding_mismatch:" + key)
                input_path = _member(task["structure_path"])
                initial_bytes = inputs.read(input_path)
                initial_sha = hashlib.sha256(initial_bytes).hexdigest()
                if initial_sha != inventory[input_path]["sha256"]:
                    raise ValueError("transition_initial_checksum_invalid")
                result_member = _member(task["result_path"])
                parent = PurePosixPath(result_member).parent
                result_bytes = returned.read(result_member)
                marker_bytes = returned.read(str(parent / "task.finished.json"))
                result_path = tmp / "result.json"
                marker_path = tmp / "task.finished.json"
                result_path.write_bytes(result_bytes)
                marker_path.write_bytes(marker_bytes)
                checked = verify_result(result_path, marker_path, task)
                if not checked["valid"]:
                    raise ValueError(
                        "transition_result_invalid:" + task["task_id"] + ":" + checked["reason"]
                    )
                result = checked["result"]
                output = result.get("outputs") or {}
                final_sha = hashlib.sha256(
                    returned.read(str(parent / "initial_relaxed.vasp"))
                ).hexdigest()
                if final_sha != output.get("structure_checksum"):
                    raise ValueError("transition_final_checksum_invalid")
                row = reassess_verified_task(task, result)
                from pymatgen.io.vasp import Poscar
                from Process_AL_MC.MC_sample import _relaxation_input_identity

                try:
                    structure = Poscar.from_str(initial_bytes.decode("utf-8-sig")).structure
                except (ValueError, UnicodeError) as error:
                    raise ValueError("transition_initial_structure_invalid") from error
                if _relaxation_input_identity(structure) != row["monitor_structure_id"]:
                    identity_mismatches.append(
                        {
                            "task_id": task["task_id"],
                            "input_member": input_path,
                            "initial_sha256": initial_sha,
                        }
                    )
                rows.append(
                    {
                        **row,
                        "branch_id": task["branch_id"],
                        "initial_sha256": initial_sha,
                        "final_sha256": final_sha,
                        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
                    }
                )
    if len({r["monitor_structure_id"] for r in rows}) != len(rows):
        raise ValueError("transition_duplicate_structure_identity")
    facts = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "model_version": record["model_version"],
        "config_version": record["config_version"],
        "plan_checksum": record["plan_checksum"],
        "original_threshold": 1.0,
        "effective_threshold": 3.0,
        "task_count": len(rows),
        "original_failure_count": sum(r["original_failure"] for r in rows),
        "effective_failure_count": sum(r["effective_failure"] for r in rows),
        "raw_archive_path": str(raw.resolve()),
        "raw_archive_sha256": raw_sha256,
        "frozen_archive_path": str(frozen.resolve()),
        "frozen_archive_sha256": checksum(frozen),
        "scheduler_receipts_path": str(receipts_file.resolve()),
        "scheduler_receipts_sha256": receipts_sha256,
        "scheduler_states_source": "externally_observed",
        "rows": sorted(rows, key=lambda r: r["task_id"]),
    }
    digest = hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return {
        **facts,
        "review_digest": digest,
        "status": "verified_relax_phase",
        "submitted": False,
        "identity_check": {
            "status": "remote_input_evidence_required" if identity_mismatches else "matched",
            "mismatches": identity_mismatches,
        },
    }


def register_transition(state, review):
    """Record only the validated policy review; leave scientific task state intact."""
    if review.get("status") != "verified_relax_phase" or review.get("submitted") is not False:
        raise ValueError("transition_review_invalid")
    current = deepcopy(state)
    batch = current["relax_mc_batches"][review["bundle_id"]]
    existing = batch.get("threshold_transition")
    if existing and existing.get("review_digest") != review.get("review_digest"):
        raise ValueError("transition_existing_review_conflict")
    batch["threshold_transition"] = {**deepcopy(review), "status": "verified_awaiting_continuation"}
    return current


def validate_input_attestation(review, attestation_path, attestation_sha256):
    """Bind externally observed remote file hashes to each unmatched input."""
    path = Path(attestation_path)
    if checksum(path) != attestation_sha256:
        raise ValueError("transition_input_attestation_checksum_mismatch")
    evidence = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        evidence.get("original_archive_sha256") != review["frozen_archive_sha256"]
        or evidence.get("remote_input_fingerprint") != review["frozen_archive_sha256"]
        or evidence.get("method") != "sha256sum"
        or not evidence.get("host")
    ):
        raise ValueError("transition_remote_archive_fingerprint_mismatch")
    mismatches = review["identity_check"]["mismatches"]
    files = evidence.get("files") or []
    observed = {
        PurePosixPath(item.get("remote_path", "")).name: item.get("sha256") for item in files
    }
    if (
        len(observed) != len(files)
        or len(observed) != len(mismatches)
        or any(
            observed.get(PurePosixPath(row["input_member"]).name) != row["initial_sha256"]
            for row in mismatches
        )
    ):
        raise ValueError("transition_remote_input_hash_mismatch")
    return {
        "status": "verified_remote_bytes",
        "source": "externally_observed",
        "attestation_path": str(path.resolve()),
        "attestation_sha256": attestation_sha256,
        "review_digest": review["review_digest"],
        "task_ids": sorted(row["task_id"] for row in mismatches),
        "count": len(mismatches),
        "host": evidence["host"],
    }

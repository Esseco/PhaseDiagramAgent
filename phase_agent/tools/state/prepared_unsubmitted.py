"""Bound, expiring human-observed evidence for unsubmitted DFT handoffs.

This evidence only changes the training wait decision. It does not change task
status, settlement, the result collection gate, or submission authorization.
"""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import zipfile

TTL = timedelta(minutes=30)
FIELD = "prepared_unsubmitted_evidence"


def _now(value=None):
    return value or datetime.now(timezone.utc)


def _timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    return result if result.tzinfo is not None else None


def _binding(task, batch):
    handoff = batch.get("handoff") or {}
    archive = handoff.get("upload_archive")
    if (
        task.get("stage") != "dft_single_point"
        or task.get("status") != "pending"
        or task.get("slurm_job_id")
        or task.get("job_id")
        or batch.get("status") != "prepared"
        or batch.get("job_id")
        or batch.get("model_version") != task.get("model_version")
        or batch.get("config_version") != task.get("config_version")
        or task.get("task_id") not in (batch.get("task_ids") or [])
        or not task.get("task_checksum")
        or not task.get("input_structure_sha256")
        or not archive
        or not Path(archive).is_file()
    ):
        return None
    path = Path(archive)
    folder = handoff.get("folder")
    if not folder or path.parent.resolve() != Path(folder).resolve():
        return None
    if (path.parent / ".submitted_jobs.json").exists():
        return None
    input_path = Path(task.get("input_path") or "")
    try:
        data = input_path.read_bytes()
        spec = json.loads(data)
        geometry = (input_path.parent / "initial_structure.vasp").read_bytes()
    except (OSError, ValueError, TypeError):
        return None
    if (
        spec.get("task_id") != task.get("task_id")
        or spec.get("batch_id") != task.get("batch_id")
        or spec.get("task_checksum") != task.get("task_checksum")
        or spec.get("input_structure_sha256") != task.get("input_structure_sha256")
        or sha256(geometry).hexdigest() != task.get("input_structure_sha256")
    ):
        return None
    return {
        "task_id": task["task_id"],
        "batch_id": task.get("batch_id"),
        "task_checksum": task["task_checksum"],
        "input_structure_sha256": task["input_structure_sha256"],
        "model_version": task.get("model_version"),
        "config_version": task.get("config_version"),
        "archive": str(path),
        "archive_sha256": sha256(path.read_bytes()).hexdigest(),
        "handoff_folder": handoff.get("folder"),
        "input_path": str(input_path),
        "input_file_sha256": sha256(data).hexdigest(),
    }


def _verify_archive(bindings):
    """Validate task identity and physical input bytes inside the frozen package."""
    expected = {x["task_id"]: x for x in bindings}
    with zipfile.ZipFile(bindings[0]["archive"]) as archive:
        seen = set()
        for name in archive.namelist():
            if not name.endswith("/task.json"):
                continue
            task = json.loads(archive.read(name))
            tid = task.get("task_id")
            if tid not in expected:
                continue
            if tid in seen:
                raise ValueError("unsubmitted_evidence_duplicate_archive_task")
            seen.add(tid)
            item = expected[tid]
            if (
                task.get("task_checksum") != item["task_checksum"]
                or task.get("input_structure_sha256") != item["input_structure_sha256"]
                or task.get("batch_id") != item["batch_id"]
                or sha256(
                    archive.read(name.rsplit("/", 1)[0] + "/initial_structure.vasp")
                ).hexdigest()
                != item["input_structure_sha256"]
            ):
                raise ValueError("unsubmitted_evidence_archive_identity_mismatch")
        if seen != set(expected):
            raise ValueError("unsubmitted_evidence_archive_tasks_missing")


def create_record(state, observation, *, now=None):
    required = {
        "task_ids",
        "observed_at",
        "server",
        "remote_input_directory",
        "scheduler_history_start",
        "directory_absent",
        "jobs_absent",
        "source",
        "user_message",
        "evidence_note",
    }
    if not isinstance(observation, dict) or set(observation) != required:
        raise ValueError("unsubmitted_evidence_invalid_fields")
    ids = observation["task_ids"]
    observed = _timestamp(observation["observed_at"])
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != len(set(ids))
        or not all(isinstance(x, str) and x for x in ids)
        or observed is None
        or not timedelta(0) <= _now(now) - observed <= TTL
        or observation["directory_absent"] is not True
        or observation["jobs_absent"] is not True
        or observation["source"] not in {"electerm_ssh", "ssh"}
        or not isinstance(observation["server"], str)
        or not observation["server"].strip()
        or not isinstance(observation["user_message"], str)
        or not observation["user_message"].strip()
        or not isinstance(observation["evidence_note"], str)
        or not observation["evidence_note"].strip()
        or not isinstance(observation["scheduler_history_start"], str)
        or not observation["scheduler_history_start"]
        or not isinstance(observation["remote_input_directory"], str)
        or not observation["remote_input_directory"].startswith("/")
    ):
        raise ValueError("unsubmitted_evidence_invalid_observation")
    tasks = state.get("tasks") or []
    batches = state.get("slurm_batches") or []
    root_value = ((state.get("confirmed_config") or {}).get("storage") or {}).get("workspace_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("unsubmitted_evidence_project_root_missing")
    root = Path(root_value).resolve()
    by_task = {t.get("task_id"): t for t in tasks}
    by_batch = {b.get("batch_id"): b for b in batches}
    if len(by_task) != len(tasks) or len(by_batch) != len(batches):
        raise ValueError("unsubmitted_evidence_duplicate_identity")
    bindings = []
    for tid in ids:
        task = by_task.get(tid) or {}
        item = _binding(task, by_batch.get(task.get("batch_id")) or {})
        if not item:
            raise ValueError("unsubmitted_evidence_task_not_prepared")
        try:
            Path(item["archive"]).resolve().relative_to(root)
            Path(task["input_path"]).resolve().relative_to(root)
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("unsubmitted_evidence_cross_project_path") from error
        bindings.append(item)
    if (
        len({x["archive_sha256"] for x in bindings}) != 1
        or len({x["handoff_folder"] for x in bindings}) != 1
    ):
        raise ValueError("unsubmitted_evidence_multiple_handoffs")
    try:
        start = date.fromisoformat(observation["scheduler_history_start"])
    except ValueError as error:
        raise ValueError("unsubmitted_evidence_invalid_history_start") from error
    archive_created = datetime.fromtimestamp(
        Path(bindings[0]["archive"]).stat().st_mtime, timezone.utc
    ).date()
    if start > archive_created or start > observed.date():
        raise ValueError("unsubmitted_evidence_history_too_short")
    try:
        _verify_archive(bindings)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("unsubmitted_evidence_archive_invalid") from error
    folder = Path(bindings[0]["handoff_folder"] or "").name
    if not folder or observation["remote_input_directory"].rstrip("/").split("/")[-1] != folder:
        raise ValueError("unsubmitted_evidence_remote_path_mismatch")
    return {
        "observation": dict(observation),
        "bindings": bindings,
        "project_root": str(root),
        "active_model_version": state.get("active_model_version"),
        "confirmed_config_version": state.get("confirmed_config_version"),
    }


def verified_ids(state, *, now=None):
    valid = set()
    tasks = state.get("tasks") or []
    batches = state.get("slurm_batches") or []
    by_task = {t.get("task_id"): t for t in tasks}
    by_batch = {b.get("batch_id"): b for b in batches}
    if len(by_task) != len(tasks) or len(by_batch) != len(batches):
        return valid
    for record in state.get(FIELD) or []:
        if not isinstance(record, dict):
            continue
        observation = record.get("observation") or {}
        observed = _timestamp(observation.get("observed_at"))
        if (
            observed is None
            or not timedelta(0) <= _now(now) - observed <= TTL
            or record.get("active_model_version") != state.get("active_model_version")
            or record.get("confirmed_config_version") != state.get("confirmed_config_version")
        ):
            continue
        try:
            current = create_record(state, observation, now=now)
        except (OSError, ValueError, TypeError):
            continue
        if current == record:
            valid.update(x["task_id"] for x in current["bindings"])
    return valid

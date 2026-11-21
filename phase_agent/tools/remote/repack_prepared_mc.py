"""Rebuild one proven, unsubmitted MC handoff using the normal packager."""

from collections import Counter
from pathlib import Path


def validate_prepared_mc_scope(
    state, events, *, decision_id, allocation_checksum, package_fingerprint
):
    claimed = [x.get("payload") or {} for x in events if x.get("event") == "claimed"]
    returned = [x.get("payload") or {} for x in events if x.get("event") == "tool_returned"]
    if len(claimed) != 1 or len(returned) != 1:
        raise ValueError("mc_repack_receipt_ambiguous")
    action = claimed[0].get("action") or {}
    identity = claimed[0].get("identity") or {}
    evidence = returned[0].get("evidence") or {}
    refs = evidence.get("returned_task_refs") or []
    preview = (action.get("parameters") or {}).get("budget_preview") or {}
    ids = [x.get("task_id") for x in refs]
    targets = action.get("target_ids") or []
    if (
        identity.get("invocation_id") != decision_id
        or returned[0].get("identity") != identity
        or action.get("tool") != "allocate_mc_bohb"
        or returned[0].get("outcome") != "completed"
        or evidence.get("result_status") != "completed"
        or preview.get("allocation_checksum") != allocation_checksum
        or not ids
        or None in ids
        or len(set(ids)) != len(ids)
        or not targets
        or len(set(targets)) != len(targets)
        or any(
            x.get("stage") != "deep_search"
            or x.get("status") != "pending"
            or x.get("parent_decision_id") != decision_id
            for x in refs
        )
    ):
        raise ValueError("mc_repack_receipt_binding_invalid")
    tasks = [t for t in state.get("tasks") or [] if t.get("parent_decision_id") == decision_id]
    by_id = {r["task_id"]: r for r in refs}
    fields = (
        "task_key",
        "stage",
        "status",
        "model_version",
        "config_version",
        "parent_decision_id",
        "input_path",
    )
    if (
        len(tasks) != len(ids)
        or {t.get("task_id") for t in tasks} != set(ids)
        or {t.get("branch_id") for t in tasks} != set(targets)
        or any(any(t.get(k) != by_id[t["task_id"]].get(k) for k in fields) for t in tasks)
    ):
        raise ValueError("mc_repack_task_scope_changed")
    if any(
        t.get("status") != "pending"
        or t.get("stage") != "deep_search"
        or t.get("job_id")
        or t.get("remote_job_id")
        for t in tasks
    ):
        raise ValueError("mc_repack_task_already_submitted")
    batches = [
        b
        for b in state.get("slurm_batches") or []
        if set(b.get("task_ids") or []).intersection(ids)
    ]
    counts = Counter(i for b in batches for i in b.get("task_ids") or [])
    if (
        not batches
        or set(counts) != set(ids)
        or any(n != 1 for n in counts.values())
        or any(
            b.get("status") != "prepared"
            or b.get("job_id")
            or (b.get("handoff") or {}).get("package_fingerprint") != package_fingerprint
            for b in batches
        )
    ):
        raise ValueError("mc_repack_batch_scope_changed")
    folders = {(b.get("handoff") or {}).get("folder") for b in batches}
    if len(folders) != 1 or None in folders:
        raise ValueError("mc_repack_handoff_scope_changed")
    return batches, tasks, action


def _verify_repacked_archive(old_handoff, new_handoff, task_count):
    import hashlib
    import zipfile

    old_root, new_root = Path(old_handoff["folder"]), Path(new_handoff["folder"])
    for filename in ("submit_all.sh", "dispatch_tasks.py"):
        if (old_root / filename).read_bytes() != (new_root / filename).read_bytes():
            raise ValueError("mc_repack_dispatcher_changed")
    with (
        zipfile.ZipFile(old_handoff["upload_archive"]) as before,
        zipfile.ZipFile(new_handoff["upload_archive"]) as after,
    ):
        old = {x.filename for x in before.infolist() if not x.is_dir()}
        new = {x.filename for x in after.infolist() if not x.is_dir()}
        removed = old - new
        if (
            new - old
            or len(removed) != task_count
            or any(not name.endswith("/_handoff_refresh_inline.py") for name in removed)
        ):
            raise ValueError("mc_repack_unexpected_file_scope")
        for name in (old & new) - {"transfer_manifest.json"}:
            if (
                hashlib.sha256(before.read(name)).digest()
                != hashlib.sha256(after.read(name)).digest()
            ):
                raise ValueError("mc_repack_input_changed:" + name)
        if after.testzip() is not None:
            raise ValueError("mc_repack_archive_corrupt")
    if (
        Path(new_handoff["upload_archive"]).stat().st_size
        >= Path(old_handoff["upload_archive"]).stat().st_size
    ):
        raise ValueError("mc_repack_not_smaller")


def rebuild_prepared_mc_handoff(
    state, events, *, decision_id, allocation_checksum, package_fingerprint
):
    from phase_agent.tools.remote.submission_handoff import handoff_root, prepare_handoff

    batches, tasks, _action = validate_prepared_mc_scope(
        state,
        events,
        decision_id=decision_id,
        allocation_checksum=allocation_checksum,
        package_fingerprint=package_fingerprint,
    )
    directory = handoff_root(Path(batches[0]["upload_directory"]).parent.parent)
    if any(handoff_root(Path(b["upload_directory"]).parent.parent) != directory for b in batches):
        raise ValueError("mc_repack_epoch_mismatch")
    old = batches[0]["handoff"]
    handoff = prepare_handoff(directory, [b["script_path"] for b in batches])
    if not handoff or handoff.get("package_fingerprint") == package_fingerprint:
        raise ValueError("mc_repack_did_not_reduce_payload")
    _verify_repacked_archive(old, handoff, len(tasks))
    handoff["batch_ids"] = sorted(b["batch_id"] for b in batches)
    for batch in batches:
        batch["handoff"] = handoff
    state.setdefault("handoff_repack_history", []).append(
        {
            "decision_id": decision_id,
            "allocation_checksum": allocation_checksum,
            "task_ids": sorted(t["task_id"] for t in tasks),
            "old_handoff": old,
            "new_handoff": handoff,
            "reason": "omit_unused_mc_refresh_helpers",
        }
    )
    return state, handoff

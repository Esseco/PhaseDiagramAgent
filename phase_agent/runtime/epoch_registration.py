"""Reconcile an already prepared recovery refresh without moving artifacts."""

from pathlib import Path
import json

from filelock import FileLock


def reconcile_state(
    state, *, model_version, old_epoch, new_epoch, round_digest, plan_checksum, archive_sha256
):
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.tools.remote.transfer_bundle import checksum
    from phase_agent.tools.workflows.model_refresh_state import fingerprint

    if (
        type(old_epoch) is not int
        or type(new_epoch) is not int
        or old_epoch < 0
        or new_epoch != old_epoch + 1
    ):
        raise ValueError("epoch_registration_invalid_epochs")
    cycle = state.get("extrapolation_recovery_cycle") or {}
    refresh = state.get("model_refresh") or {}
    plan = refresh.get("plan") or {}
    approved = refresh.get("approved_scope") or {}
    auth = refresh.get("input_authorization") or {}

    def valid_plan(value):
        contents = dict(value)
        return (
            contents.pop("checksum", None) == plan_checksum
            and fingerprint(contents) == plan_checksum
        )

    if (
        state.get("active_model_version") != model_version
        or cycle.get("replacement_model_version") != model_version
        or cycle.get("status") != "awaiting_new_model_refresh"
        or cycle.get("policy_version") != "epoch_refresh_v1"
        or cycle.get("round_evidence_digest") != round_digest
        or refresh.get("new_model_version") != model_version
        or refresh.get("status") != "waiting_results"
        or refresh.get("handoff_status") != "complete"
        or plan.get("checksum") != plan_checksum
        or approved.get("plan_checksum") != plan_checksum
        or approved.get("model_version") != model_version
        or auth.get("status") != "approved"
        or not valid_plan(plan)
        or not valid_plan(auth.get("plan") or {})
    ):
        raise ValueError("epoch_registration_evidence_mismatch")
    parent = cycle.get("training_parent_model_version")
    if not parent or model_epoch(state, parent) != f"epoch{old_epoch}":
        raise ValueError("epoch_registration_parent_mismatch")
    layout = state.get("upload_layout") or {}
    epochs = layout.get("scientific_epochs") or {}
    current = epochs.get(model_version)
    if current not in (old_epoch, new_epoch):
        raise ValueError("epoch_registration_current_mismatch")
    versions = set(epochs) | set(layout.get("model_rounds") or {})
    if any(k != model_version and model_epoch(state, k) == f"epoch{new_epoch}" for k in versions):
        raise ValueError("epoch_registration_target_conflict")
    ids = refresh.get("task_ids") or []
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("epoch_registration_invalid_tasks")
    id_set = set(ids)
    tasks = [t for t in state.get("tasks", []) if t.get("task_id") in id_set]
    if (
        len(tasks) != len(ids)
        or {t.get("task_id") for t in tasks} != id_set
        or any(
            t.get("status") != "pending"
            or t.get("model_version") != model_version
            or t.get("model_refresh_id") != plan_checksum
            for t in tasks
        )
    ):
        raise ValueError("epoch_registration_tasks_changed")
    if any(t.get("result_path") and Path(t["result_path"]).is_file() for t in tasks):
        raise ValueError("epoch_registration_results_already_exist")
    batch_ids = {t.get("batch_id") for t in tasks}
    if any(
        t.get("batch_id") in batch_ids and t.get("task_id") not in id_set
        for t in state.get("tasks", [])
    ):
        raise ValueError("epoch_registration_batch_scope_mismatch")
    batches = [b for b in state.get("slurm_batches", []) if b.get("batch_id") in batch_ids]
    if (
        None in batch_ids
        or len(batches) != len(batch_ids)
        or any(
            b.get("status") != "prepared"
            or b.get("job_id")
            or b.get("model_version") != model_version
            for b in batches
        )
    ):
        raise ValueError("epoch_registration_batches_changed")
    handoffs = [b.get("handoff") or {} for b in batches]
    if (
        not handoffs
        or any(h != handoffs[0] for h in handoffs)
        or set(handoffs[0].get("batch_ids") or []) != batch_ids
    ):
        raise ValueError("epoch_registration_handoff_mismatch")
    archive = Path(handoffs[0].get("upload_archive") or "")
    if not archive.is_file() or checksum(archive).lower() != archive_sha256.lower():
        raise ValueError("epoch_registration_archive_mismatch")
    receipt = {
        "model_version": model_version,
        "old_epoch": old_epoch,
        "new_epoch": new_epoch,
        "round_digest": round_digest,
        "plan_checksum": plan_checksum,
        "archive_sha256": archive_sha256.lower(),
        "upload_archive": str(archive),
        "import_directory": handoffs[0].get("import_directory"),
        "return_directory": handoffs[0].get("return_directory"),
        "task_count": len(ids),
        "batch_count": len(batch_ids),
        "paths_preserved": True,
    }
    prior = refresh.get("epoch_registration")
    if current == new_epoch:
        if prior != receipt:
            raise ValueError("epoch_registration_already_conflicts")
        return state, {"status": "already_registered", **receipt}
    if prior:
        raise ValueError("epoch_registration_existing_receipt")
    state["upload_layout"]["scientific_epochs"][model_version] = new_epoch
    state["model_refresh"]["epoch_registration"] = receipt
    return state, {"status": "registered", **receipt}


def reconcile(control, **args):
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery
    from phase_agent.tools.remote.transfer_bundle import checksum

    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=30),
    ):
        before = checksum(path)
        state = json.loads(path.read_text(encoding="utf8"))
        updated, receipt = reconcile_state(state, **args)
        if receipt["status"] == "registered":
            commit_result_recovery(
                path,
                updated,
                {},
                expected_hashes={str(path): before},
                evidence={"operation": "reconcile_recovery_refresh_epoch", **receipt},
            )
        control._codex_context_cache = None
        return receipt

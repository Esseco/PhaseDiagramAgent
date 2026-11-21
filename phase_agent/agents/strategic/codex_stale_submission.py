"""Verify a blocked refresh proposal without replaying its old checkpoint."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.persistence.compact_checkpoints import CompressedSerde
from phase_agent.persistence.sqlite_paths import readonly_uri


def _proven_superseded_refresh(old, state):
    """Prove the old completed refresh was followed by a recorded activation.

    Everything outside the explicit transition fields must still be equal.
    The old frame itself remains in the immutable checkpoint.
    """
    changed = {
        "active_model",
        "active_model_version",
        "analysis_publications",
        "candidate_models",
        "model_activation_history",
        "model_refresh",
        "model_registry",
        "model_transitions",
        "model_update_epochs",
        "new_dft_records",
        "phase_diagrams",
        "phase_epoch_publications",
        "relax_mc_batches",
        "remote_finetune_jobs",
        "training_handoff_history",
        "upload_layout",
    }
    if set(old) - set(state):
        return False
    for key in set(old) - changed:
        if key == "tasks":

            def without_terminal_evidence(items):
                return [
                    {k: v for k, v in task.items() if k != "scheduler_terminal_evidence"}
                    for task in items or []
                ]

            if without_terminal_evidence(old[key]) != without_terminal_evidence(state.get(key)):
                return False
        elif old[key] != state.get(key):
            return False
    before, after = old.get("active_model_version"), state.get("active_model_version")
    refresh, next_refresh = old.get("model_refresh") or {}, state.get("model_refresh") or {}
    if (
        not before
        or not after
        or before == after
        or refresh.get("status") != "completed"
        or refresh.get("new_model_version") != before
        or next_refresh.get("old_model_version") != before
        or next_refresh.get("new_model_version") != after
    ):
        return False
    activation = state.get("model_activation_history") or []
    if not any(
        item.get("old_model_version") == before
        and item.get("new_model_version") == after
        and str(item.get("approval_reason") or "").strip()
        for item in activation
    ):
        return False
    for key in (
        "model_activation_history",
        "model_transitions",
        "model_update_epochs",
        "training_handoff_history",
    ):
        prior, current = old.get(key) or [], state.get(key) or []
        if current[: len(prior)] != prior:
            return False
    for key in ("candidate_models",):
        if any((state.get(key) or {}).get(k) != v for k, v in (old.get(key) or {}).items()):
            return False
    for key in ("model_registry", "remote_finetune_jobs", "relax_mc_batches"):
        for identity, prior in (old.get(key) or {}).items():
            current = (state.get(key) or {}).get(identity)
            if not isinstance(prior, dict) or not isinstance(current, dict):
                return False
            ignored = {"phase_analysis"} if key == "relax_mc_batches" else {"status"}
            if {k: v for k, v in prior.items() if k not in ignored} != {
                k: v for k, v in current.items() if k not in ignored
            }:
                return False
            if key == "model_registry" and prior.get("status") != current.get("status"):
                if not (
                    identity == before
                    and prior.get("status") == "active"
                    and current.get("status") == "historical_active"
                ):
                    return False
            if key == "remote_finetune_jobs" and prior.get("status") != current.get("status"):
                if not (
                    prior.get("status") == "inputs_prepared"
                    and current.get("status") == "abandoned"
                ):
                    return False
    if (state.get("active_model") or {}).get("version") != after:
        return False
    task_map = {task.get("task_id"): task for task in state.get("tasks") or []}
    if len(task_map) != len(state.get("tasks") or []):
        return False
    if not refresh.get("task_ids") or any(
        task_map.get(task_id, {}).get("status") not in {"completed", "failed", "cancelled"}
        for task_id in refresh["task_ids"]
    ):
        return False
    training = {row.get("data_id"): row for row in state.get("dft_training_records") or []}
    if any(
        not row.get("data_id") or training.get(row["data_id"]) != row
        for row in old.get("new_dft_records") or []
    ):
        return False
    for kind, diagram in (old.get("phase_diagrams") or {}).items():
        current = (state.get("phase_diagrams") or {}).get(kind) or {}
        if diagram.get("version") != current.get("version") or diagram.get(
            "entries"
        ) != current.get("entries"):
            return False
    return True


def _stale_dft_selection_checkpoint(
    state_path, session, *, checkpoint_id, request_id, checkpoint_sha256
):
    """Resume a proven pre-action selection checkpoint without replaying observation."""
    from threading import RLock
    from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde
    from phase_agent.graphs.project.runner import (
        _business_state_is_clean_for_stale_checkpoint,
    )
    from phase_agent.tools.step_runner.file_protocol import read_json

    path = Path(state_path).resolve()
    state = read_json(path, {}) or {}
    if not _business_state_is_clean_for_stale_checkpoint(path, state=state):
        raise ValueError("stale_checkpoint_business_state_unsettled")
    database = path.parent / "langgraph_lifecycle.sqlite"
    with closing(sqlite3.connect(readonly_uri(database), uri=True)) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,parent_checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        if (
            not row
            or row[1] != checkpoint_id
            or hashlib.sha256(row[4]).hexdigest() != checkpoint_sha256
        ):
            raise ValueError("submission_checkpoint_changed")
        decoder = CompressedSerde(JsonPlusSerializer())

        def decode(item):
            return (
                decoder.loads_typed((item[3].removeprefix("phase-cas-v1:"), item[4])).get(
                    "channel_values"
                )
                or {}
            )

        values = decode(row)
        frame = values.get("business_frame") or {}
        feedback = frame.get("feedback") or {}
        if (
            values.get("request_id") != request_id
            or values.get("phase") != "wait_checked"
            or values.get("current_node") != "results_wait_gate"
            or values.get("node_status") != "completed"
            or "branch:to:analyze" not in values
            or values.get("response")
            or values.get("completed_response")
            or frame.get("automatic_results")
            or frame.get("combined_results")
            or frame.get("training_returns")
            or frame.get("training_handoffs")
            or frame.get("recovered_count")
            or frame.get("manual_wait")
            or feedback.get("recorded")
            or feedback.get("rejected")
        ):
            raise ValueError("stale_dft_checkpoint_not_pre_action")
        parent = row[2]
        initial = None
        while parent:
            prior = conn.execute(
                "SELECT thread_id,checkpoint_id,parent_checkpoint_id,type,checkpoint FROM checkpoints "
                "WHERE thread_id=? AND checkpoint_ns='' AND checkpoint_id=?",
                (row[0], parent),
            ).fetchone()
            if not prior:
                break
            initial_values = decode(prior)
            if (
                initial_values.get("request_id") == request_id
                and initial_values.get("phase") == "initialized"
                and initial_values.get("current_node") == "initialize_confirmed_run"
            ):
                initial = initial_values
                break
            parent = prior[2]
        if not initial:
            raise ValueError("stale_dft_initial_checkpoint_missing")
        loaded = (initial.get("business_frame") or {}).get("loaded_state")
        if not isinstance(loaded, dict):
            raise ValueError("checkpoint_saved_business_state_missing")
        serde = CheckpointBlobSerde(decoder, conn, RLock())
        if serde._restore_state(loaded) != state:
            raise ValueError("checkpoint_business_state_not_proven_persisted")
    receipt_path = path.parent / "execution_receipts.sqlite"
    if not receipt_path.exists():
        raise ValueError("execution_journal_required")
    with closing(sqlite3.connect(readonly_uri(receipt_path), uri=True)) as conn:
        match = conn.execute(
            "SELECT 1 FROM execution_events WHERE instr(action_id, ?) > 0 OR instr(payload, ?) > 0 LIMIT 1",
            (request_id, request_id),
        ).fetchone()
    if match:
        raise ValueError("dft_action_has_execution_receipt")
    if any(
        row.get("record_id") == request_id or row.get("invocation_id") == request_id
        for row in state.get("action_records") or []
    ):
        raise ValueError("dft_action_has_business_record")
    return {
        "classification": "stale_dft_selection",
        "resume_from_wait_checked": True,
        "resume_next": "analyze",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "request_id": request_id,
        "thread_id": row[0],
        "executed": False,
    }


def _reconciled_pre_dispatch_checkpoint(
    state_path, *, checkpoint_id, request_id, checkpoint_sha256
):
    """Verify a pre-dispatch execution_reconciliation_required block has since been reconciled."""
    from phase_agent.tools.state.execution_receipts import recovery_report
    from phase_agent.tools.step_runner.file_protocol import read_json

    path = Path(state_path).resolve()
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
    if (
        not row
        or row[1] != checkpoint_id
        or hashlib.sha256(row[3]).hexdigest() != checkpoint_sha256
    ):
        raise ValueError("submission_checkpoint_changed")
    payload = CompressedSerde(JsonPlusSerializer()).loads_typed(
        (row[2].removeprefix("phase-cas-v1:"), row[3])
    )
    values = payload.get("channel_values") or {}
    if values.get("request_id") != request_id:
        raise ValueError("stale_checkpoint_not_initialized")
    state = read_json(path, {}) or {}
    if recovery_report(path, state)["unsettled"]:
        raise ValueError("execution_reconciliation_required")
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "request_id": request_id,
        "executed": False,
        "classification": "pre_dispatch_execution_reconciliation_cleared",
    }


def stale_submission_checkpoint(
    state_path, session, *, checkpoint_id, request_id, checkpoint_sha256
):
    action = (session.get("draft") or {}).get("action") or {}
    if session.get("status") == "submitting" and action.get("tool") == "select_dft_candidates":
        return _stale_dft_selection_checkpoint(
            state_path,
            session,
            checkpoint_id=checkpoint_id,
            request_id=request_id,
            checkpoint_sha256=checkpoint_sha256,
        )
    branch_interrupted = (
        session.get("status") == "submission_uncertain"
        and not session.get("submission")
        and action.get("tool") == "generate_branches"
        and action.get("operation") == "generate_branches"
    )
    if (
        session.get("status") != "submitting"
        and not branch_interrupted
        and (session.get("submission") or {}).get("status") != "execution_reconciliation_required"
    ):
        raise ValueError("stale_refresh_submission_required")
    if (session.get("submission") or {}).get(
        "status"
    ) == "execution_reconciliation_required" and not action.get("task_key"):
        return _reconciled_pre_dispatch_checkpoint(
            state_path,
            checkpoint_id=checkpoint_id,
            request_id=request_id,
            checkpoint_sha256=checkpoint_sha256,
        )
    refresh = (
        action.get("tool") == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "model_refresh_inputs"
    )
    if not (refresh or branch_interrupted) or not action.get("task_key"):
        raise ValueError("stale_refresh_submission_required")
    path = Path(state_path).resolve()
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
    if (
        not row
        or row[1] != checkpoint_id
        or hashlib.sha256(row[3]).hexdigest() != checkpoint_sha256
    ):
        raise ValueError("submission_checkpoint_changed")
    payload = CompressedSerde(JsonPlusSerializer()).loads_typed(
        (row[2].removeprefix("phase-cas-v1:"), row[3])
    )
    values = payload.get("channel_values") or {}
    frame = values.get("business_frame") or {}
    initialized = (
        values.get("phase") == "initialized"
        and values.get("current_node") == "initialize_confirmed_run"
        and not frame.get("collection_report")
    )
    assessed = (
        values.get("phase") == "assessed"
        and values.get("current_node") == "assess_and_export_round"
        and "branch:to:react" in values
        and not frame.get("combined_results")
        and not frame.get("training_returns")
        and not frame.get("training_handoffs")
        and not frame.get("recovered_count")
        and not (frame.get("feedback") or {}).get("recorded")
        and not (frame.get("feedback") or {}).get("rejected")
    )
    if (
        values.get("request_id") != request_id
        or not (initialized or assessed)
        or values.get("node_status") != "completed"
        or values.get("completed_response")
        or values.get("response")
        or frame.get("automatic_results")
    ):
        raise ValueError("stale_checkpoint_not_initialized")
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.graphs.project.runner import _business_state_is_clean_for_stale_checkpoint

    state = read_json(path, {}) or {}
    # An initialized frame can contain state that never reached state.json.
    # Do not infer supersession from a newer model or a clear receipt report.
    from threading import RLock
    from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde

    loaded = frame.get("loaded_state")
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint_saved_business_state_missing")
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        serde = CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), conn, RLock())
        loaded = serde._restore_state(loaded)
        if assessed:
            for part in ("pre_reconciled", "feedback"):
                saved = (frame.get(part) or {}).get("state")
                if not isinstance(saved, dict) or serde._restore_state(saved) != state:
                    raise ValueError("checkpoint_analysis_not_proven_persisted")
    if loaded != state and not _proven_superseded_refresh(loaded, state):
        raise ValueError("checkpoint_business_state_not_proven_persisted")
    if not _business_state_is_clean_for_stale_checkpoint(path, state=state):
        raise ValueError("stale_checkpoint_business_state_unsettled")
    key = action["task_key"]
    if any(
        task.get("task_key") == key or task.get("scientific_parent_task_key") == key
        for task in state.get("tasks") or []
    ):
        raise ValueError("refresh_action_has_registered_tasks")
    if any(
        (record.get("final_action") or {}).get("task_key") == key
        for record in state.get("action_records") or []
    ):
        raise ValueError("refresh_action_has_business_record")
    if branch_interrupted and any(
        row.get("task_key") == key for row in state.get("generation_history") or []
    ):
        raise ValueError("branch_action_has_generation_history")
    # The frozen preparation must never have crossed its side-effect boundary.
    receipt_path = path.parent / "execution_receipts.sqlite"
    if not receipt_path.exists():
        raise ValueError("execution_journal_required")
    with closing(sqlite3.connect(readonly_uri(receipt_path), uri=True)) as conn:
        rows = conn.execute("SELECT payload FROM execution_events WHERE event='claimed'").fetchall()
    for (raw,) in rows:
        event = json.loads(raw)
        if (event.get("action") or {}).get("task_key") == action["task_key"]:
            raise ValueError("refresh_action_has_execution_receipt")
    return {
        "classification": "stale_lifecycle_block",
        "resume_from_assessed": assessed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "request_id": request_id,
        "thread_id": row[0],
        "action_task_key": action["task_key"],
        "executed": False,
    }


def _proven_region_reconciliation_delta(old, current, invocation, state_path):
    """Prove that only one reviewed, zero-cost staging disposition changed."""
    from phase_agent.tools.budget.settle_budget import settle_budget
    from phase_agent.tools.state.abandoned_region_generation import (
        inspect_abandoned_region_generation,
    )

    allowed = {
        "action_records",
        "budget_reservations",
        "budget_usage",
        "effective_decisions",
        "execution_reconciliations",
        "invocations",
        "recovery_review_history",
    }
    if set(old) != set(current) or any(old[k] != current[k] for k in set(old) - allowed):
        return False
    prior_audits = old.get("execution_reconciliations") or {}
    audits = current.get("execution_reconciliations") or {}
    audit = audits.get(invocation) or {}
    if (
        {k: v for k, v in audits.items() if k != invocation} != prior_audits
        or invocation in prior_audits
        or audit.get("resolution") != "verified_abandoned_staging"
        or (audit.get("identity") or {}).get("invocation_id") != invocation
        or not audit.get("reviewed_at")
        or not audit.get("plan_id")
    ):
        return False
    identity = audit["identity"]
    evidence = audit.get("evidence") or {}
    checks, attestation = evidence.get("checks") or {}, evidence.get("attestation") or {}
    staging = checks.get("abandoned_staging") or {}
    if (
        staging.get("kind") != "abandoned_region_generation_staging"
        or attestation.get("output_inventory") != "abandoned_local_staging"
        or attestation.get("external_jobs") != "not_applicable"
        or type(attestation.get("external_cost")) not in (int, float)
        or attestation.get("external_cost") != 0
    ):
        return False
    try:
        if inspect_abandoned_region_generation(state_path, old, identity) != staging:
            return False
    except (OSError, ValueError, TypeError, KeyError):
        return False
    key, plan_id = staging.get("task_key"), audit["plan_id"]
    before, after = old.get("action_records") or [], current.get("action_records") or []
    if len(before) != len(after):
        return False
    matched = 0
    for a, b in zip(before, after):
        if a.get("record_id") == invocation:
            matched += 1
            if (
                a.get("status") != "failed"
                or b.get("status") != "reconciled_abandoned_staging"
                or {k: v for k, v in a.items() if k != "status"}
                != {k: v for k, v in b.items() if k != "status"}
            ):
                return False
        elif a != b:
            return False
    if matched != 1:
        return False
    subset = {
        k: old.get(k) for k in ("budget_reservations", "budget_usage", "reserved_relative_cost")
    }
    reservation = (subset.get("budget_reservations") or {}).get(key) or {}
    if (
        reservation.get("status") != "reserved"
        or type(reservation.get("reserved_cost", reservation.get("relative_cost")))
        not in (int, float)
        or reservation.get("reserved_cost", reservation.get("relative_cost")) != 0
    ):
        return False
    settled = settle_budget(
        subset,
        task_key=key,
        settlement_id=plan_id,
        task_status="failed",
        actual_cost=0,
        cost_source="human_verified_abandoned_staging",
        failure_reason=attestation.get("evidence_note"),
    )["state"]
    settled["budget_reservations"][key].pop("reconciliation_required", None)
    if any(settled.get(k) != current.get(k) for k in subset):
        return False
    decisions = dict(old.get("effective_decisions") or {})
    if key in decisions:
        decisions[key] = {**decisions[key], "status": "failed", "verified_abandoned_staging": True}
    if decisions != (current.get("effective_decisions") or {}):
        return False
    invocations = dict(old.get("invocations") or {})
    invocations[invocation] = {
        "status": "reconciled_abandoned_staging",
        "record_id": invocation,
        "reconciliation_id": plan_id,
        "submitted": False,
    }
    invocations[plan_id] = {
        "status": "reconciled_abandoned_staging",
        "reconciliation_id": invocation,
    }
    if invocations != (current.get("invocations") or {}):
        return False
    history = list(old.get("recovery_review_history") or [])
    history.append({"plan_id": plan_id, "decision": "confirm_sensitive", "audit": audit})
    return history == (current.get("recovery_review_history") or [])


def blocked_energy_submission_checkpoint(
    state_path, session, *, checkpoint_id, request_id, checkpoint_sha256
):
    """Prove an energy-repair proposal was blocked before dispatch; keep checkpoint intact."""
    action = (session.get("draft") or {}).get("action") or {}
    if (
        (session.get("submission") or {}).get("status") != "execution_reconciliation_required"
        or action.get("tool") != "prepare_local_batch_files"
        or (action.get("parameters") or {}).get("mode") != "mc_energy_repair_inputs"
        or not action.get("task_key")
    ):
        raise ValueError("blocked_energy_submission_required")
    path = Path(state_path).resolve()
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        if (
            not row
            or row[1] != checkpoint_id
            or hashlib.sha256(row[3]).hexdigest() != checkpoint_sha256
        ):
            raise ValueError("submission_checkpoint_changed")
        payload = CompressedSerde(JsonPlusSerializer()).loads_typed(
            (row[2].removeprefix("phase-cas-v1:"), row[3])
        )
        values = payload.get("channel_values") or {}
        frame = values.get("business_frame") or {}
        result = values.get("completed_response") or {}
        issues = (result.get("recovery_report") or {}).get("unsettled") or []
        ids = {(item.get("identity") or {}).get("invocation_id") for item in issues}
        if (
            values.get("request_id") != request_id
            or values.get("phase") != "wait_checked"
            or values.get("current_node") != "results_wait_gate"
            or values.get("node_status") != "waiting"
            or (values.get("response") or {}).get("status") != "execution_reconciliation_required"
            or result.get("status") != "execution_reconciliation_required"
            or result.get("steps_executed") != 0
            or result.get("submitted") is not False
            or len(ids) != 1
            or None in ids
            or not issues
            or frame.get("automatic_results")
            or frame.get("combined_results")
            or frame.get("training_returns")
            or frame.get("training_handoffs")
            or frame.get("recovered_count")
            or (frame.get("feedback") or {}).get("recorded")
            or (frame.get("feedback") or {}).get("rejected")
        ):
            raise ValueError("blocked_checkpoint_not_proven")
        from threading import RLock
        from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde

        serde = CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), conn, RLock())
        old = serde._restore_state(result.get("state"))
        feedback = serde._restore_state((frame.get("feedback") or {}).get("state"))
        if not isinstance(old, dict) or feedback != old:
            raise ValueError("blocked_checkpoint_feedback_not_persisted")
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.graphs.project.runner import _business_state_is_clean_for_stale_checkpoint

    state = read_json(path, {}) or {}
    invocation = next(iter(ids))
    if not _proven_region_reconciliation_delta(old, state, invocation, path):
        raise ValueError("blocked_checkpoint_business_state_not_proven")
    if not _business_state_is_clean_for_stale_checkpoint(path, state=state):
        raise ValueError("stale_checkpoint_business_state_unsettled")
    key = action["task_key"]
    if (
        any(
            t.get("task_key") == key or t.get("scientific_parent_task_key") == key
            for t in state.get("tasks") or []
        )
        or any(
            (r.get("final_action") or {}).get("task_key") == key
            for r in state.get("action_records") or []
        )
        or state.get("pending_execution_policies")
    ):
        raise ValueError("blocked_energy_action_has_business_record")
    receipt = path.parent / "execution_receipts.sqlite"
    if not receipt.exists():
        raise ValueError("execution_journal_required")
    with closing(sqlite3.connect(readonly_uri(receipt), uri=True)) as conn:
        for (raw,) in conn.execute("SELECT payload FROM execution_events WHERE event='claimed'"):
            if (json.loads(raw).get("action") or {}).get("task_key") == key:
                raise ValueError("blocked_energy_action_has_execution_receipt")
    return {
        "classification": "blocked_energy_before_dispatch",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "request_id": request_id,
        "thread_id": row[0],
        "action_task_key": key,
        "executed": False,
    }

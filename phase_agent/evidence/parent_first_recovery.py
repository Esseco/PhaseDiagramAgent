"""Bind completed parent-first DFT labels to the invalidated MLIP epoch."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path


def small_parent_structure_ids(manager_data):
    """Return sorted candidate structure IDs with det_H <= 6 for initial cold start."""
    structures = (manager_data or {}).get("structures") or {}
    return [
        sid
        for sid, row in sorted(structures.items())
        if isinstance(row, dict) and int(row.get("det_H") or 1) <= 6
    ]


def initial_perturbation_direct_ready(state, manager_data, *, allowed=None):
    """True when a fresh project has det_H <= 6 parents ready for initial_direct."""
    from phase_agent.tools.workflows.model_refresh_state import refresh_active

    if allowed is not None and "prepare_local_batch_files" not in allowed:
        return False
    current = state or {}
    data = manager_data or {}
    if data.get("branches") or not small_parent_structure_ids(data):
        return False
    return (
        not current.get("dft_dataset_records")
        and not current.get("initial_direct_sampling")
        and not current.get("generation_history")
        and not current.get("tasks")
        and not refresh_active(current)
    )


def valid_parent_first_cycle(state):
    from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result

    cycle = state.get("extrapolation_recovery_cycle") or {}
    if cycle.get("recovery_route") != "parent_first_perturbation":
        return False
    batch = (state.get("parent_first_recovery_batches") or {}).get(cycle.get("batch_id")) or {}
    try:
        checked = validate_parent_first_result(batch["input_dir"], batch["selection_dir"])
        tasks = {row.get("task_id"): row for row in state.get("tasks") or []}
        ids = batch["dft_task_ids"]
        expected = {
            (r["input_structure_sha256"], r["structure_id"]) for r in (tasks[i] for i in ids)
        }
        selected = {(r["structure_sha256"], r["candidate_id"]) for r in cycle["selected"]}
        return (
            batch.get("status") in {"waiting_dft_results", "results_received"}
            and batch.get("sampling_result_sha256") == checked["result_sha256"]
            and cycle.get("direct_source_digest") == checked["result_sha256"]
            and batch.get("approval_record_id") == cycle.get("recovery_approval_record_id")
            and batch.get("model_version") == cycle.get("invalid_model_version")
            and len(ids) == checked["target"] == len(selected) == len(expected)
            and selected == expected
            and {r["structure_sha256"] for r in checked["selected"]} == {x[0] for x in selected}
        )
    except (ValueError, OSError, KeyError, TypeError):
        return False


def advance_parent_first_recovery(state):
    """After all DFT results return, set up the existing training cycle."""
    from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled
    from phase_agent.evidence.dft_failure_policy import unreviewed_failure_batches
    from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result

    current = deepcopy(state)
    tasks = {row.get("task_id"): row for row in current.get("tasks") or []}
    for batch_id, batch in (current.get("parent_first_recovery_batches") or {}).items():
        if batch.get("status") != "waiting_dft_results":
            continue
        ids = batch.get("dft_task_ids") or []
        if not ids or len(set(ids)) != batch.get("target") or any(i not in tasks for i in ids):
            raise ValueError("parent_first_recovery_task_inventory_invalid")
        rows = [tasks[i] for i in ids]
        existing_cycle = current.get("extrapolation_recovery_cycle") or {}
        if existing_cycle.get("batch_id") == batch_id:
            if not valid_parent_first_cycle(current):
                raise ValueError("parent_first_recovery_cycle_invalid")
            if all(r.get("status") in {"completed", "failed", "timeout"} for r in rows) and (
                existing_cycle.get("policy_version") == "epoch_refresh_v1"
                or (
                    sum(r.get("status") in {"failed", "timeout"} for r in rows) * 5 < len(rows)
                    and not unreviewed_failure_batches(current)
                )
            ):
                batch["status"] = "results_received"
            continue
        checked = validate_parent_first_result(batch["input_dir"], batch["selection_dir"])
        if checked["result_sha256"] != batch.get("sampling_result_sha256"):
            raise ValueError("parent_first_recovery_selection_changed")
        for task in rows:
            if hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest() != task.get(
                "input_structure_sha256"
            ):
                raise ValueError("parent_first_recovery_task_frame_changed")
        previous = current.get("extrapolation_recovery_cycle") or {}
        lineage_previous = previous
        if (
            previous.get("status") == "awaiting_new_model_refresh"
            and previous.get("replacement_model_version") == batch.get("model_version")
            and previous.get("invalid_model_version") != batch.get("model_version")
        ):
            current.setdefault("extrapolation_recovery_history", []).append(
                {**previous, "status": "replacement_refresh_failed"}
            )
            current.pop("extrapolation_recovery_cycle", None)
            previous = {}
        if previous:
            if previous.get("scope_id") != batch.get("scope_id") or previous.get(
                "invalid_model_version"
            ) != batch.get("model_version"):
                raise ValueError("parent_first_recovery_prior_cycle_mismatch")
            if not recovery_dft_failure_settled(current, previous):
                raise ValueError("parent_first_recovery_prior_dft_not_settled")
            baseline = previous.get("baseline_diagram") or {}
            current.setdefault("extrapolation_recovery_history", []).append(
                {**previous, "status": "superseded_by_parent_first_perturbation"}
            )
        else:
            refresh = current.get("model_refresh") or {}
            baseline = refresh.get("old_diagram") or {}
            if refresh.get("new_model_version") != batch.get("model_version"):
                raise ValueError("parent_first_recovery_refresh_mismatch")
        if (
            baseline.get("status") != "completed"
            or not baseline.get("version")
            or not baseline.get("model_version")
        ):
            raise ValueError("parent_first_recovery_baseline_invalid")
        refresh = current.get("model_refresh") or {}
        diagram = (current.get("phase_diagrams") or {}).get("mlip") or {}
        if diagram.get("model_version") not in {
            baseline["model_version"],
            batch["model_version"],
        } or (
            diagram.get("model_version") == baseline["model_version"]
            and diagram.get("version") != baseline["version"]
        ):
            raise ValueError("parent_first_recovery_current_hull_invalid")
        invalid_hull = previous.get("invalid_hull_version") or (
            diagram.get("version")
            if diagram.get("model_version") == batch["model_version"]
            else None
        )
        current["extrapolation_recovery_cycle"] = {
            "scope_id": batch["scope_id"],
            "batch_id": batch_id,
            "recovery_route": "parent_first_perturbation",
            "recovery_approval_record_id": batch["approval_record_id"],
            "status": "awaiting_reviewed_dft_labels",
            "invalid_model_version": batch["model_version"],
            "baseline_model_version": baseline["model_version"],
            "baseline_hull_version": baseline["version"],
            "baseline_diagram": deepcopy(baseline),
            "invalid_hull_version": invalid_hull,
            "invalidated_task_ids": previous.get(
                "invalidated_task_ids", refresh.get("task_ids") or []
            ),
            "invalidated_model_versions": [batch["model_version"]],
            "direct_source_digest": checked["result_sha256"],
            "selected": [
                {"candidate_id": r["structure_id"], "structure_sha256": r["input_structure_sha256"]}
                for r in rows
            ],
        }
        if (
            batch.get("scope_id") == "epoch_refresh"
            and batch.get("policy_version") == "epoch_refresh_v1"
            and (not previous or previous.get("policy_version") == "epoch_refresh_v1")
        ):
            from phase_agent.evidence.extrapolation_policy import recovery_lineage
            from phase_agent.evidence.model_epoch import model_epoch

            current["extrapolation_recovery_cycle"]["policy_version"] = "epoch_refresh_v1"
            current["extrapolation_recovery_cycle"]["round_evidence_digest"] = batch.get(
                "round_evidence_digest"
            )
            current["extrapolation_recovery_cycle"].update(
                recovery_lineage(
                    lineage_previous,
                    batch.get("model_version"),
                    batch.get("round_evidence_digest"),
                    model_epoch(current, batch.get("model_version")),
                )
            )
        current.setdefault("phase_diagrams", {})["mlip"] = deepcopy(baseline)
        current.setdefault("model_refresh", {})["status"] = "abandoned_extrapolation"
        if not valid_parent_first_cycle(current):
            raise ValueError("parent_first_recovery_cycle_invalid")
        if all(r.get("status") in {"completed", "failed", "timeout"} for r in rows) and (
            current["extrapolation_recovery_cycle"].get("policy_version") == "epoch_refresh_v1"
            or (
                sum(r.get("status") in {"failed", "timeout"} for r in rows) * 5 < len(rows)
                and not unreviewed_failure_batches(current)
            )
        ):
            batch["status"] = "results_received"
    return current

"""Refresh phase labels before a phase-diagram update, with restart-safe caching."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from phase_agent.analysis.phase.identify_result_phase import STAGES, identify_result_phase
from phase_agent.analysis.phase.phase_identification_cache import (
    load_cache as _merge_cache_file,
    compact_cache as _persist_cache_file,
    append_entry,
)


def ensure_phase_identification(
    state,
    manager,
    *,
    phase_references=None,
    cache_path=None,
    _owned=False,
):
    """Identify completed result structures and synchronize phase records.

    Successful identifications are written to a small sidecar cache immediately,
    so an interrupted run can resume without classifying those structures again.
    Unknown results are deliberately not cached: they are retried at the next
    phase-diagram refresh, as a missing/failed identification is not final.
    """
    current = state if _owned and state is not None else deepcopy(state or {})
    from phase_agent.analysis.convergence.mlip_relaxation import (
        reconcile_relaxation_quality,
        DISCARDED,
        EXCLUDED,
    )

    reconcile_relaxation_quality(current, _owned=True)
    reusable_statuses = {"identified", "outside_boundary"}
    cache = {
        key: deepcopy(value)
        for key, value in (current.get("phase_identification_cache") or {}).items()
        if isinstance(value, dict) and value.get("status") in reusable_statuses
    }
    cache_file = Path(cache_path) if cache_path else None
    cache_error = _merge_cache_file(cache, cache_file)
    persisted_keys = set(cache)
    task_by_id = {}
    task_by_identity = {}
    checked = identified = unknown = outside = newly_identified = 0

    tasks = current.setdefault("tasks", [])
    for index, task in enumerate(tasks):
        if task.get("task_id") in (current.get("scientific_feedback_errors") or {}):
            continue
        if task.get("result_disposition") == DISCARDED:
            continue
        if task.get("stage") not in STAGES or task.get("status") != "completed":
            continue
        checked += 1
        updated = identify_result_phase(
            task, manager, phase_references=phase_references, cache=cache
        )
        tasks[index] = updated
        if updated.get("task_id"):
            task_by_id[updated["task_id"]] = updated
        output = updated.get("outputs") or {}
        if (
            updated.get("structure_id")
            and updated.get("model_version")
            and output.get("energy") is not None
        ):
            identity = (
                updated["structure_id"],
                updated["model_version"],
                round(float(output["energy"]), 8),
            )
            task_by_identity.setdefault(identity, []).append(updated)
        evidence = output.get("phase_identification") or {}
        if evidence.get("status") in reusable_statuses:
            identified += evidence.get("status") == "identified"
            outside += evidence.get("status") == "outside_boundary"
            key = _cache_key(evidence)
            if key and key not in cache:
                cache[key] = deepcopy(evidence)
            if key and key not in persisted_keys:
                append_entry(cache_file, key, evidence)
                persisted_keys.add(key)
                newly_identified += 1
        else:
            unknown += 1

    records = current.setdefault("phase_records", [])
    for index, record in enumerate(records):
        if record.get("status") in {EXCLUDED, "excluded_invalid_result"} or record.get(
            "source_task_id"
        ) in (current.get("scientific_feedback_errors") or {}):
            continue
        if record.get("energy_method") not in {"mlip", "dft"} or record.get("energy") is None:
            continue
        task = task_by_id.get(record.get("source_task_id"))
        if task is None and record.get("structure_id"):
            identity = (
                record["structure_id"],
                record.get("model_version") or record.get("source_version"),
                round(float(record["energy"]), 8),
            )
            matches = task_by_identity.get(identity) or []
            task = matches[0] if len(matches) == 1 else None
        if task and task.get("structure_id") == record.get("structure_id"):
            outputs = task.get("outputs") or {}
            evidence = outputs.get("phase_identification") or {}
            record_path = outputs.get("structure_path") or outputs.get("final_structure_path")
        else:
            task = None
            checked += 1
            structure = manager.data.get("structures", {}).get(record.get("structure_id")) or {}
            branch_id = record.get("branch_id") or structure.get("branch_id")
            stage = record.get("stage")
            if stage not in STAGES:
                stage = (
                    "relax_and_feature"
                    if record.get("energy_method") == "mlip"
                    else "dft_single_point"
                )
            probe = {
                "task_id": record.get("source_task_id"),
                "stage": stage,
                "status": "completed",
                "structure_id": record.get("structure_id"),
                "branch_id": branch_id,
                "outputs": {
                    "structure_path": record.get("structure_path"),
                    "structure": record.get("structure"),
                    "final_frame_valid": record.get("final_frame_valid"),
                    "actual_phase": record.get("phase")
                    if record.get("phase_identification_status") == "identified"
                    else None,
                    "phase_identification": record.get("phase_identification") or {},
                },
            }
            probe = identify_result_phase(
                probe, manager, phase_references=phase_references, cache=cache
            )
            outputs = probe.get("outputs") or {}
            evidence = outputs.get("phase_identification") or {}
            record_path = outputs.get("structure_path") or outputs.get("final_structure_path")
            if evidence.get("status") in reusable_statuses:
                identified += evidence.get("status") == "identified"
                outside += evidence.get("status") == "outside_boundary"
                key = _cache_key(evidence)
                if key and key not in cache:
                    cache[key] = deepcopy(evidence)
                if key and key not in persisted_keys:
                    append_entry(cache_file, key, evidence)
                    persisted_keys.add(key)
                    newly_identified += 1
            else:
                unknown += 1

        record = deepcopy(record)
        structure = manager.data.get("structures", {}).get(record.get("structure_id")) or {}
        branch_id = record.get("branch_id") or structure.get("branch_id")
        branch = manager.data.get("branches", {}).get(branch_id) or {}
        record["source_phase"] = record.get("source_phase") or branch.get("P")
        if task and not record.get("source_task_id"):
            record["source_task_id"] = task.get("task_id")
        if record_path:
            record["structure_path"] = record_path
        actual_phase = evidence.get("phase") if evidence.get("status") == "identified" else None
        observed_phase = evidence.get("phase") or evidence.get("observed_phase")
        record["phase"] = actual_phase
        record["actual_phase"] = actual_phase
        record["observed_phase"] = observed_phase
        record["phase_identification_status"] = evidence.get("status", "unknown")
        record["phase_identification"] = deepcopy(evidence)
        record["structure_sha256"] = evidence.get("structure_sha256")
        record["status"] = (
            "completed"
            if actual_phase
            else "outside_search_boundary"
            if evidence.get("status") == "outside_boundary"
            else "pending_phase_identification"
        )
        records[index] = record

        if task and task.get("task_id"):
            task_outputs = task.setdefault("outputs", {})
            task_outputs.update(
                {
                    "source_phase": record["source_phase"],
                    "actual_phase": actual_phase,
                    "observed_phase": observed_phase,
                    "phase_identification": deepcopy(evidence),
                }
            )
            task["source_phase"] = record["source_phase"]
            task["actual_phase"] = actual_phase
            task["observed_phase"] = observed_phase
            task["phase_identification_status"] = record["phase_identification_status"]

    if newly_identified:
        _persist_cache_file(cache_file, cache)
    current["phase_identification_cache"] = cache
    report = {
        "checked_results": checked,
        "identified_results": identified,
        "unknown_results": unknown,
        "outside_boundary_results": outside,
        "newly_cached_results": newly_identified,
        "cache_path": str(cache_file) if cache_file else None,
    }
    if cache_error:
        report["cache_warning"] = cache_error
    current["phase_identification_summary"] = report
    return current, report


def _cache_key(evidence):
    digest = evidence.get("structure_sha256")
    scope = evidence.get("classification_scope")
    return f"{scope}:{digest}" if scope and digest else None

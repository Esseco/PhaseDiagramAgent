"""Refresh phase labels before a phase-diagram update, with restart-safe caching."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from phase_agent.analysis.phase.identify_result_phase import STAGES, identify_result_phase


def ensure_phase_identification(
    state,
    manager,
    *,
    phase_references=None,
    cache_path=None,
):
    """Identify completed result structures and synchronize phase records.

    Successful identifications are written to a small sidecar cache immediately,
    so an interrupted run can resume without classifying those structures again.
    Unknown results are deliberately not cached: they are retried at the next
    phase-diagram refresh, as a missing/failed identification is not final.
    """
    current = deepcopy(state or {})
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
                _persist_cache_file(cache_file, cache)
                persisted_keys.add(key)
                newly_identified += 1
        else:
            unknown += 1

    records = current.setdefault("phase_records", [])
    for index, record in enumerate(records):
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
                    _persist_cache_file(cache_file, cache)
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


def _merge_cache_file(cache, path):
    if path is None or not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            cache.update(
                {
                    key: value
                    for key, value in stored.items()
                    if isinstance(value, dict)
                    and value.get("status") in {"identified", "outside_boundary"}
                }
            )
        return None
    except (OSError, json.JSONDecodeError) as error:
        # The cache is derivable from final structures. Preserve task/state data
        # and rebuild the cache rather than blocking result recovery.
        return f"{type(error).__name__}: {error}"


def _persist_cache_file(path, cache):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    _merge_cache_file(merged, path)
    merged.update(
        {
            key: value
            for key, value in cache.items()
            if isinstance(value, dict) and value.get("status") in {"identified", "outside_boundary"}
        }
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    cache.clear()
    cache.update(merged)

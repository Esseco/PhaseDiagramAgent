"""Read-only view of recovered MC segments and the once-per-round source."""

from copy import deepcopy
import hashlib
import json
import math


def reconciled_mc_state(state):
    tier_state = deepcopy(state.get("tiered_mc_state") or {})
    tasks = {row.get("task_key"): row for row in state.get("tasks") or []
             if row.get("stage") == "deep_search"}
    tier_state["segments"] = [
        {**segment, **deepcopy(tasks.get(segment.get("task_key")) or {})}
        for segment in tier_state.get("segments") or []
    ]
    return tier_state


def second_round_completed(state, model_version):
    """Require a recorded allocation and every allocated task completed."""
    allocations = list(state.get("mc_second_round_allocations") or [])
    if state.get("mc_second_round_allocation"):
        allocations.append(state["mc_second_round_allocation"])
    allocations = [row for row in allocations if row.get("model_version") == model_version]
    tasks = [row for row in state.get("tasks") or []
             if row.get("stage") == "deep_search" and row.get("model_version") == model_version]
    if not allocations or not tasks or any(row.get("status") != "completed" for row in tasks):
        return False
    ids = set(allocations[-1].get("task_ids") or [])
    keys = set(allocations[-1].get("task_keys") or [])
    second = [row for row in tasks if row.get("segment_index") == 1]
    return bool(second) and (not ids or ids.issubset({row.get("task_id") for row in second})) \
        and (not keys or keys.issubset({row.get("task_key") for row in second}))


def first_round_source(state, model_version):
    """Only a fully recovered first wave can seed a second allocation."""
    segments = [row for row in reconciled_mc_state(state)["segments"]
                if row.get("model_version") == model_version]
    indexed = _with_legacy_segment_indices(segments)
    first = [row for row in indexed if row["_effective_segment_index"] == 0]
    if not first or len(first) != len(segments) or any(row.get("status") != "completed" for row in first):
        return None
    evidence = [{"branch_id": row.get("branch_id"), "task_key": row.get("task_key"),
                 "energy_improvement": row.get("energy_improvement"),
                 "stop_reason": row.get("stop_reason"),
                 "result_path": row.get("result_path") or row.get("structure_path")}
                for row in sorted(first, key=lambda item: item.get("task_key") or "")]
    checksum = hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {"checksum": checksum, "branch_ids": [row["branch_id"] for row in evidence],
            "completed_count": len(first)}


def _with_legacy_segment_indices(segments):
    """Infer old missing indices by each branch's saved segment order."""
    next_index = {}
    indexed = []
    for row in segments:
        branch_id = row.get("branch_id")
        if not branch_id:
            return []
        raw_index = row.get("segment_index")
        if isinstance(raw_index, int) and not isinstance(raw_index, bool) and raw_index >= 0:
            effective = raw_index
            next_index[branch_id] = max(next_index.get(branch_id, 0), effective + 1)
        else:
            effective = next_index.get(branch_id, 0)
            next_index[branch_id] = effective + 1
        indexed.append({**row, "_effective_segment_index": effective})
    return indexed


def second_round_already_allocated(state, model_version, source_checksum):
    records = list(state.get("mc_second_round_allocations") or [])
    if state.get("mc_second_round_allocation"):
        records.append(state["mc_second_round_allocation"])
    return any(row.get("model_version") == model_version
               and row.get("source_checksum") == source_checksum for row in records)


def second_round_candidates(candidates, state, diagram, model_version):
    """Use each MC final structure's identified phase and current Ehull/atom."""
    source = first_round_source(state, model_version)
    if not source:
        return [], []
    completed = {row.get("branch_id"): row for row in reconciled_mc_state(state)["segments"]
                 if row.get("model_version") == model_version}
    entries_by_path = {}
    for entry in diagram.get("entries") or []:
        if (entry.get("structure_path") and entry.get("phase_identification_status") == "identified"
                and entry.get("ehull_unit") == "eV/atom" and entry.get("ehull") is not None):
            entries_by_path.setdefault(entry["structure_path"], []).append(entry)
    selected, missing = [], []
    for candidate in candidates:
        branch_id = candidate["branch_id"]
        if branch_id not in completed:
            continue
        previous = completed[branch_id]
        outputs = previous.get("outputs") or {}
        path = (previous.get("result_path") or outputs.get("structure_path")
                or outputs.get("final_structure_path"))
        matches = entries_by_path.get(path) or []
        if len(matches) != 1:
            missing.append(branch_id)
            continue
        entry = matches[0]
        gap = float(entry["ehull"])
        if not math.isfinite(gap) or gap < -1e-8:
            missing.append(branch_id)
            continue
        improvement = max(0.0, float(previous.get("energy_improvement") or 0.0))
        steps = float(previous.get("actual_mc_steps") or previous.get("max_mc_steps") or 1)
        selected.append({**candidate, "P": entry.get("phase") or candidate.get("P"),
                         "relaxed_ehull": gap, "relaxed_ehull_unit": "eV/atom",
                         "ehull_source": "phase_diagram",
                         "first_round_energy_improvement_ev_per_atom": improvement,
                         "first_round_actual_mc_steps": previous.get("actual_mc_steps"),
                         "first_round_stop_reason": previous.get("stop_reason") or "unknown",
                         "second_round_priority": improvement / max(steps, 1.0)})
    return selected, missing

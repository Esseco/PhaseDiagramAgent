"""Record completed DFT feedback once; all hull evidence belongs to one MLIP."""

from copy import deepcopy
from math import isfinite

from phase_agent.analysis.state.dft_round_status import dft_recovery_rounds


def compare_hulls(before, after):
    if (
        not before
        or before.get("model_version") != after.get("model_version")
        or before.get("energy_basis_id") != after.get("energy_basis_id")
    ):
        return None, None

    def keyed(snapshot):
        return {
            r.get("structure_sha256") or r.get("structure_id"): r
            for r in snapshot.get("entries") or []
        }

    old, new = keyed(before), keyed(after)
    common = old.keys() & new.keys()
    values = [
        abs(float(old[k]["ehull"]) - float(new[k]["ehull"]))
        for k in common
        if old[k].get("ehull") is not None and new[k].get("ehull") is not None
    ]
    change = max(values) if values and all(isfinite(x) for x in values) else None
    ground = lambda rows: {k for k, r in rows.items() if r.get("is_stable")}
    return change, ground(old) == ground(new) if old and new else None


def record_feedback_cycle(state):
    version = state.get("active_model_version")
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("model_version") != version or snapshot.get("status") != "completed":
        return
    if any(
        t.get("status") in {"pending", "running", "submitted", "unknown", "submission_unknown"}
        for t in state.get("tasks") or []
    ):
        return
    cycles = state.setdefault("feedback_cycles", [])
    refresh = state.get("model_refresh") or {}
    if (
        refresh.get("status") == "completed"
        and refresh.get("new_model_version") == version
        and not any(r.get("model_version") == version for r in cycles)
        and not any(
            r.get("model_version") == version
            and r.get("stage") in {"dft_single_point", "dft_relax"}
            for r in state.get("tasks") or []
        )
    ):
        state.setdefault("mlip_feedback_baselines", {})[version] = deepcopy(snapshot)
    seen = {r["cycle_id"] for r in cycles}
    used_search = {
        key
        for cycle in cycles
        if cycle.get("model_version") == version
        for key in cycle.get("search_task_ids", [])
    }
    new_search = [
        task["task_id"]
        for task in state.get("tasks") or []
        if task.get("stage") == "deep_search"
        and task.get("status") == "completed"
        and task.get("model_version") == version
        and task.get("task_id") not in used_search
    ]
    if not new_search:
        return
    for batch in dft_recovery_rounds(state):
        if (
            batch["scope"].get("model_version") != version
            or batch["pending_task_ids"]
            or batch["waived_task_ids"]
            or batch["resolved_tasks"] != batch["expected_tasks"]
            or batch["scope_key"] in seen
        ):
            continue
        returned_ids = set(batch["recovered_task_ids"])
        linked_search = {
            task.get("source_search_task_id")
            for task in state.get("tasks") or []
            if task.get("task_id") in returned_ids and task.get("status") == "completed"
        }
        if not linked_search.intersection(new_search):
            continue  # Legacy/unrelated DFT cannot prove a new search feedback cycle.
        baseline = (state.get("mlip_feedback_baselines") or {}).get(version)
        change, unchanged = compare_hulls(baseline, snapshot)
        cycles.append(
            {
                "cycle_id": batch["scope_key"],
                "model_version": version,
                "method": "mlip",
                "complete": True,
                "hull_change": change,
                "ground_state_unchanged": unchanged,
                "hull_version": snapshot.get("version"),
                "dft_task_ids": batch["recovered_task_ids"],
                "search_task_ids": new_search,
            }
        )
        state.setdefault("mlip_feedback_baselines", {})[version] = deepcopy(snapshot)
        seen.add(batch["scope_key"])
        break  # One actual search/DFT feedback cycle, even if several batches returned together.

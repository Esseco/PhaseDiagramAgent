"""Validate offline dedup results before formal compute planning."""

from copy import deepcopy


def accept_dedup_results(state, results):
    current = deepcopy(state)
    by_id = {row.get("task_id"): row for row in current.get("tasks") or []}
    accepted, duplicates, rejected = [], [], []
    for result in results:
        task = by_id.get(result.get("task_id"))
        if not task or task.get("stage") != "offline_check_dedup":
            rejected.append({"task_id": result.get("task_id"), "reason": "unknown_dedup_task"})
            continue
        if any(
            result.get(key) != task.get(key)
            for key in ("task_key", "config_version", "model_version")
        ):
            rejected.append(
                {"task_id": result.get("task_id"), "reason": "version_or_task_mismatch"}
            )
            continue
        duplicate_of = result.get("duplicate_of")
        if duplicate_of:
            target = next(
                (
                    row
                    for row in current.get("tasks") or []
                    if row.get("structure_id") == duplicate_of
                ),
                None,
            )
            if target and target.get("periodic_search_space_id") != task.get(
                "periodic_search_space_id"
            ):
                rejected.append(
                    {
                        "task_id": task["task_id"],
                        "reason": "cross_periodic_search_space_merge_forbidden",
                    }
                )
                continue
            duplicates.append({"structure_id": task["structure_id"], "duplicate_of": duplicate_of})
        elif result.get("legal") is True:
            accepted.append(task["structure_id"])
        else:
            rejected.append({"task_id": task["task_id"], "reason": "illegal_or_unknown"})
        task.update(status="completed", outputs=deepcopy(result))
    expected = set((current.get("dedup_gate") or {}).get("task_ids") or [])
    finished = {
        row.get("task_id") for row in current.get("tasks") or [] if row.get("status") == "completed"
    }
    current["pending_tasks"] = [
        row for row in current.get("tasks") or [] if row.get("status") in {"pending", "running"}
    ]
    current["dedup_gate"] = {
        **(current.get("dedup_gate") or {}),
        "status": "ready" if expected and expected <= finished else "partial",
        "valid_structure_ids": accepted,
        "duplicates": duplicates,
        "rejected": rejected,
    }
    return {
        "status": current["dedup_gate"]["status"],
        "state": current,
        "valid_structure_ids": accepted,
        "duplicates": duplicates,
        "rejected": rejected,
    }

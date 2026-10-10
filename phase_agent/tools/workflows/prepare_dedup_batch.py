"""Prepare an independently approved offline legality/deduplication batch."""

from copy import deepcopy
import hashlib
import json


def prepare_dedup_batch(
    candidates,
    state,
    *,
    config_version,
    model_version,
    approved,
    budget_limit,
    resource_limit,
    budget_limits=None,
):
    if not approved:
        return {"status": "awaiting_approval", "state": deepcopy(state), "tasks": []}
    if budget_limit is None or float(budget_limit) <= 0 or not resource_limit:
        return {
            "status": "not_configured",
            "state": deepcopy(state),
            "tasks": [],
            "reason": "dedup_budget_and_resource_limit_required",
        }
    current = deepcopy(state)
    tasks = []
    for candidate in candidates:
        structure_id = candidate.get("structure_id") or candidate.get("candidate_id")
        search_space_id = candidate.get("periodic_search_space_id")
        if not structure_id or not search_space_id:
            raise ValueError("dedup candidate requires structure_id and periodic_search_space_id")
        key = f"dedup:{config_version}:{structure_id}:{search_space_id}"
        existing = next(
            (row for row in current.get("tasks") or [] if row.get("task_key") == key), None
        )
        if existing:
            continue
        task = {
            "task_id": "DEDUP-" + hashlib.sha256(key.encode()).hexdigest()[:12],
            "task_key": key,
            "structure_id": structure_id,
            "object_id": structure_id,
            "stage": "offline_check_dedup",
            "status": "pending",
            "config_version": config_version,
            "model_version": model_version,
            "periodic_search_space_id": search_space_id,
            "parameters": {
                "checks": ["legality", "periodic_duplicate"],
                "resource_limit": deepcopy(resource_limit),
            },
            "planned_relative_cost": float(budget_limit) / max(1, len(candidates)),
            "candidate_version": hashlib.sha256(
                json.dumps(candidate, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        cost = task["planned_relative_cost"]
        total = (budget_limits or {}).get("total_relative_cost")
        used = float((current.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
        reserved = float(current.get("reserved_relative_cost", 0) or 0)
        if total is not None and used + reserved + cost > float(total):
            return {"status": "budget_exhausted", "state": current, "tasks": tasks}
        current.setdefault("budget_reservations", {})[key] = {
            "status": "reserved",
            "reserved_cost": cost,
            "relative_cost": cost,
            "stage": "offline_check_dedup",
            "config_version": config_version,
            "model_version": model_version,
        }
        current["reserved_relative_cost"] = reserved + cost
        current.setdefault("tasks", []).append(task)
        current.setdefault("pending_tasks", []).append(task)
        tasks.append(task)
    current["dedup_gate"] = {
        "status": "pending" if tasks else current.get("dedup_gate", {}).get("status", "ready"),
        "config_version": config_version,
        "task_ids": [row["task_id"] for row in tasks],
    }
    return {"status": "prepared" if tasks else "already_prepared", "state": current, "tasks": tasks}

"""Ensure one active decision for each BOHB branch/fidelity task."""

from copy import deepcopy


def register_effective_decisions(round_state: dict, actions: list[dict]) -> dict:
    updated = deepcopy(round_state)
    accepted, duplicates = [], []
    registry = updated.setdefault("effective_decisions", {})
    for action in actions:
        identity = f"{action.get('scope_id')}:{action.get('branch_id')}:{action.get('budget')}"
        existing = registry.get(identity)
        if (
            existing
            and existing.get("task_key") != action.get("task_key")
            and existing.get("status") in {"pending", "running", "completed"}
        ):
            duplicates.append(
                {
                    "identity": identity,
                    "rejected_task_key": action.get("task_key"),
                    "existing_task_key": existing.get("task_key"),
                }
            )
            continue
        registry[identity] = {
            "task_key": action.get("task_key"),
            "status": action.get("status"),
            "source": "bohb",
            "selection_source": action.get("selection_source"),
            "strategy_version": updated.get("strategy_version"),
        }
        accepted.append(action)
    return {"round_state": updated, "accepted": accepted, "duplicates": duplicates}

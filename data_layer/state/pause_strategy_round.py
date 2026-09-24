"""Explicitly pause/end a round before changing model or policy."""

from copy import deepcopy


def pause_strategy_round(round_state: dict, *, reason: str, end=False) -> dict:
    updated = deepcopy(round_state)
    updated["status"] = "ended" if end else "paused"
    updated["pause_reason"] = reason
    updated["pending_task_count"] = len((updated.get("bohb_state") or {}).get("pending_tasks", []))
    return updated

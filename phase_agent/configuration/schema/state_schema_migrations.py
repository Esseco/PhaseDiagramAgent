"""Forward-only migrations for persisted Agent-visible state snapshots."""

from copy import deepcopy


CURRENT_STATE_SCHEMA_VERSION = 1


def migrate_state_snapshot(snapshot: dict) -> dict:
    current = deepcopy(snapshot or {})
    version = int(current.get("schema_version", 0))
    if version > CURRENT_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported future state schema version: {version}")
    if version == 0:
        current = {
            **current,
            "schema_version": 1,
            "current_status": current.get("current_status", current.get("status", "ready")),
            "available_budget": current.get("available_budget", current.get("remaining_budget")),
            "uncertainty": deepcopy(
                current.get("uncertainty") or current.get("qbc_uncertainty") or []
            ),
            "search_history": deepcopy(
                current.get("search_history") or current.get("recent_action_history") or []
            ),
        }
    return current


def migrate_persisted_state(state: dict) -> dict:
    current = deepcopy(state or {})
    if current.get("current_state_snapshot"):
        current["current_state_snapshot"] = migrate_state_snapshot(
            current["current_state_snapshot"]
        )
    current["state_snapshots"] = [
        deepcopy(item)
        if item.get("history_format") == "summary-v1"
        else migrate_state_snapshot(item)
        for item in current.get("state_snapshots") or []
    ]
    return current

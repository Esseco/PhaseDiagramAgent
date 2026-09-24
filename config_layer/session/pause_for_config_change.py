"""Pause affected tasks and return to configuration dialogue for hard changes."""

from copy import deepcopy

from config_layer.session.create_config_draft import create_config_draft


def pause_for_config_change(state: dict, confirmed_session: dict, *, reason: str) -> dict:
    updated = deepcopy(state)
    for task in updated.get("tasks", []):
        if task.get("status") in {"pending", "running"}:
            task["status"] = "paused"; task["pause_reason"] = reason
    updated["status"] = "configuration_change_required"
    draft = create_config_draft((confirmed_session.get("confirmed_snapshot") or {}).get("config", confirmed_session.get("config", {})))
    draft["parent_config_version"] = (confirmed_session.get("confirmed_snapshot") or {}).get("config_version")
    return {"state": updated, "config_session": draft}

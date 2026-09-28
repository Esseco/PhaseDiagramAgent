"""Open a new editable draft while preserving the last confirmed snapshot."""

from copy import deepcopy


def begin_config_revision(session: dict, *, reason: str) -> dict:
    if session.get("status") != "confirmed" or not session.get("confirmed_snapshot"):
        raise ValueError("只有已确认配置才能开启新修订")
    updated = deepcopy(session)
    snapshot = updated["confirmed_snapshot"]
    updated["status"] = "draft"
    updated["setup_stage"] = "json_ready"
    updated["config"] = deepcopy(snapshot["config"])
    updated["draft_revision"] = int(updated.get("draft_revision", 0)) + 1
    for key in (
        "agent_reviewed_revision", "agent_reviewed_config_hash",
        "agent_reviewed_config_digest", "agent_reviewed_mother_digest",
        "last_imported_config_hash", "last_imported_config_path",
        "last_imported_config_digest", "last_imported_mother_digest",
    ):
        updated.pop(key, None)
    updated.setdefault("dialogue", []).append({
        "type": "config_revision_started",
        "from_config_version": snapshot["config_version"],
        "revision": updated["draft_revision"],
        "reason": reason,
    })
    return updated

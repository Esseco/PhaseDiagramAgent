"""固定模型、凸包和候选范围，隔离不可比较记录。"""

import hashlib
import json


def validate_bohb_scope(config: dict, candidates: list[dict]) -> dict:
    scope = dict(config.get("scope") or {})
    missing = [
        key
        for key in ("mlip_version", "hull_reference_version", "candidate_set_version")
        if not scope.get(key)
    ]
    if missing:
        return {"status": "unknown", "scope_id": None, "missing": missing}
    identifiers = sorted(str(item.get("branch_id")) for item in candidates)
    payload = {
        "scope": scope,
        "candidate_ids": identifiers,
        "objective": config.get("objective"),
        "agent_search_scope": config.get("agent_search_scope"),
    }
    scope_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return {
        "status": "completed",
        "scope_id": scope_id,
        "scope": scope,
        "candidate_count": len(candidates),
        "missing": [],
    }

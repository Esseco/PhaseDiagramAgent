"""Explicit lifecycle for one Agent-selected Branch batch."""
from copy import deepcopy
import hashlib
import json


ACTIVE_BRANCH_BATCH_STATES = {
    "selected", "relax_pending", "relax_completed", "hull_ready", "hb_active",
}
TRANSITIONS = {
    "selected": {"relax_pending", "failed"},
    "relax_pending": {"relax_pending", "relax_completed", "failed"},
    "relax_completed": {"hull_ready", "failed"},
    "hull_ready": {"hb_active", "completed", "failed"},
    "hb_active": {"hb_active", "completed", "failed"},
    "completed": set(), "failed": set(),
}


def create_branch_batch(branch_ids, *, selected_by, reason=None) -> dict:
    identifiers = sorted(set(branch_ids))
    if not identifiers:
        raise ValueError("branch batch cannot be empty")
    token = hashlib.sha256(json.dumps(identifiers).encode()).hexdigest()[:12]
    return {"batch_id": f"branch-batch-{token}", "status": "selected",
            "branch_ids": identifiers, "selected_by": selected_by,
            "reason": reason, "history": [{"status": "selected"}]}


def transition_branch_batch(batch, status, **facts) -> dict:
    current = deepcopy(batch)
    previous = current.get("status")
    if status not in TRANSITIONS.get(previous, set()):
        raise ValueError(f"invalid branch batch transition: {previous} -> {status}")
    current["status"] = status
    current.update(deepcopy(facts))
    current.setdefault("history", []).append({"status": status, **deepcopy(facts)})
    return current

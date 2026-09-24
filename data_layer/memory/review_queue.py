"""Human-reviewed durable-memory proposals with conflict preservation."""

from copy import deepcopy
import hashlib
import json

from data_layer.memory.decision_memory import LONG_TERM_CATEGORIES, update_long_term_memory


def propose_memory_update(state, *, category, items, system_id, evidence_refs, source="agent"):
    if category not in LONG_TERM_CATEGORIES:
        raise ValueError("unsupported long-term memory category")
    if not system_id or not isinstance(items, list) or not isinstance(evidence_refs, list):
        raise ValueError("system_id, items and evidence_refs are required")
    updated = deepcopy(state); active = (((updated.get("decision_memory") or {}).get("long_term") or {}).get(category) or [])
    conflict = bool(active and list(active) != list(items))
    payload = {"category": category, "items": items, "system_id": system_id,
               "evidence_refs": evidence_refs, "source": source}
    proposal_id = "memory-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    proposal = {"proposal_id": proposal_id, **payload, "status": "conflict_review" if conflict else "pending_review",
                "conflicts_with": deepcopy(active) if conflict else []}
    queue = updated.setdefault("memory_review_queue", [])
    if not any(row.get("proposal_id") == proposal_id for row in queue): queue.append(proposal)
    return {"state": updated, "proposal": proposal}


def review_memory_update(state, proposal_id, *, approved, reviewer="user"):
    updated = deepcopy(state); proposal = next((row for row in updated.get("memory_review_queue") or []
                                                if row.get("proposal_id") == proposal_id), None)
    if proposal is None: raise KeyError(proposal_id)
    if proposal.get("status") in {"approved", "rejected"}: return {"state": updated, "status": "already_reviewed"}
    proposal["status"] = "approved" if approved else "rejected"; proposal["reviewer"] = reviewer
    if approved:
        updated = update_long_term_memory(updated, {proposal["category"]: proposal["items"]},
                                          source=f"review:{proposal_id}:{reviewer}")
        proposal = next(row for row in updated["memory_review_queue"] if row["proposal_id"] == proposal_id)
        proposal["status"] = "approved"; proposal["reviewer"] = reviewer
    return {"state": updated, "status": proposal["status"]}

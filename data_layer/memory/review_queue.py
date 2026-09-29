"""Human-reviewed durable-memory proposals with conflict preservation."""

from copy import deepcopy
import hashlib
import json

from data_layer.memory.decision_memory import LONG_TERM_CATEGORIES, update_long_term_memory
from data_layer.memory.knowledge_record import knowledge_record
from data_layer.memory.verify_evidence_refs import verify_evidence_refs


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
        if proposal.get("record"):
            record = knowledge_record(proposal["record"])
            if proposal.get("source") == "published_domain_skill":
                record["status"] = "active"
                record["maturity"] = "heuristic"
            record["approved_by_user"] = True
            records = updated.setdefault("decision_memory", {}).setdefault("records", [])
            if not any(row.get("knowledge_id") == record["knowledge_id"] for row in records):
                records.append(record)
                updated["decision_memory"]["version"] = int(
                    updated["decision_memory"].get("version", 0)) + 1
                updated["decision_memory"].setdefault("history", []).append({
                    "version": updated["decision_memory"]["version"],
                    "knowledge_id": record["knowledge_id"], "source": f"review:{proposal_id}:{reviewer}"})
        else:
            updated = update_long_term_memory(updated, {proposal["category"]: proposal["items"]},
                                              source=f"review:{proposal_id}:{reviewer}")
        proposal = next(row for row in updated["memory_review_queue"] if row["proposal_id"] == proposal_id)
        proposal["status"] = "approved"; proposal["reviewer"] = reviewer
    return {"state": updated, "status": proposal["status"]}


def propose_knowledge_record(state, record, *, source="agent"):
    """Queue structured knowledge; review_memory_update remains the approval gate."""
    normalized = knowledge_record(record)
    if normalized["maturity"] == "validated":
        convergence = state.get("convergence_result") or state.get("convergence") or {}
        if not (convergence.get("status") == "finished" and convergence.get("converged") is True
                and state.get("user_accepted_convergence") is True):
            raise ValueError("validated knowledge requires accepted convergence evidence")
    missing = verify_evidence_refs(state, normalized["evidence_refs"], scope=normalized["scope"])
    if missing:
        raise ValueError(f"knowledge evidence references are missing: {missing}")
    updated = deepcopy(state)
    proposal_id = "memory-" + normalized["knowledge_id"]
    queue = updated.setdefault("memory_review_queue", [])
    existing = next((row for row in queue if row.get("proposal_id") == proposal_id), None)
    if existing:
        return {"state": updated, "proposal": existing}
    active = ((updated.get("decision_memory") or {}).get("records") or [])
    conflicts = [row["knowledge_id"] for row in active
                 if row.get("category") == normalized.get("category")
                 and row.get("scope") == normalized["scope"]
                 and row.get("system_id") == normalized.get("system_id")
                 and row.get("statement") != normalized["statement"]]
    proposal = {"proposal_id": proposal_id, "record": normalized, "source": source,
                "status": "conflict_review" if conflicts else "pending_review",
                "conflicts_with": conflicts}
    queue.append(proposal)
    return {"state": updated, "proposal": proposal}

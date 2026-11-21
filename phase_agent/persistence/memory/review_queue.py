"""Human-reviewed durable-memory proposals with conflict preservation."""

from copy import deepcopy
import hashlib
import json

from phase_agent.persistence.memory.decision_memory import (
    LONG_TERM_CATEGORIES,
    update_long_term_memory,
)
from phase_agent.persistence.memory.knowledge_record import knowledge_record
from phase_agent.persistence.memory.verify_evidence_refs import EvidenceResolver
from phase_agent.persistence.memory.applicability import project_scope_matches


def propose_memory_update(state, *, category, items, system_id, evidence_refs, source="agent"):
    if category not in LONG_TERM_CATEGORIES:
        raise ValueError("unsupported long-term memory category")
    if not system_id or not isinstance(items, list) or not isinstance(evidence_refs, list):
        raise ValueError("system_id, items and evidence_refs are required")
    updated = deepcopy(state)
    active = ((updated.get("decision_memory") or {}).get("long_term") or {}).get(category) or []
    conflict = bool(active and list(active) != list(items))
    payload = {
        "category": category,
        "items": items,
        "system_id": system_id,
        "evidence_refs": evidence_refs,
        "source": source,
    }
    proposal_id = (
        "memory-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    )
    proposal = {
        "proposal_id": proposal_id,
        **payload,
        "status": "conflict_review" if conflict else "pending_review",
        "conflicts_with": deepcopy(active) if conflict else [],
    }
    queue = updated.setdefault("memory_review_queue", [])
    if not any(row.get("proposal_id") == proposal_id for row in queue):
        queue.append(proposal)
    return {"state": updated, "proposal": proposal}


def review_memory_update(state, proposal_id, *, approved, reviewer="user", _owned=False):
    updated = state if _owned else deepcopy(state)
    proposal = next(
        (
            row
            for row in updated.get("memory_review_queue") or []
            if row.get("proposal_id") == proposal_id
        ),
        None,
    )
    if proposal is None:
        raise KeyError(proposal_id)
    if proposal.get("status") in {"approved", "rejected"}:
        return {"state": updated, "status": "already_reviewed"}
    if not approved:
        proposal["status"] = "rejected"
        proposal["reviewer"] = reviewer
    if approved:
        if proposal.get("record"):
            record = knowledge_record(proposal["record"])
            if (
                proposal["record"].get("revision_id")
                and record["revision_id"] != proposal["record"]["revision_id"]
            ):
                raise ValueError("knowledge_proposal_revision_changed; propose a new revision")
            resolver = EvidenceResolver(updated)
            missing = resolver.missing(record["evidence_refs"], scope=record["scope"])
            if missing:
                raise ValueError(f"knowledge evidence references are missing: {missing}")
            current_fingerprint = resolver.fingerprint(record["evidence_refs"])
            if (
                proposal.get("evidence_fingerprint")
                and proposal["evidence_fingerprint"] != current_fingerprint
            ):
                raise ValueError(
                    "knowledge evidence changed before approval; propose a new revision"
                )
            if record["maturity"] == "validated":
                convergence = updated.get("convergence_result") or updated.get("convergence") or {}
                if not (
                    convergence.get("status") == "finished"
                    and convergence.get("converged") is True
                    and updated.get("user_accepted_convergence") is True
                ):
                    raise ValueError("validated knowledge requires accepted convergence evidence")
            if proposal.get("source") == "published_domain_skill":
                record["status"] = "active"
                record["maturity"] = "heuristic"
            record["evidence_fingerprint"] = current_fingerprint
            from phase_agent.persistence.memory.knowledge_record import knowledge_revision

            record["revision_id"] = knowledge_revision(record)
            record["approved_by_user"] = True
            records = updated.setdefault("decision_memory", {}).setdefault("records", [])
            targets = set(record.get("supersedes") or [])
            eligible = {
                row["knowledge_id"]
                for row in records
                if row.get("scope") == record.get("scope")
                and row.get("system_id") == record.get("system_id")
                and row.get("category") == record.get("category")
                and row.get("applicability", {}) == record.get("applicability", {})
                and project_scope_matches(row, record)
            }
            if targets - eligible:
                raise ValueError(
                    "supersedes must reference existing knowledge with the same scope and applicability"
                )
            existing = next(
                (row for row in records if row.get("knowledge_id") == record["knowledge_id"]), None
            )
            if existing and (
                existing.get("revision_id") != record["revision_id"]
                or existing.get("status") != "active"
            ):
                raise ValueError("knowledge_revision_collision; propose a new revision")
            if existing is None:
                for existing_record in records:
                    if existing_record.get("knowledge_id") in targets:
                        existing_record["status"] = "superseded"
                        existing_record["superseded_by"] = record["knowledge_id"]
                records.append(record)
                dependencies = updated["decision_memory"].setdefault("evidence_dependencies", {})
                for ref in record["evidence_refs"]:
                    identities = dependencies.setdefault(ref, [])
                    if record["knowledge_id"] not in identities:
                        identities.append(record["knowledge_id"])
                updated["decision_memory"]["version"] = (
                    int(updated["decision_memory"].get("version", 0)) + 1
                )
                updated["decision_memory"].setdefault("history", []).append(
                    {
                        "version": updated["decision_memory"]["version"],
                        "knowledge_id": record["knowledge_id"],
                        "source": f"review:{proposal_id}:{reviewer}",
                    }
                )
        else:
            updated = update_long_term_memory(
                updated,
                {proposal["category"]: proposal["items"]},
                source=f"review:{proposal_id}:{reviewer}",
                _owned=True,
            )
        proposal = next(
            row for row in updated["memory_review_queue"] if row["proposal_id"] == proposal_id
        )
        proposal["status"] = "approved"
        proposal["reviewer"] = reviewer
    result = {
        "state": updated,
        "status": proposal["status"],
        "memory_version": int((updated.get("decision_memory") or {}).get("version", 0)),
    }
    if proposal.get("record"):
        saved = next(
            (
                row
                for row in (updated.get("decision_memory") or {}).get("records", [])
                if row.get("knowledge_id") == proposal["record"]["knowledge_id"]
            ),
            None,
        )
        result.update(
            {
                "knowledge_id": proposal["record"]["knowledge_id"],
                "revision_id": (saved or {}).get("revision_id"),
                "applied": saved is not None and saved.get("status") == "active",
            }
        )
    return result


def propose_knowledge_record(state, record, *, source="agent", _owned=False):
    """Queue structured knowledge; review_memory_update remains the approval gate."""
    from phase_agent.persistence.memory.applicability import memory_scope

    project_id = memory_scope(state)["project_id"]
    if isinstance(record, dict) and record.get("scope") == "project" and project_id:
        if record.get("project_id") not in {None, project_id}:
            raise ValueError("project knowledge must be proposed in its applicable project")
        record = {**record, "project_id": project_id}
    normalized = knowledge_record(record)
    if normalized["maturity"] == "validated":
        convergence = state.get("convergence_result") or state.get("convergence") or {}
        if not (
            convergence.get("status") == "finished"
            and convergence.get("converged") is True
            and state.get("user_accepted_convergence") is True
        ):
            raise ValueError("validated knowledge requires accepted convergence evidence")
    resolver = EvidenceResolver(state)
    missing = resolver.missing(normalized["evidence_refs"], scope=normalized["scope"])
    if missing:
        raise ValueError(f"knowledge evidence references are missing: {missing}")
    updated = state if _owned else deepcopy(state)
    fingerprint = resolver.fingerprint(normalized["evidence_refs"])
    normalized["evidence_fingerprint"] = fingerprint
    from phase_agent.persistence.memory.knowledge_record import knowledge_revision

    normalized["revision_id"] = knowledge_revision(normalized)
    active = (updated.get("decision_memory") or {}).get("records") or []
    family = normalized["knowledge_family_id"]
    previous = [
        row
        for row in active
        if (row.get("knowledge_family_id") or row.get("knowledge_id")) == family
    ]
    if previous:
        same = next(
            (
                row
                for row in previous
                if row.get("revision_id") == normalized["revision_id"]
                and row.get("status") == "active"
            ),
            None,
        )
        normalized["knowledge_id"] = (
            same["knowledge_id"] if same else family + "-" + normalized["revision_id"]
        )
        if not same:
            normalized["supersedes"] = list(
                dict.fromkeys(
                    normalized["supersedes"]
                    + [row["knowledge_id"] for row in previous if row.get("status") == "active"]
                )
            )
    proposal_id = "memory-" + normalized["knowledge_id"] + "-" + normalized["revision_id"]
    queue = updated.setdefault("memory_review_queue", [])
    existing = next((row for row in queue if row.get("proposal_id") == proposal_id), None)
    if existing:
        return {"state": updated, "proposal": existing}
    active = (updated.get("decision_memory") or {}).get("records") or []
    conflicts = [
        row["knowledge_id"]
        for row in active
        if row.get("category") == normalized.get("category")
        and row.get("status") == "active"
        and row.get("scope") == normalized["scope"]
        and row.get("system_id") == normalized.get("system_id")
        and row.get("applicability", {}) == normalized.get("applicability", {})
        and project_scope_matches(row, normalized)
        and row.get("statement") != normalized["statement"]
    ]
    proposal = {
        "proposal_id": proposal_id,
        "record": normalized,
        "source": source,
        "status": "conflict_review" if conflicts else "pending_review",
        "conflicts_with": conflicts,
    }
    proposal["evidence_fingerprint"] = fingerprint
    queue.append(proposal)
    return {"state": updated, "proposal": proposal}

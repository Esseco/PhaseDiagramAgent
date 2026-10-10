"""Expose old approved string lists as read-only structured knowledge."""

import hashlib


def legacy_knowledge_records(state, *, system_id=None):
    memory = state.get("decision_memory") or {}
    categories = memory.get("long_term") or {}
    if not categories and memory.get("long_term_advice"):
        categories = {"human_system_knowledge": memory["long_term_advice"]}
    records = []
    for category, items in categories.items():
        for statement in items or []:
            digest = hashlib.sha256(f"{category}:{statement}".encode()).hexdigest()[:16]
            records.append(
                {
                    "knowledge_id": f"LEGACY-{digest}",
                    "scope": "project",
                    "category": category,
                    "statement": statement,
                    "status": "active",
                    "maturity": "heuristic",
                    "applicability": {},
                    "evidence_refs": [],
                    "approved_by_user": True,
                    "source_project": "legacy_project_memory",
                    "system_id": system_id,
                }
            )
    return records

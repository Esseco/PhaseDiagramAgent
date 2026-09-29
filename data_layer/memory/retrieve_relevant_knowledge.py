"""Select a small, applicable set of approved knowledge for one decision."""

from copy import deepcopy
from data_layer.memory.legacy_knowledge_records import legacy_knowledge_records


def retrieve_relevant_knowledge(state, *, system_id=None, phase=None, action=None,
                                model_version=None, limit=8):
    if not 1 <= int(limit) <= 10:
        raise ValueError("knowledge limit must be between 1 and 10")
    context = {"system_id": system_id, "phase": phase, "action": action,
               "model_version": model_version}
    ranked = []
    records = list(((state.get("decision_memory") or {}).get("records") or []))
    records.extend(legacy_knowledge_records(state, system_id=system_id))
    for row in records:
        if row.get("status") != "active" or row.get("approved_by_user") is not True:
            continue
        if row.get("scope") in {"project", "system"} and row.get("system_id") != system_id:
            continue
        applicability = row.get("applicability") or {}
        if any(value is not None and context.get(key) != value
               for key, value in applicability.items() if key in context):
            continue
        score = {"project": 40, "user": 30, "system": 20, "global": 10}.get(row.get("scope"), 0)
        score += {"validated": 4, "project_observed": 2, "heuristic": 1}.get(row.get("maturity"), 0)
        score += len([key for key in applicability if key in context and context[key] is not None])
        ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], item[1].get("knowledge_id", "")))
    return deepcopy([row for _, row in ranked[:int(limit)]])

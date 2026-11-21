"""Select a small, applicable set of approved knowledge for one decision."""

from copy import deepcopy
from collections import defaultdict
import json
import re
from phase_agent.persistence.memory.legacy_knowledge_records import legacy_knowledge_records


def retrieve_relevant_knowledge(
    state,
    *,
    system_id=None,
    phase=None,
    action=None,
    model_version=None,
    project_id=None,
    limit=8,
    query="",
    token_budget=2000,
    diagnostics=None,
    required_constraints=None,
):
    if not 1 <= int(limit) <= 10:
        raise ValueError("knowledge limit must be between 1 and 10")
    from phase_agent.persistence.memory.applicability import (
        memory_scope,
        applicability_errors,
        project_scope_matches,
    )

    scope = memory_scope(state)
    context = {
        "system_id": system_id,
        "phase": phase if phase is not None else scope["phase"],
        "action": action,
        "model_version": model_version,
        "project_id": project_id or scope["project_id"],
    }

    def filtered(row, reason):
        if diagnostics is not None and len(diagnostics) < 128:
            diagnostics.append({"knowledge_id": row.get("knowledge_id"), "reason": reason})

    ranked = []
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError("memory token_budget must be a positive integer")
    terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2}", str(query).lower()))
    records = list(((state.get("decision_memory") or {}).get("records") or []))
    records.extend(legacy_knowledge_records(state, system_id=system_id))
    eligible = [
        row
        for row in records
        if isinstance(row, dict)
        and isinstance(row.get("knowledge_id"), str)
        and isinstance(row.get("statement"), str)
        and isinstance(row.get("applicability", {}), dict)
        and isinstance(row.get("supersedes", []), list)
        and isinstance(row.get("scope", ""), str)
        and isinstance(row.get("maturity", ""), str)
        and row.get("status") == "active"
        and row.get("approved_by_user") is True
    ]
    for row in records:
        if isinstance(row, dict) and row.get("status") != "active":
            filtered(row, "status:" + str(row.get("status")))
    by_id = defaultdict(list)
    for row in eligible:
        by_id[row["knowledge_id"]].append(row)
    superseded = {
        old.get("knowledge_id")
        for row in eligible
        for identity in (row.get("supersedes") or [])
        if isinstance(identity, str)
        for old in by_id.get(identity, [])
        if all(row.get(key) == old.get(key) for key in ("scope", "system_id", "category"))
        and row.get("applicability", {}) == old.get("applicability", {})
        and project_scope_matches(row, old)
    }
    from phase_agent.persistence.memory.verify_evidence_refs import (
        EvidenceResolver,
        revalidation_reason,
    )

    resolver = EvidenceResolver(state)
    for row in eligible:
        reason = revalidation_reason(row, resolver)
        if reason:
            filtered(row, "needs_revalidation:" + reason)
            continue
        if row.get("knowledge_id") in superseded:
            filtered(row, "superseded")
            continue
        if row.get("model_dependent") and row.get("model_version") != model_version:
            filtered(row, "model_scope_mismatch")
            continue
        if row.get("scope") in {"project", "system"} and row.get("system_id") != system_id:
            filtered(row, "system_scope_mismatch")
            continue
        applicability = row.get("applicability") or {}
        if applicability_errors(applicability):
            continue
        if row.get("scope") == "project" and not row.get("knowledge_id", "").startswith("LEGACY-"):
            bound_project = (
                row.get("project_id")
                or applicability.get("project_id")
                or row.get("source_project")
            )
            # Old inline records without provenance are local to this state file.
            if bound_project and (
                not context["project_id"] or bound_project != context["project_id"]
            ):
                continue
        if any(
            value is not None
            and (
                value not in context[key]
                if isinstance(context.get(key), (list, tuple, set))
                else context.get(key) != value
            )
            for key, value in applicability.items()
            if key in context
        ):
            continue
        if (
            required_constraints is not None
            and row.get("scope") == "user"
            and row.get("category")
            in {
                "human_system_knowledge",
                "frozen_parameter_advice",
                "constraint",
                "user_constraint",
            }
        ):
            required_constraints.append(
                {
                    key: deepcopy(row.get(key))
                    for key in ("knowledge_id", "statement", "applicability", "revision_id")
                }
            )
        score = {"project": 40, "user": 30, "system": 20, "global": 10}.get(row.get("scope"), 0)
        score += {"validated": 4, "project_observed": 2, "heuristic": 1}.get(row.get("maturity"), 0)
        score += len([key for key in applicability if key in context and context[key] is not None])
        matched = sorted(term for term in terms if term in row.get("statement", "").lower())
        score += 10 * len(matched)
        row = deepcopy(row)
        row["retrieval_reason"] = {
            "scope": row.get("scope"),
            "matched_terms": matched,
            "applicability": deepcopy(applicability),
        }
        ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], item[1].get("knowledge_id", "")))
    selected, used, seen = [], 0, set()
    for _, row in ranked:
        if row.get("knowledge_id") in seen:
            continue
        # Conservative character budget, not a provider-specific token count.
        cost = len(json.dumps(row, ensure_ascii=False))
        if used + cost > token_budget:
            filtered(row, "retrieval_budget")
            continue
        selected.append(row)
        used += cost
        seen.add(row.get("knowledge_id"))
        if len(selected) == int(limit):
            break
    return selected

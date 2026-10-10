"""Validate one reviewed, scoped knowledge statement."""

from copy import deepcopy
import hashlib
import json


SCOPES = {"project", "system", "user", "global"}
MATURITIES = {"validated", "project_observed", "heuristic", "deprecated"}
STATUSES = {"active", "superseded", "deprecated", "needs_revalidation"}


def knowledge_record(value):
    if not isinstance(value, dict):
        raise ValueError("knowledge record must be an object")
    row = deepcopy(value)
    if row.get("scope") not in SCOPES or row.get("maturity") not in MATURITIES:
        raise ValueError("knowledge scope or maturity is invalid")
    if row.get("status", "active") not in STATUSES:
        raise ValueError("knowledge status is invalid")
    if not isinstance(row.get("statement"), str) or not row["statement"].strip():
        raise ValueError("knowledge statement is required")
    if not isinstance(row.get("category"), str) or not row["category"].strip():
        raise ValueError("knowledge category is required")
    if not isinstance(row.get("applicability", {}), dict):
        raise ValueError("applicability must be an object")
    refs = row.get("evidence_refs", [])
    if not isinstance(refs, list) or any(
        not isinstance(ref, str) or not ref.strip() for ref in refs
    ):
        raise ValueError("evidence_refs must contain non-empty IDs")
    if row["maturity"] == "validated" and not refs:
        raise ValueError("validated knowledge requires evidence references")
    for field in ("source_project", "system_id"):
        if row.get("scope") in {"project", "system"} and not row.get(field):
            raise ValueError(f"{field} is required for project/system knowledge")
    row["statement"] = row["statement"].strip()
    row["status"] = row.get("status", "active")
    row["applicability"] = deepcopy(row.get("applicability") or {})
    row["evidence_refs"] = list(dict.fromkeys(refs))
    row["supersedes"] = list(dict.fromkeys(row.get("supersedes") or []))
    identity = {
        key: row.get(key)
        for key in (
            "scope",
            "category",
            "statement",
            "source_project",
            "system_id",
            "applicability",
        )
    }
    row.setdefault(
        "knowledge_id",
        "KM-"
        + hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[
            :16
        ],
    )
    return row

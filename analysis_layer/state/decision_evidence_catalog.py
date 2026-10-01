"""Reference existing decision evidence; never create scientific evidence."""

import hashlib
import json


def decision_evidence_catalog(context):
    catalog = {}
    for method, snapshot in (context.get("current_phase_diagram") or {}).items():
        version = snapshot.get("version")
        if version:
            reference = f"phase_diagram:{method}:{version}"
            catalog[reference] = {"kind": "phase_diagram", "method": method,
                "version": version, "status": snapshot.get("status"),
                "energy_basis_id": snapshot.get("energy_basis_id")}
    for row in context.get("relevant_approved_knowledge") or []:
        if row.get("knowledge_id"):
            reference = f"knowledge:{row['knowledge_id']}"
            catalog[reference] = {"kind": "approved_knowledge", "knowledge_id": row["knowledge_id"],
                                  "source_project": row.get("source_project")}
    for row in (context.get("recent_experience") or {}).get("actions") or []:
        if row.get("record_id"):
            reference = f"action:{row['record_id']}"
            catalog[reference] = {"kind": "action_record", "record_id": row["record_id"],
                                  "status": row.get("status"), "tool": row.get("tool")}
    for row in (context.get("task_round_evidence") or {}).get("groups") or []:
        lineage = row.get("lineage") or {}
        if not any(value is not None for value in lineage.values()):
            continue
        digest = hashlib.sha256(json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        catalog[f"task_group:{digest}"] = {"kind": "saved_task_lineage", "lineage": lineage,
            "task_count": row.get("task_count"), "stage_evidence": row.get("stage_evidence"),
            "missing_lineage_fields": row.get("missing_lineage_fields")}
    for report in (context.get("calculation_cost_reference") or {}).get("reports") or []:
        stage = report.get("stage")
        if stage:
            catalog[f"cost_reference:{stage}"] = {"kind": "reference_cost_estimate",
                "report": report, "instruction": "Reference estimate, not an individual measured sample or task quote."}
    return {"references": catalog,
        "instruction": "Use exact reference keys for proposal evidence_refs. Existence does not prove scientific adequacy; cite source versions and explain their relevance. Missing references must not be invented."}


def check_decision_evidence_refs(references, catalog):
    available = catalog.get("references") or {}
    if not isinstance(references, list) or any(not isinstance(ref, str) for ref in references):
        return {"status": "invalid_reference_format", "missing": []}
    missing = sorted(set(references) - set(available))
    return {"status": "unknown_references" if missing else "references_found" if references else "no_references",
            "missing": missing, "instruction": "Reference lookup is not execution authorization or scientific validation."}

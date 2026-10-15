"""Current-model return receipts for conversation; never collect or alter results."""

from collections import Counter


def batch_return_facts(state):
    model = state.get("active_model_version") or (state.get("mlip_status") or {}).get(
        "active_version"
    )
    rows = state.get("relax_mc_batches") or []
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and model
        and row.get("model_version") == model
        and row.get("status") != "cancelled"
    ]
    verified = []
    for row in rows:
        summary = row.get("result_summary") or {}
        matching_plan = (
            not row.get("plan_checksum") or summary.get("plan_checksum") == row["plan_checksum"]
        )
        if (
            row.get("status") == "completed"
            and row.get("import_checksum")
            and not row.get("return_error")
            and summary.get("status") == "completed"
            and summary.get("model_version") == model
            and matching_plan
        ):
            verified.append(row)
    counts = Counter(
        t.get("stage")
        for t in state.get("tasks") or []
        if t.get("model_version") == model and t.get("status") == "completed"
    )
    diagrams = {
        method: {
            "status": row.get("status"),
            "version": row.get("version"),
            "entry_count": len(row.get("entries") or []),
        }
        for method, row in (state.get("phase_diagrams") or {}).items()
        if isinstance(row, dict) and row.get("model_version") in (None, model)
    }
    return {
        "model_version": model,
        "registered_batches": len(rows),
        "verified_batches": len(verified),
        "all_registered_returns_verified": bool(rows) and len(verified) == len(rows),
        "reported_returned_tasks": sum(
            (row.get("result_summary") or {}).get("task_count", 0) for row in verified
        ),
        "completed_tasks_by_stage": dict(counts),
        "phase_diagrams": diagrams,
        "instruction": "Verified completed return receipts supersede historical transfer errors. A later collection of zero new results does not erase previously registered results. History is not current failure evidence. No authority to create, approve or submit tasks.",
    }

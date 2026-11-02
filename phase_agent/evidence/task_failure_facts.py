"""Read-only failure lineage; recorded attempts remain unchanged in the ledger."""


def task_failure_facts(state):
    tasks = state.get("tasks") or []
    by_id = {row.get("task_id"): row for row in tasks if row.get("task_id")}
    actionable = []
    resolved = ongoing = superseded = 0
    for row in tasks:
        if row.get("status") not in {"failed", "timeout"}:
            continue
        current = row
        seen = {row.get("task_id")}
        disposition = None
        cycle = False
        while current.get("retry_task_id"):
            key = current["retry_task_id"]
            if key in seen:
                cycle = True
                break
            if key not in by_id:
                break
            seen.add(key)
            retry = by_id[key]
            if any(
                row.get(field) is not None and retry.get(field) != row[field]
                for field in ("model_version", "structure_id", "stage")
            ):
                break
            if retry.get("status") == "completed":
                disposition = "resolved"
                break
            if retry.get("status") in {
                "pending",
                "running",
                "submitted",
                "queued",
                "prepared",
                "inputs_prepared",
            }:
                disposition = "ongoing"
                break
            current = retry
        if disposition == "resolved":
            resolved += 1
        elif disposition == "ongoing":
            ongoing += 1
        elif current is not row and not cycle:
            superseded += 1
        else:
            actionable.append(row)
    return {
        "actionable_failures": actionable,
        "resolved_failure_attempts": resolved,
        "failure_attempts_with_active_retry": ongoing,
        "superseded_failure_attempts": superseded,
        "instruction": "Task counts include historical attempts. A failed attempt with a same-scope successful or active retry is not a new retry candidate. Never rewrite the original failure or automatically restart any task.",
    }

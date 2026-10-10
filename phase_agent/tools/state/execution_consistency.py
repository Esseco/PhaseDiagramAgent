"""Read-only checks for contradictory execution audit and reservation facts."""


def execution_state_issues(state):
    latest = {}
    for record in state.get("action_records") or []:
        action = record.get("final_action") or {}
        if action.get("task_key"):
            latest[action["task_key"]] = record
    issues = []
    for key, decision in (state.get("effective_decisions") or {}).items():
        record = latest.get(key) or {}
        reason = None
        if decision.get("status") == "reconciliation_required":
            reason = "handler_failed_effects_unknown"
        elif decision.get("status") == "reserved" and record.get("status") == "failed":
            reason = "failed_action_still_reserved"
        elif decision.get("verified_no_effect"):
            reservation = (state.get("budget_reservations") or {}).get(key) or {}
            if decision.get("status") != "failed" or reservation.get("status") != "settled":
                reason = "verified_failure_not_settled"
        if reason:
            issues.append(
                {
                    "identity": {
                        "tool": decision.get("tool"),
                        "invocation_id": decision.get("record_id") or record.get("record_id"),
                    },
                    "task_key": key,
                    "reason": reason,
                    "required_action": "reconcile_business_artifacts",
                }
            )
    return issues

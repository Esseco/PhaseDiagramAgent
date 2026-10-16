"""Read-only checks for contradictory execution audit and reservation facts."""


def execution_state_issues(state):
    latest = {}
    for record in state.get("action_records") or []:
        action = (
            record.get("final_action")
            or (record.get("agent_proposal") or {}).get("raw_action")
            or {}
        )
        if action.get("task_key"):
            latest[action["task_key"]] = record
    issues = []
    for key, record in latest.items():
        action = (
            record.get("final_action")
            or (record.get("agent_proposal") or {}).get("raw_action")
            or {}
        )
        execution = record.get("execution_result") or record.get("execution") or {}
        if (
            record.get("status") != "failed"
            or execution.get("status") != "failed"
            or not action.get("tool")
            or action.get("tool") in {"check_convergence", "pause_search"}
        ):
            continue
        audit = (state.get("execution_reconciliations") or {}).get(record.get("record_id")) or {}
        identity = audit.get("identity") or {}
        from phase_agent.tools.policy.file_approval import proposal_hash

        if (
            audit.get("resolution") in {"verified_no_effect", "verified_registered_effects"}
            and audit.get("evidence")
            and audit.get("reviewed_at")
            and identity.get("invocation_id") == record.get("record_id")
            and identity.get("tool") == action.get("tool")
            and identity.get("action_hash") == proposal_hash(action)
        ):
            continue
        # Failed child-budget handlers may have no parent reservation at all.
        # Missing reservation/receipt does not prove that no files were written.
        issues.append(
            {
                "identity": {"tool": action["tool"], "invocation_id": record.get("record_id")},
                "task_key": key,
                "reason": "failed_action_requires_recovery",
                "required_action": "reconcile_business_artifacts",
            }
        )
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

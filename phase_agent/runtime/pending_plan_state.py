"""Shared lifecycle classification for persisted scientific approval plans."""

from copy import deepcopy


def stale_pending_plan_ids(state):
    pending = state.get("pending_execution_policies") or {}
    if not pending:
        return []
    from phase_agent.evidence.project_authority import project_identity

    current = project_identity(state)
    return [
        plan_id
        for plan_id, row in pending.items()
        if row.get("project_identity") and row.get("project_identity") != current
    ]


def visible_pending(state):
    pending = state.get("pending_execution_policies") or {}
    stale = set(stale_pending_plan_ids(state))
    return {key: row for key, row in pending.items() if key not in stale}


def archived_plan(plan_id, row, *, reason, tool=None):
    """Preserve the complete frozen proposal when it becomes unapprovable."""
    entry = {"plan_id": plan_id, "reason": reason, "plan": deepcopy(row)}
    if tool is not None:
        entry["tool"] = tool
    return entry


def supersede_stale_pending(state, *, reason="模型或生效设置已改变，旧方案不可继续审批"):
    pending = state.get("pending_execution_policies") or {}
    stale = stale_pending_plan_ids(state)
    if not stale:
        return [], False
    records = {row.get("record_id"): row for row in state.get("action_records") or []}
    removed = []
    for plan_id in stale:
        row = pending.pop(plan_id)
        removed.append(archived_plan(plan_id, row, reason=reason))
        record = records.get(plan_id)
        if record and record.get("status") in {None, "awaiting_approval", "prepared"}:
            record["status"] = "superseded"
            record["superseded_reason"] = reason
    state["pending_execution_policies"] = pending
    state.setdefault("superseded_pending_plans", []).extend(removed)
    return removed, True

"""Bridge existing validated execution policy to durable approval."""

from phase_agent.tools.policy.file_approval import proposal_hash


def native_approval_gate(frame, policy):
    if frame["mode"] != "interactive":
        return None
    context = frame.get("context") or {}
    state_path = context.get("state_path") or (context.get("effective_config") or {}).get(
        "state_path"
    )
    if not state_path:
        return None
    status = policy["status"]
    if status not in {"awaiting_approval", "approved", "rejected_by_user"} and not policy.get(
        "execute"
    ):
        return None
    from phase_agent.graphs.approval_graph import durable_approval

    envelope = {
        "invocation_id": frame["pending_key"],
        "config_version": frame["current"].get("confirmed_config_version"),
        "proposal_hash": proposal_hash(frame["proposal"]),
        "revision": (frame.get("stored") or {}).get("revision", 0),
    }
    decision = (
        None
        if status == "awaiting_approval"
        else ("reject" if status == "rejected_by_user" else "approve")
    )
    if decision:
        from phase_agent.tools.state.approved_direction import record_action_direction

        action = policy.get("final_action") or frame["proposal"].get("raw_action") or {}
        try:
            record_action_direction(frame["current"], action, state_path, decision)
        except ValueError as error:
            return {
                "status": "rejected",
                "state": frame["current"],
                "reason": str(error),
                "submitted": False,
            }
    result = durable_approval(state_path, envelope, decision)
    if result["status"] == "approval_reconciliation_required":
        return {
            "status": "approval_reconciliation_required",
            "state": frame["current"],
            "reason": (
                "此拒绝决定已交付，但业务记录未确认；请核对拒绝记录，不要重新批准旧方案。"
                if result.get("decision") == "reject"
                else "此批准决定已交付，但业务完成记录未确认；请核对已有文件/任务并对账，不能自动重复执行。"
            ),
            "submitted": False,
        }
    return None

"""Reuse Agent report presentation for Codex-controlled proposals."""

from pathlib import Path


def save_codex_report(state_path, result):
    """Write the Agent's normal proposal artifacts and interaction Markdown."""
    from phase_agent.runtime.interaction_reports import save_interaction
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    state = result.get("state") or {}
    config = state.get("confirmed_config") or {}
    root = (state.get("workspace_layout") or {}).get("workspace_root") or (
        config.get("storage") or (config.get("config") or {}).get("storage") or {}
    ).get("workspace_root")
    if not root:
        raise ValueError("report_requires_project_workspace_root")
    root = Path(root).resolve()
    if not Path(state_path).resolve().is_relative_to(root):
        raise ValueError("report_state_outside_project_workspace")
    if result.get("status") == "awaiting_approval" and not result.get("agent_proposal"):
        pending = state.get("pending_execution_policies") or {}
        plan_id = result.get("record_id")
        row = pending.get(plan_id) if plan_id else None
        if not row and len(pending) == 1:
            row = next(iter(pending.values()))
        if not row or not row.get("agent_proposal"):
            raise ValueError("report_cannot_resolve_pending_proposal")
        result = {**result, "agent_proposal": row["agent_proposal"]}
    reply = format_workflow_reply(result, state_path)
    record = {}
    save_interaction(root, "Codex 结构化提案", reply, reply, state, record=record)
    return record["interaction_report"]

"""Reuse completed evidence for an idle, frozen DFT input request."""


def prepare_saved_dft_input_frame(frame):
    current = frame["loaded_state"]
    if frame["execution_mode"] != "interactive" or frame.get("recovered_results"):
        return frame
    from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection

    stored = (current.get("pending_execution_policies") or {}).get(frame.get("invocation_id"))
    proposal = (stored or {}).get("agent_proposal") or {}
    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    frozen = (frame.get("runtime_adapters") or {}).get("frozen_selection_only")
    if frozen and not stored:
        proposal = reusable_selection(current) or {}
    elif decision not in {"approve", "reject"}:
        return frame
    if (proposal.get("raw_action") or {}).get("tool") != "select_dft_candidates":
        return frame

    from phase_agent.tools.policy.workflow_policy import active_tasks
    from phase_agent.tools.remote.transfer_bundle import has_registered_return
    from phase_agent.tools.workflows.model_refresh_state import refresh_active

    if active_tasks(current) or refresh_active(current) or has_registered_return(current):
        return frame
    if set(current.get("processed_task_ids") or []) - set(
        current.get("feedback_processed_task_ids") or []
    ):
        return frame
    if any(
        not job.get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
        for job in (current.get("remote_finetune_jobs") or {}).values()
    ):
        return frame
    diagrams = current.get("phase_diagrams") or {}
    diagram = (diagrams.get("diagrams") or diagrams).get("mlip") or {}
    if diagram.get("status") != "completed" or diagram.get("model_version") != current.get(
        "active_model_version"
    ):
        return frame
    from phase_agent.tools.state.execution_receipts import recovery_report

    if recovery_report(frame.get("state_path") or frame["effective_config"]["state_path"], current)[
        "unsettled"
    ]:
        return frame
    # The same action/native approval/receipt/finalization nodes remain in use.
    # Only observation and hull rebuilding are unnecessary without new evidence.
    return {
        **frame,
        "saved_input_only": True,
        "pre_reconciled": {"state": current, "reconciled": []},
        "feedback": {"state": current, "recorded": [], "rejected": [], "feedback_count": 0},
        "recovered_count": 0,
        "manual_wait": None,
        "training_handoffs": [],
    }

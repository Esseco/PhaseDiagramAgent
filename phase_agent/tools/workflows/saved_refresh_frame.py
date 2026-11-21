"""Reuse verified idle state only for the existing model-refresh approval gate."""


def prepare_saved_refresh_frame(frame):
    current = frame["loaded_state"]
    if (frame["execution_mode"] != "interactive" or frame.get("recovered_results")
            or frame.get("automatic_results")):
        return frame
    pending = current.get("pending_execution_policies") or {}
    stored = pending.get(frame.get("invocation_id"))
    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    action = ((stored or {}).get("agent_proposal") or {}).get("raw_action") or {}
    client = frame.get("agent_client")
    fresh = (not stored and not pending and decision is None
             and getattr(client, "_codex_structured", False) is True
             and getattr(client, "_frozen_refresh_draft", False) is True)
    saved = (stored and len(pending) == 1 and decision in {"approve", "reject"}
             and action.get("tool") == "prepare_local_batch_files"
             and (action.get("parameters") or {}).get("mode") == "model_refresh_inputs")
    if not (fresh or saved):
        return frame
    refresh = current.get("model_refresh") or {}
    snapshot = frame.get("snapshot") or {}
    active_version = current.get("active_model_version")
    old_diagram = refresh.get("old_diagram") or {}
    if (refresh.get("status") != "approval_required" or not active_version
            or refresh.get("new_model_version") != active_version
            or (current.get("active_model") or {}).get("version") != active_version
            or current.get("confirmed_config_version") != snapshot.get("config_version")
            or current.get("confirmed_config") != snapshot.get("config")
            or old_diagram.get("status") != "completed"
            or old_diagram.get("model_version") != refresh.get("old_model_version")):
        return frame
    if saved and stored.get("config_version") != snapshot.get("config_version"):
        return frame
    from phase_agent.policies.workflow_policy import active_tasks
    from phase_agent.tools.remote.transfer_bundle import has_registered_return
    from phase_agent.tools.workflows.recover_saved_results import recovery_candidate

    if active_tasks(current) or has_registered_return(current):
        return frame
    candidate = recovery_candidate(current, state_path=frame.get("state_path"))
    if candidate.get("pending") or candidate.get("backlog_count"):
        return frame
    collection = current.get("result_collection") or {}
    if collection.get("recovery_error") or collection.get("transfer_errors"):
        return frame
    if any(not job.get("activated") and job.get("status") not in
           {"abandoned", "cancelled", "superseded"}
           for job in (current.get("remote_finetune_jobs") or {}).values()):
        return frame
    from phase_agent.tools.state.execution_receipts import recovery_report

    report = recovery_report(frame.get("state_path") or frame["effective_config"]["state_path"], current)
    if report.get("status") != "clear" or report.get("unsettled"):
        return frame
    return {
        **frame,
        "saved_input_only": True,
        "pre_reconciled": {"state": current, "reconciled": []},
        "feedback": {"state": current, "recorded": [], "rejected": [], "feedback_count": 0},
        "recovered_count": 0,
        "manual_wait": None,
        "training_handoffs": [],
    }

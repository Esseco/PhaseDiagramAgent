"""Fast approval of frozen exact-frame relax inputs only when no work is pending."""


def saved_exact_relax_binding_matches(stored, current):
    """Use the binding the action writer saved, including its content fingerprints."""
    from phase_agent.evidence.project_authority import project_identity

    version = current.get("confirmed_config_version")
    explicit_version = stored.get("config_version")
    binding = stored.get("project_identity")
    if not version or (explicit_version is not None and explicit_version != version):
        return False
    if binding is not None:
        return binding == project_identity(current)
    return explicit_version == version


def prepare_saved_exact_relax_frame(frame):
    if (
        frame.get("execution_mode") != "interactive"
        or frame.get("recovered_results")
        or frame.get("automatic_results")
        or frame.get("result_collector") is not None
        or frame.get("task_runner") is not None
    ):
        return frame
    current = frame["loaded_state"]
    pending = current.get("pending_execution_policies") or {}
    stored = pending.get(frame.get("invocation_id")) or {}
    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    action = (stored.get("agent_proposal") or {}).get("raw_action") or {}
    params = action.get("parameters") or {}
    auth = current.get("exact_candidate_relax_authorization") or {}
    snapshot = frame.get("snapshot") or {}
    active = current.get("active_model_version")
    if (
        len(pending) != 1
        or decision not in {"approve", "reject"}
        or action.get("tool") != "prepare_local_batch_files"
        or action.get("operation") != "prepare_mlip_relax"
        or params.get("mode") != "relax_inputs"
        or params.get("selection_scope") != "candidate_ids"
        or auth.get("status") != "approved_for_preparation"
        or auth.get("submit_authorized") is not False
        or not params.get("candidate_frame_sha256")
        or params.get("candidate_frame_sha256") != auth.get("candidate_frame_sha256")
        or set(action.get("target_ids") or []) != set(auth.get("candidate_frame_sha256") or {})
        or not active
        or params.get("model_version") != active
        or auth.get("model_version") != active
        or (current.get("active_model") or {}).get("version") != active
        or current.get("confirmed_config_version") != snapshot.get("config_version")
        or auth.get("config_version") != snapshot.get("config_version")
        or not saved_exact_relax_binding_matches(stored, current)
        or current.get("confirmed_config") != snapshot.get("config")
        or ((current.get("phase_diagrams") or {}).get("mlip") or {}).get("version")
        != auth.get("hull_version")
    ):
        return frame
    from phase_agent.policies.workflow_policy import active_tasks
    from phase_agent.tools.remote.transfer_bundle import has_registered_return
    from phase_agent.tools.workflows.recover_saved_results import recovery_candidate
    from phase_agent.tools.state.execution_receipts import recovery_report

    if active_tasks(current) or has_registered_return(current):
        return frame
    candidate = recovery_candidate(current, state_path=frame.get("state_path"))
    if candidate.get("pending") or candidate.get("backlog_count"):
        return frame
    collection = current.get("result_collection") or {}
    if collection.get("recovery_error") or collection.get("transfer_errors"):
        return frame
    if any(
        not job.get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
        for job in (current.get("remote_finetune_jobs") or {}).values()
    ):
        return frame
    report = recovery_report(
        frame.get("state_path") or frame["effective_config"]["state_path"], current
    )
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

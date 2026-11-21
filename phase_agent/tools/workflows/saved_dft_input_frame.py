"""Reuse completed evidence for an idle, frozen scientific action."""


def _idle_frozen_relax_mc(frame, state, *, pending, action, decision):
    from phase_agent.persistence.ledger.mc_energy_repair import (
        missing_verified_repair_tasks,
        stale_mc_energy_model_snapshots,
    )
    from phase_agent.persistence.shared_dft import settings
    from phase_agent.tools.relax_mc_batch.plan import frozen_plan_matches

    snapshot = frame.get("snapshot") or {}
    collection = state.get("result_collection") or {}
    return (
        pending is not None
        and decision in {"approve", "reject"}
        and len(state.get("pending_execution_policies") or {}) == 1
        and frame.get("result_collector") is None
        and not (frame.get("runtime_adapters") or {}).get("dft_recovery_decision")
        and snapshot.get("config_version") == state.get("confirmed_config_version")
        and snapshot.get("config") == state.get("confirmed_config")
        and not collection.get("recovery_error")
        and not collection.get("transfer_errors")
        and not settings(state, frame.get("effective_config"))
        and frozen_plan_matches(action, state)
        and not missing_verified_repair_tasks(state)
        and not stale_mc_energy_model_snapshots(state)
    )


def _idle_exact_candidate_relax(frame, state, action, *, stored, decision, after_collection=False):
    """Reuse the acknowledged frame only when every execution dependency is current."""
    from hashlib import sha256
    from pathlib import Path
    from phase_agent.evidence.dft_failure_policy import unreviewed_failure_batches
    from phase_agent.persistence.shared_dft import settings
    from phase_agent.tools.workflows.recover_saved_results import recovery_candidate
    from phase_agent.tools.workflows.saved_exact_relax_frame import (
        saved_exact_relax_binding_matches,
    )

    params = action.get("parameters") or {}
    auth = state.get("exact_candidate_relax_authorization") or {}
    snapshot = frame.get("snapshot") or {}
    hull = (state.get("phase_diagrams") or {}).get("mlip") or {}
    ids = action.get("target_ids") or []
    hashes = params.get("candidate_frame_sha256")
    pending = state.get("pending_execution_policies") or {}
    runtime_model = (frame.get("effective_config") or {}).get("mlip") or {}
    runtime_version = runtime_model.get("version") or runtime_model.get("name")
    active_model = state.get("active_model") or {}
    if (
        action.get("tool") != "prepare_local_batch_files"
        or action.get("operation") != "prepare_mlip_relax"
        or params.get("mode") != "relax_inputs"
        or params.get("selection_scope") != "candidate_ids"
        or not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
        or not isinstance(hashes, dict)
        or set(ids) != set(hashes)
        or auth.get("status") != "approved_for_preparation"
        or auth.get("submit_authorized") is not False
        or not str(auth.get("user_instruction") or "").strip()
        or auth.get("candidate_frame_sha256") != hashes
        or not state.get("active_model_version")
        or auth.get("model_version") != state.get("active_model_version")
        or params.get("model_version") != state.get("active_model_version")
        or (runtime_version and runtime_version != state.get("active_model_version"))
        or (
            active_model.get("version")
            and active_model.get("version") != state.get("active_model_version")
        )
        or not state.get("confirmed_config_version")
        or auth.get("config_version") != state.get("confirmed_config_version")
        or snapshot.get("config_version") != state.get("confirmed_config_version")
        or snapshot.get("config") != state.get("confirmed_config")
        or hull.get("status") != "completed"
        or hull.get("model_version") != state.get("active_model_version")
        or not hull.get("version")
        or auth.get("hull_version") != hull.get("version")
        or (
            stored
            and (
                len(pending) != 1
                or decision not in {"approve", "reject"}
                or not saved_exact_relax_binding_matches(stored, state)
            )
        )
        or (not stored and (pending or decision is not None))
        or (frame.get("result_collector") is not None and not after_collection)
        or (frame.get("runtime_adapters") or {}).get("dft_recovery_decision")
        or frame.get("replay_record") is not None
        or state.get("pending_dft_recovery_question")
        or state.get("scientific_feedback_errors")
        or state.get("result_validation_errors")
        or (state.get("reference_dft_startup") or {}).get("status") not in {None, "completed"}
        or (state.get("result_collection") or {}).get("recovery_error")
        or (state.get("result_collection") or {}).get("transfer_errors")
        or settings(state, frame.get("effective_config") or {})
    ):
        return False
    manager = frame.get("manager")
    if manager is None:
        return False
    candidates = {}
    for row in state.get("qbc_candidates") or []:
        candidate_id = row.get("candidate_id")
        if candidate_id in hashes:
            if candidate_id in candidates:
                return False
            candidates[candidate_id] = row
    if set(candidates) != set(ids):
        return False
    for candidate_id in ids:
        row = candidates[candidate_id]
        digest = hashes[candidate_id]
        path = Path(row.get("structure_path") or "")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or row.get("structure_sha256") != digest
            or row.get("structure_id") not in manager.data.get("structures", {})
            or not path.is_file()
        ):
            return False
        try:
            if sha256(path.read_bytes()).hexdigest() != digest:
                return False
        except OSError:
            return False
    candidate = recovery_candidate(state, state_path=frame.get("state_path"))
    return not (
        candidate.get("pending")
        or candidate.get("backlog_count")
        or unreviewed_failure_batches(state)
    )


def _reusable_frozen_phase_status(record, evidence):
    """Keep verified exclusions unchanged during frozen local input preparation."""
    status = evidence.get("status")
    if status == "identified":
        return record.get("status") == "completed"
    # Legacy path/timestamp scopes are insufficient to reuse an exclusion.
    if record.get("phase") is not None or not evidence.get("portable_classification_scope"):
        return False
    if status == "outside_boundary":
        return (
            record.get("status") == "outside_search_boundary"
            and bool(evidence.get("observed_phase"))
            and record.get("observed_phase") == evidence["observed_phase"]
            and evidence.get("reason")
            in {"phase_outside_search_boundary", "phase_outside_composition_boundary"}
        )
    # This is the classifier's explicit no-match result, not a missing frame,
    # transient parse error, or scientific determination of the phase.
    return (
        status == "unknown"
        and record.get("status") == "pending_phase_identification"
        and record.get("observed_phase") is None
        and evidence.get("reason") == "ValueError: unrecognized_phase:X"
    )


def reuse_exact_after_collection(frame):
    """Reuse only after normal recovery and verified, unchanged scientific inputs."""

    def fallback(reason):
        frame["idle_exact_feedback_reason"] = reason
        return False

    if (
        frame.get("execution_mode") != "interactive"
        or frame.get("combined_results")
        or frame.get("training_returns")
        or frame.get("training_handoffs")
        or frame.get("relax_recovery_updated") is not False
        or (frame.get("pre_reconciled") or {}).get("reconciled")
    ):
        return fallback("new_or_unsettled_results")
    collection = frame.get("collection_report") or {}
    metadata = collection.get("metadata_refresh_report") or {}
    if (
        collection.get("status") in {"needs_attention", "partial", "waiting"}
        or any(
            collection.get(key)
            for key in (
                "recovered_count",
                "invalid_count",
                "missing_result_count",
                "missing_marker_count",
                "missing_structure_count",
                "transfer_errors",
            )
        )
        or metadata.get("invalid_count")
        or metadata.get("missing_structure_count")
    ):
        return fallback("collection_needs_review")
    state = frame["pre_reconciled"]["state"]
    stored = (state.get("pending_execution_policies") or {}).get(frame.get("invocation_id"))
    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    client = frame.get("agent_client")
    if stored:
        action = (stored.get("agent_proposal") or {}).get("raw_action") or {}
    elif getattr(client, "_codex_structured", False) is True:
        action = getattr(client, "_frozen_exact_relax_action", None) or {}
    else:
        return fallback("no_frozen_action")
    if (
        action.get("tool") != "prepare_local_batch_files"
        or action.get("operation") != "prepare_mlip_relax"
        or (action.get("parameters") or {}).get("selection_scope") != "candidate_ids"
    ):
        return fallback("action_out_of_scope")
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
    from phase_agent.tools.remote.transfer_bundle import has_registered_return
    from phase_agent.policies.workflow_policy import active_tasks
    from phase_agent.tools.workflows.model_refresh_state import refresh_active
    from phase_agent.tools.state.execution_receipts import recovery_report
    from phase_agent.analysis.phase.identify_result_phase import (
        _classification_scope,
        _configured_phase,
        _portable_classification_scope,
    )
    from phase_agent.science.structures.load_result_structure import load_result_structure
    from phase_agent.analysis.convergence.mlip_relaxation import relaxation_failure
    from phase_agent.persistence.ledger.result_validation import (
        InvalidScientificResult,
        validate_scientific_result,
    )

    collector, runner = frame.get("result_collector"), frame.get("task_runner")
    if (collector is not None and type(collector) is not RemoteBatchRunner) or (
        runner is not None and type(runner) is not ManualUploadBatchRunner
    ):
        return fallback("collector_unverified")
    if any(
        not job.get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
        for job in (state.get("remote_finetune_jobs") or {}).values()
    ):
        return fallback("training_pending")
    unprocessed = set(state.get("processed_task_ids") or []) - set(
        state.get("feedback_processed_task_ids") or []
    )
    if unprocessed:
        by_id = {
            row.get("task_id"): row
            for row in state.get("tasks") or []
            if row.get("task_id") in unprocessed
        }
        if any(
            not (
                isinstance(tid, str)
                and tid.startswith("webui-")
                and tid.endswith(":task")
                and (by_id.get(tid) or {}).get("stage") is None
                and (by_id.get(tid) or {}).get("status") in {"completed", "cancelled"}
            )
            for tid in unprocessed
        ):
            return fallback("feedback_backlog")
    if (
        not _idle_exact_candidate_relax(
            frame, state, action, stored=stored, decision=decision, after_collection=True
        )
        or active_tasks(state)
        or refresh_active(state)
        or has_registered_return(state)
    ):
        return fallback("action_binding_or_recovery")
    state_path = (
        frame.get("runtime_state_path")
        or frame.get("state_path")
        or frame["effective_config"].get("state_path")
    )
    if not state_path:
        return fallback("state_path_missing")
    report = recovery_report(state_path, state)
    if report.get("status") != "clear" or report.get("unsettled"):
        return fallback("receipts_unsettled")
    # Legacy or unresolved evidence is rechecked normally. For portable evidence,
    # prove the current rule scope and final-frame content before reusing it.
    manager = frame["manager"]
    records = state.get("phase_records") or []
    if records:
        from phase_agent.tools.workflows.apply_scientific_feedback import phase_records_checksum

        if state.get("phase_diagram_input_checksum") != phase_records_checksum(records):
            return fallback("phase_diagram_input_changed")
    recorded_tasks = {record.get("source_task_id") for record in records}
    for task in state.get("tasks") or []:
        try:
            validate_scientific_result(task)
        except InvalidScientificResult:
            return fallback("scientific_result_invalid")
        if relaxation_failure(task):
            return fallback("relaxation_failure")
        if (
            task.get("status") == "completed"
            and task.get("stage")
            in {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}
            and (task.get("outputs") or {}).get("energy") is not None
            and not task.get("perturbation_batch_id")
            and task.get("task_id") not in recorded_tasks
        ):
            return fallback("scientific_task_missing_phase_record")
    scopes = {}
    last_file = last_digest = last_structure = None
    tasks = {task.get("task_id"): task for task in state.get("tasks") or []}
    for record in records:
        task = tasks.get(record.get("source_task_id"))
        if not task:
            return fallback("phase_source_task_missing")
        output = task.get("outputs") or {}
        expected_version = (
            output.get("dft_version") or "atomate"
            if record.get("energy_method") == "dft"
            else task.get("model_version")
            or output.get("mlip_version")
            or output.get("mlip_relax_version")
        )
        if (
            task.get("status") != "completed"
            or task.get("structure_id") != record.get("structure_id")
            or (record.get("model_version") or record.get("source_version")) != expected_version
            or output.get("energy") != record.get("energy")
            or output.get("energy_unit") != record.get("energy_unit")
            or task.get("stage") != record.get("stage")
            or task.get("checks_passed", True) != record.get("checks_passed")
            or task.get("converged") != record.get("converged")
            or (
                output.get("composition") is not None
                and output.get("composition") != record.get("composition")
            )
            or output.get("actual_phase") != record.get("phase")
            or output.get("phase_identification") != record.get("phase_identification")
        ):
            return fallback("phase_source_binding_changed")
        evidence = record.get("phase_identification") or {}
        if not _reusable_frozen_phase_status(record, evidence):
            return fallback("phase_status_unverified")
        try:
            inputs = {
                "structure_path": record.get("structure_path"),
                "structure": record.get("structure"),
                "final_frame_valid": record.get("final_frame_valid"),
                "final_frame_index": record.get("final_frame_index"),
                "training_frame_index": record.get("training_frame_index"),
            }
            file_key = (
                inputs["structure_path"],
                inputs["final_frame_valid"],
                inputs["final_frame_index"],
                inputs["training_frame_index"],
            )
            if inputs["structure"] is None and file_key == last_file:
                from hashlib import sha256
                from pathlib import Path

                digest = sha256(Path(inputs["structure_path"]).read_bytes()).hexdigest()
                if digest == last_digest:
                    structure = last_structure
                else:
                    structure, digest = load_result_structure(inputs)
            else:
                structure, digest = load_result_structure(inputs)
            last_file = file_key if inputs["structure"] is None else None
            last_digest, last_structure = digest, structure
            configured = _configured_phase(structure, manager)
        except Exception:
            return fallback("phase_structure_unverified")
        if not digest or digest != evidence.get("structure_sha256"):
            return fallback("phase_structure_changed")
        portable = evidence.get("portable_classification_scope")
        scope_key = (configured, bool(portable))
        if scope_key not in scopes:
            scopes[scope_key] = (
                _portable_classification_scope(
                    manager, frame.get("phase_references"), configured_phase=configured
                )
                if portable
                else _classification_scope(
                    manager, frame.get("phase_references"), configured_phase=configured
                )
            )
        if (
            not scopes[scope_key]
            or (portable or evidence.get("classification_scope")) != scopes[scope_key]
        ):
            return fallback("phase_rules_changed")
    from phase_agent.analysis.feedback.export_dft_products import export_dft_products

    try:
        verified = (
            export_dft_products(
                state,
                frame["effective_config"].get("phase_diagram_directory"),
                check_reuse_only=True,
            )
            is True
        )
    except (OSError, ValueError, TypeError, KeyError):
        return fallback("export_unverified")
    if not verified:
        return fallback("export_unverified")
    frame["idle_exact_feedback_reason"] = "verified_unchanged"
    return True


def prepare_saved_dft_input_frame(frame):
    current = frame["loaded_state"]
    if (
        frame["execution_mode"] != "interactive"
        or frame.get("recovered_results")
        or frame.get("automatic_results")
    ):
        return frame
    from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection

    stored = (current.get("pending_execution_policies") or {}).get(frame.get("invocation_id"))
    proposal = (stored or {}).get("agent_proposal") or {}
    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    frozen = (frame.get("runtime_adapters") or {}).get("frozen_selection_only")
    client = frame.get("agent_client")
    generation_draft = (
        not stored
        and decision is None
        and getattr(client, "_codex_structured", False) is True
        and getattr(client, "_frozen_generation_draft", False) is True
    )
    mc_draft = (
        not stored
        and not current.get("pending_execution_policies")
        and decision is None
        and getattr(client, "_codex_structured", False) is True
        and isinstance(getattr(client, "_frozen_mc_action", None), dict)
    )
    dft_draft = (
        not stored
        and not current.get("pending_execution_policies")
        and decision is None
        and getattr(client, "_codex_structured", False) is True
        and isinstance(getattr(client, "_frozen_dft_action", None), dict)
    )
    exact_draft = (
        not stored
        and not current.get("pending_execution_policies")
        and decision is None
        and getattr(client, "_codex_structured", False) is True
        and isinstance(getattr(client, "_frozen_exact_relax_action", None), dict)
    )
    if frozen and not stored:
        proposal = reusable_selection(current) or {}
    elif generation_draft:
        proposal = {"raw_action": {"tool": "generate_branches"}}
    elif mc_draft:
        proposal = {"raw_action": client._frozen_mc_action}
    elif dft_draft:
        proposal = {"raw_action": client._frozen_dft_action}
    elif exact_draft:
        proposal = {"raw_action": client._frozen_exact_relax_action}
    elif decision not in {"approve", "reject"}:
        return frame
    action = proposal.get("raw_action") or {}
    tool = action.get("tool")
    energy_repair = (
        tool == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "mc_energy_repair_inputs"
    )
    relax_mc = (
        tool == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "relax_mc_inputs"
    )
    exact_relax = (
        tool == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "relax_inputs"
        and (action.get("parameters") or {}).get("selection_scope") == "candidate_ids"
    )
    if tool not in {"select_dft_candidates", "generate_branches", "allocate_mc_bohb"} and not (
        energy_repair or relax_mc or exact_relax
    ):
        return frame
    if dft_draft:
        review = action.get("scientific_decision") or {}
        rows = (action.get("parameters") or {}).get("decisions") or []
        ids = [row.get("candidate_id") for row in rows if isinstance(row, dict)]
        candidates = current.get("qbc_candidates") or []
        snapshot = frame.get("snapshot") or {}
        if (
            tool != "select_dft_candidates"
            or not action.get("_decision_team")
            or not rows
            or len(ids) != len(rows)
            or len(set(ids)) != len(ids)
            or any(not isinstance(item, str) or not item for item in ids)
            or not set(ids).issubset({row.get("candidate_id") for row in candidates})
            or not current.get("active_model_version")
            or not current.get("confirmed_config_version")
            or review.get("model_version") != current.get("active_model_version")
            or review.get("config_version") != current.get("confirmed_config_version")
            or snapshot.get("config_version") != current.get("confirmed_config_version")
            or snapshot.get("config") != current.get("confirmed_config")
        ):
            return frame
    if exact_relax:
        # Always collect registered results and reconcile receipts first.
        # The verified no-change path is selected after observation.
        return frame
    if relax_mc and not _idle_frozen_relax_mc(
        frame, current, pending=stored, action=action, decision=decision
    ):
        return frame
    if energy_repair:
        from phase_agent.tools.remote.integrity import payload_checksum

        preview = (action.get("parameters") or {}).get("energy_repair_preview") or {}
        model = preview.get("model") or {}
        active = current.get("active_model") or {}
        hull = (current.get("phase_diagrams") or {}).get("mlip") or {}
        frames = preview.get("frames") or []
        if (
            not stored
            or decision not in {"approve", "reject"}
            or len(current.get("pending_execution_policies") or {}) != 1
            or not preview.get("checksum")
            or payload_checksum({key: value for key, value in preview.items() if key != "checksum"})
            != preview.get("checksum")
            or preview.get("config_version") != current.get("confirmed_config_version")
            or preview.get("config_version") != (frame.get("snapshot") or {}).get("config_version")
            or model.get("version") != current.get("active_model_version")
            or active.get("version") != model.get("version")
            or model.get("sha256") != active.get("sha256")
            or model.get("path") != (active.get("remote_model_path") or active.get("model_path"))
            or hull.get("status") != "completed"
            or preview.get("hull_version") != hull.get("version")
            or not frames
            or {row.get("source_task_id") for row in frames} != set(action.get("target_ids") or [])
            or preview.get("task_count") != len(frames)
        ):
            return frame
    if tool == "allocate_mc_bohb":
        action = proposal.get("raw_action") or {}
        params = action.get("parameters") or {}
        preview = params.get("budget_preview") or {}
        model = current.get("active_model_version")
        pool = (current.get("branch_hull_batches") or {}).get(
            params.get("hull_reference_version")
        ) or {}
        from phase_agent.analysis.phase.mc_snapshot import mc_model_snapshot

        try:
            diagram = mc_model_snapshot(current, model)
        except ValueError:
            return frame
        targets = action.get("target_ids") or []
        allocations = preview.get("allocations") or []
        if (
            not (mc_draft or (stored and decision in {"approve", "reject"}))
            or not model
            or not current.get("confirmed_config_version")
            or current.get("confirmed_config_version")
            != (frame.get("snapshot") or {}).get("config_version")
            or (
                stored
                and stored.get("config_version")
                and stored.get("config_version") != current.get("confirmed_config_version")
            )
            or not targets
            or len(set(targets)) != len(targets)
            or len(targets) > 500
            or pool.get("version") != params.get("hull_reference_version")
            or pool.get("model_version") != model
            or diagram.get("status") != "completed"
            or diagram.get("version") != params.get("phase_diagram_version")
            or preview.get("model_version") != model
            or preview.get("phase_diagram_version") != diagram.get("version")
            or preview.get("hull_reference_version") != pool.get("version")
            or not preview.get("allocation_checksum")
            or not allocations
            or any(row.get("branch_id") not in targets for row in allocations)
        ):
            return frame
    if tool == "generate_branches" and not (generation_draft or stored):
        return frame
    if tool == "generate_branches" and (
        generation_draft and current.get("pending_execution_policies")
    ):
        return frame

    from phase_agent.policies.workflow_policy import active_tasks
    from phase_agent.tools.remote.transfer_bundle import has_registered_return
    from phase_agent.tools.workflows.model_refresh_state import refresh_active

    if active_tasks(current) or refresh_active(current):
        return frame
    if tool == "generate_branches":
        has_return = has_registered_return(current, ignore_known_diagnostics=True)
    else:
        has_return = has_registered_return(current)
    if has_return:
        return frame
    unprocessed = set(current.get("processed_task_ids") or []) - set(
        current.get("feedback_processed_task_ids") or []
    )
    # Only known terminal UI bookkeeping tasks may be ignored; unknown IDs
    # and all scientific stages continue through the full feedback path.
    task_by_id = (
        {
            row.get("task_id"): row
            for row in current.get("tasks") or []
            if row.get("task_id") in unprocessed
        }
        if unprocessed
        else {}
    )
    unprocessed = {
        identifier
        for identifier in unprocessed
        if not (
            isinstance(identifier, str)
            and identifier.startswith("webui-")
            and identifier.endswith(":task")
            and (task_by_id.get(identifier) or {}).get("stage") is None
            and (task_by_id.get(identifier) or {}).get("status") in {"completed", "cancelled"}
        )
    }
    if unprocessed:
        return frame
    if any(
        not job.get("activated")
        and job.get("status") not in {"abandoned", "cancelled", "superseded"}
        for job in (current.get("remote_finetune_jobs") or {}).values()
    ):
        return frame
    if tool != "allocate_mc_bohb":
        diagrams = current.get("phase_diagrams") or {}
        diagram = (diagrams.get("diagrams") or diagrams).get("mlip") or {}
        if diagram.get("status") != "completed" or diagram.get("model_version") != current.get(
            "active_model_version"
        ):
            return frame
    from phase_agent.tools.state.execution_receipts import recovery_report

    report = recovery_report(
        frame.get("state_path") or frame["effective_config"]["state_path"], current
    )
    if report["unsettled"] or ((relax_mc or exact_relax or dft_draft) and report.get("status") != "clear"):
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

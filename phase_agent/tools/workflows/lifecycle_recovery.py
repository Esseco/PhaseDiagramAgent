"""Result collection, scientific feedback, waiting and round assessment."""

from copy import deepcopy
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback
from phase_agent.tools.workflows.lifecycle_support import (
    _save_runner_state,
    _branch_candidates_for_agent,
    _count_newly_recovered,
    _dft_comparison_evaluator,
)


def _collect_workflow_results(frame):
    automatic_results = frame["automatic_results"]
    collection_report = frame["collection_report"]
    loaded_state = frame["loaded_state"]
    manager = frame["manager"]
    task_runner = frame["task_runner"]
    result_collector = frame["result_collector"]
    recovered_results = frame["recovered_results"]
    effective_config = frame["effective_config"]
    execution_mode = frame["execution_mode"]
    runtime_adapters = frame["runtime_adapters"]
    state_path = frame["state_path"]
    from phase_agent.tools.state.restore_generation_gate import restore_generation_gate

    loaded_state = restore_generation_gate(loaded_state, manager)
    from phase_agent.tools.local.recover_remote_training import recover_remote_training

    loaded_state, training_returns = recover_remote_training(loaded_state)
    collector = task_runner or result_collector
    if collector is not None:
        if hasattr(collector, "collect_results_with_report"):
            collected = collector.collect_results_with_report(loaded_state)
            automatic_results = collected["results"]
            collection_report = collected["report"]
        else:
            automatic_results = collector.collect_results(loaded_state)
    combined_results = [*(recovered_results or []), *automatic_results]
    pre_reconciled = reconcile_task_results(loaded_state, combined_results)
    from phase_agent.tools.workflows.comparison_model_registry import remember_comparison_models

    remember_comparison_models(pre_reconciled["state"], effective_config)
    if execution_mode == "interactive":
        from phase_agent.tools.state.dft_recovery_decision import update_dft_recovery_question

        pre_reconciled["state"] = update_dft_recovery_question(
            pre_reconciled["state"], runtime_adapters.get("dft_recovery_decision")
        )
    runtime_state_path = state_path or effective_config.get("state_path")
    phase_cache_path = runtime_adapters.get("phase_identification_cache_path")
    if phase_cache_path is None and runtime_state_path:
        from phase_agent.configuration.session.phase_cache_location import phase_cache_location

        phase_cache_path = phase_cache_location(runtime_state_path)
    return {
        **frame,
        "training_returns": training_returns,
        "loaded_state": loaded_state,
        "automatic_results": automatic_results,
        "collection_report": collection_report,
        "combined_results": combined_results,
        "pre_reconciled": pre_reconciled,
        "runtime_state_path": runtime_state_path,
        "phase_cache_path": phase_cache_path,
    }


def _advance_training_workflow(frame, *, graph=None):
    from phase_agent.tools.local.training_handoff import advance_training_handoffs

    current, waits = advance_training_handoffs(
        frame["pre_reconciled"]["state"],
        frame["effective_config"],
        state_path=frame.get("state_path") or frame["effective_config"].get("state_path"),
        graph=graph,
    )
    frame["pre_reconciled"]["state"] = current
    return {**frame, "loaded_state": current, "training_handoffs": waits}


def _review_training_direction(frame):
    from phase_agent.tools.local.training_agent_review import review_training_choice

    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    message = (frame.get("runtime_adapters") or {}).get("user_message")
    if decision in {"approve", "reject", "confirm_sensitive"}:
        message = None  # Approval consumes the saved plan; it never requests a new review.
    current, waits = review_training_choice(
        frame["pre_reconciled"]["state"],
        frame.get("training_handoffs") or [],
        frame.get("agent_client"),
        state_path=frame.get("state_path") or frame.get("runtime_state_path"),
        user_message=message,
    )
    frame["pre_reconciled"]["state"] = current
    return {**frame, "loaded_state": current, "training_handoffs": waits}


def _analyze_workflow_results(frame):
    pre_reconciled = frame["pre_reconciled"]
    combined_results = frame["combined_results"]
    manager = frame["manager"]
    effective_config = frame["effective_config"]
    runtime_adapters = frame["runtime_adapters"]
    phase_references = frame["phase_references"]
    phase_cache_path = frame["phase_cache_path"]
    execution_mode = frame["execution_mode"]
    collection_report = frame["collection_report"]
    feedback = apply_scientific_feedback(
        pre_reconciled["state"],
        combined_results,
        manager=manager,
        ledger_path=effective_config.get("ledger_path"),
        phase_diagram_directory=effective_config.get("phase_diagram_directory"),
        final_frame_mlip_evaluator=_dft_comparison_evaluator(
            runtime_adapters, effective_config, pre_reconciled["state"]
        ),
        active_model_version=(effective_config.get("mlip") or {}).get("version")
        or (effective_config.get("mlip") or {}).get("name"),
        phase_references=phase_references,
        phase_identification_cache_path=phase_cache_path,
    )
    from phase_agent.analysis.phase.update_local_mlip_hull_pool import update_local_mlip_hull_pool

    feedback["state"] = update_local_mlip_hull_pool(
        feedback["state"],
        config=effective_config,
        path=effective_config.get("branch_energy_pool_ledger_path"),
    )
    feedback["state"]["budget_limits"] = deepcopy(effective_config.get("budgets") or {})
    feedback["state"]["branch_candidates"] = _branch_candidates_for_agent(manager)
    total_limit = (effective_config.get("budgets") or {}).get("total_relative_cost")
    if total_limit is not None:
        used = float(
            (feedback["state"].get("budget_usage") or {}).get("total_relative_cost", 0) or 0
        )
        reserved = float(feedback["state"].get("reserved_relative_cost", 0) or 0)
        feedback["state"]["budget_remaining"] = max(0.0, float(total_limit) - used - reserved)
    recovered_count = _count_newly_recovered(pre_reconciled["reconciled"])
    from phase_agent.tools.remote.summarize_manual_upload_wait import summarize_manual_upload_wait

    manual_wait = (
        summarize_manual_upload_wait(feedback["state"], recovered_count=recovered_count)
        if execution_mode == "interactive"
        else None
    )
    if manual_wait is not None and collection_report is not None:
        manual_wait["result_collection"] = deepcopy(collection_report)
    from phase_agent.tools.local.rebuild_relax_inputs import is_relax_rebuild_request

    rebuilding = is_relax_rebuild_request(runtime_adapters.get("user_message")) or any(
        (((row.get("agent_proposal") or {}).get("raw_action") or {}).get("parameters") or {}).get(
            "rebuild_inputs"
        )
        for row in feedback["state"].get("pending_execution_policies", {}).values()
    )
    recovery_question = feedback["state"].get("pending_dft_recovery_question")
    return {
        **frame,
        "feedback": feedback,
        "recovered_count": recovered_count,
        "manual_wait": manual_wait,
        "rebuilding": rebuilding,
        "recovery_question": recovery_question,
    }


def _workflow_wait_gate(frame):
    recovery_question = frame["recovery_question"]
    execution_mode = frame["execution_mode"]
    feedback = frame["feedback"]
    state_path = frame["state_path"]
    effective_config = frame["effective_config"]
    recovered_count = frame["recovered_count"]
    collection_report = frame["collection_report"]
    snapshot = frame["snapshot"]
    pre_reconciled = frame["pre_reconciled"]
    manual_wait = frame["manual_wait"]
    rebuilding = frame["rebuilding"]
    runtime_path = state_path or effective_config.get("state_path")
    if runtime_path:
        from phase_agent.graphs.execution_recovery_graph import execution_recovery_report

        report = execution_recovery_report(runtime_path, feedback["state"])
        feedback["state"]["execution_recovery_report"] = report
        if report["unsettled"]:
            _save_runner_state(feedback["state"], runtime_path)
            return {
                "status": "execution_reconciliation_required",
                "state": feedback["state"],
                "recovery_report": report,
                "steps_executed": 0,
                "submitted": False,
                "recovered_count": recovered_count,
            }
    training_waits = frame.get("training_handoffs") or []
    review_ready = bool(training_waits) and all(
        row.get("stage")
        in {"validation_prerequisites_required", "execution_plan_ready", "execution_plan_approved"}
        and row.get("cv_review")
        for row in training_waits
    )
    if review_ready:
        feedback["state"]["training_validation_planning"] = {
            "reviews": [deepcopy(row["cv_review"]) for row in training_waits],
            "agent_model_reviews": [
                deepcopy(candidate.get("agent_review"))
                for candidate in feedback["state"].get("candidate_models", {}).values()
                if candidate.get("old_model_version")
                == feedback["state"].get("active_model_version")
                and candidate.get("agent_review")
            ],
            "instruction": "微调后已有方向与具体计划。复用保存的followup_action，经现有动作校验形成完整执行方案，统一展示执行范围、成本与目的，一次审批确认方向和本次动作。不要重新选择方向，不默认改为独立验证配置，不重复本轮训练或自动激活；新增科学动作仍另行审批。",
        }
        pending = feedback["state"].get("pending_execution_policies") or {}
        for record_id, record in list(pending.items()):
            proposal = record.get("agent_proposal") or {}
            action = proposal.get("raw_action") or {}
            if (
                action.get("tool")
                or action.get("action_type")
                or proposal.get("recommended_action")
            ) == "update_mlip":
                feedback["state"].setdefault("superseded_training_proposals", []).append(
                    {
                        "record_id": record_id,
                        "record": deepcopy(record),
                        "reason": "Training results already recovered; proceed to validation planning.",
                    }
                )
                pending.pop(record_id)
        # Training report is evidence for the next action, not an external-result wait.
    if training_waits and not review_ready:
        from phase_agent.persistence.memory.collect_memory_candidates import (
            collect_memory_candidates,
        )

        feedback["state"] = collect_memory_candidates(feedback["state"])
        _save_runner_state(feedback["state"], runtime_path)
        return {
            "status": "training_handoff",
            "state": feedback["state"],
            "training_handoffs": frame["training_handoffs"],
            "reason": "\n".join(row["reason"] for row in frame["training_handoffs"]),
            "steps_executed": 0,
            "submitted": False,
        }
    if frame.get("training_returns") and "training_handoffs" not in frame:
        from phase_agent.tools.local.recover_remote_training import training_result_message

        _save_runner_state(feedback["state"], runtime_path)
        return {
            "status": "training_results_received",
            "state": feedback["state"],
            "reason": "\n".join(training_result_message(r) for r in frame["training_returns"]),
            "steps_executed": 0,
            "submitted": False,
        }
    if recovery_question and execution_mode == "interactive":
        _save_runner_state(feedback["state"], state_path or effective_config.get("state_path"))
        return {
            "status": "awaiting_dft_recovery_decision",
            "state": feedback["state"],
            "dft_recovery_question": recovery_question,
            "steps_executed": 0,
            "submitted": False,
            "scientific_feedback": {
                key: value for key, value in feedback.items() if key != "state"
            },
            "recovered_count": recovered_count,
            "result_collection": collection_report,
            "effective_config": effective_config,
            "config_version": snapshot["config_version"],
            "reconciled": pre_reconciled["reconciled"],
        }
    from phase_agent.tools.workflows.model_refresh_state import refresh_active

    if manual_wait and not rebuilding and not refresh_active(feedback["state"]):
        wait_state = feedback["state"]
        _save_runner_state(wait_state, state_path or effective_config.get("state_path"))
        return {
            "status": "awaiting_manual_submission",
            "state": wait_state,
            "manual_wait": manual_wait,
            "recovered_count": recovered_count,
            "scientific_feedback": {
                key: value for key, value in feedback.items() if key != "state"
            },
            "reconciled": pre_reconciled["reconciled"],
            "steps_executed": 0,
            "submitted": False,
            "config_version": snapshot["config_version"],
            "effective_config": effective_config,
            "result_collection": collection_report,
        }
    return frame


def _assess_workflow_round(frame):
    feedback = frame["feedback"]
    effective_config = frame["effective_config"]
    manager = frame["manager"]
    runtime_adapters = frame["runtime_adapters"]
    state_path = frame["state_path"]
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment
    from phase_agent.tools.workflows.refresh_dft_comparisons import refresh_dft_comparisons

    assessment = post_dft_assessment(feedback["state"], effective_config)
    feedback["state"] = refresh_dft_comparisons(
        feedback["state"],
        assessment=assessment,
        manager=manager,
        evaluator=_dft_comparison_evaluator(runtime_adapters, effective_config, feedback["state"]),
    )
    # Persist analysis products before asking the LLM what to do next. The
    # proposal must follow the recovered-round analysis, not precede its export.
    from phase_agent.analysis.feedback.export_dft_products import export_dft_products

    export_dft_products(feedback["state"], effective_config.get("phase_diagram_directory"))
    from phase_agent.analysis.phase.combined_phase_diagram import refresh_combined_phase_diagram

    refresh_combined_phase_diagram(
        feedback["state"], effective_config.get("phase_diagram_directory")
    )
    _save_runner_state(feedback["state"], state_path or effective_config.get("state_path"))
    return {
        **frame,
        "feedback": feedback,
    }

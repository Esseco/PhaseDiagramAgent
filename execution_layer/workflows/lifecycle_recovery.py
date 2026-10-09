"""Result collection, scientific feedback, waiting and round assessment."""
from copy import deepcopy
from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.workflows.apply_scientific_feedback import apply_scientific_feedback
from execution_layer.workflows.lifecycle_support import (
    _save_runner_state, _branch_candidates_for_agent, _count_newly_recovered,
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
    from execution_layer.state.restore_generation_gate import restore_generation_gate
    loaded_state = restore_generation_gate(loaded_state, manager)
    from execution_layer.local.recover_remote_training import recover_remote_training
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
    from execution_layer.workflows.comparison_model_registry import remember_comparison_models
    remember_comparison_models(pre_reconciled["state"], effective_config)
    if execution_mode == "interactive":
        from execution_layer.state.dft_recovery_decision import update_dft_recovery_question
        pre_reconciled["state"] = update_dft_recovery_question(
            pre_reconciled["state"], runtime_adapters.get("dft_recovery_decision"))
    runtime_state_path = state_path or effective_config.get("state_path")
    phase_cache_path = runtime_adapters.get("phase_identification_cache_path")
    if phase_cache_path is None and runtime_state_path:
        from config_layer.session.phase_cache_location import phase_cache_location
        phase_cache_path = phase_cache_location(runtime_state_path)
    return {**frame,
            "training_returns": training_returns,
            "loaded_state": loaded_state,
            "automatic_results": automatic_results,
            "collection_report": collection_report,
            "combined_results": combined_results,
            "pre_reconciled": pre_reconciled,
            "runtime_state_path": runtime_state_path,
            "phase_cache_path": phase_cache_path,
    }


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
        pre_reconciled["state"], combined_results, manager=manager,
        ledger_path=effective_config.get("ledger_path"),
        phase_diagram_directory=effective_config.get("phase_diagram_directory"),
        final_frame_mlip_evaluator=_dft_comparison_evaluator(runtime_adapters, effective_config, pre_reconciled["state"]),
        active_model_version=(effective_config.get("mlip") or {}).get("version")
            or (effective_config.get("mlip") or {}).get("name"),
        phase_references=phase_references,
        phase_identification_cache_path=phase_cache_path,
    )
    from analysis_layer.phase.update_local_mlip_hull_pool import update_local_mlip_hull_pool
    feedback["state"] = update_local_mlip_hull_pool(
        feedback["state"], config=effective_config,
        path=effective_config.get("branch_energy_pool_ledger_path"))
    feedback["state"]["budget_limits"] = deepcopy(effective_config.get("budgets") or {})
    feedback["state"]["branch_candidates"] = _branch_candidates_for_agent(manager)
    total_limit = (effective_config.get("budgets") or {}).get("total_relative_cost")
    if total_limit is not None:
        used = float((feedback["state"].get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
        reserved = float(feedback["state"].get("reserved_relative_cost", 0) or 0)
        feedback["state"]["budget_remaining"] = max(0.0, float(total_limit) - used - reserved)
    recovered_count = _count_newly_recovered(pre_reconciled["reconciled"])
    from execution_layer.remote.summarize_manual_upload_wait import summarize_manual_upload_wait
    manual_wait = (summarize_manual_upload_wait(feedback["state"], recovered_count=recovered_count)
                   if execution_mode == "interactive" else None)
    if manual_wait is not None and collection_report is not None:
        manual_wait["result_collection"] = deepcopy(collection_report)
    from execution_layer.local.rebuild_relax_inputs import is_relax_rebuild_request
    rebuilding = is_relax_rebuild_request(runtime_adapters.get("user_message")) or any(
        (((row.get("agent_proposal") or {}).get("raw_action") or {}).get("parameters") or {}).get("rebuild_inputs")
        for row in feedback["state"].get("pending_execution_policies", {}).values())
    recovery_question = feedback["state"].get("pending_dft_recovery_question")
    return {**frame,
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
    if frame.get("training_returns"):
        from execution_layer.local.recover_remote_training import training_result_message
        _save_runner_state(feedback["state"], runtime_path)
        return {"status": "training_results_received", "state": feedback["state"],
                "reason": "\n".join(training_result_message(r) for r in frame["training_returns"]),
                "steps_executed": 0, "submitted": False}
    if runtime_path:
        from execution_layer.state.execution_receipts import recovery_report
        report = recovery_report(runtime_path, feedback["state"])
        if report["unsettled"]:
            _save_runner_state(feedback["state"], runtime_path)
            return {"status": "execution_reconciliation_required", "state": feedback["state"],
                    "recovery_report": report, "steps_executed": 0, "submitted": False,
                    "recovered_count": recovered_count}
    if recovery_question and execution_mode == "interactive":
        _save_runner_state(feedback["state"], state_path or effective_config.get("state_path"))
        return {"status": "awaiting_dft_recovery_decision", "state": feedback["state"],
                "dft_recovery_question": recovery_question, "steps_executed": 0, "submitted": False,
                "scientific_feedback": {key: value for key, value in feedback.items() if key != "state"},
                "recovered_count": recovered_count, "result_collection": collection_report,
                "effective_config": effective_config, "config_version": snapshot["config_version"],
                "reconciled": pre_reconciled["reconciled"]}
    from execution_layer.workflows.model_refresh_state import refresh_active
    if manual_wait and not rebuilding and not refresh_active(feedback["state"]):
        wait_state = feedback["state"]
        _save_runner_state(wait_state, state_path or effective_config.get("state_path"))
        return {
            "status": "awaiting_manual_submission", "state": wait_state,
            "manual_wait": manual_wait, "recovered_count": recovered_count,
            "scientific_feedback": {key: value for key, value in feedback.items() if key != "state"},
            "reconciled": pre_reconciled["reconciled"], "steps_executed": 0,
            "submitted": False, "config_version": snapshot["config_version"],
            "effective_config": effective_config, "result_collection": collection_report,
        }
    return frame


def _assess_workflow_round(frame):
    feedback = frame["feedback"]
    effective_config = frame["effective_config"]
    manager = frame["manager"]
    runtime_adapters = frame["runtime_adapters"]
    state_path = frame["state_path"]
    from analysis_layer.state.post_dft_assessment import post_dft_assessment
    from execution_layer.workflows.refresh_dft_comparisons import refresh_dft_comparisons
    assessment = post_dft_assessment(feedback["state"], effective_config)
    feedback["state"] = refresh_dft_comparisons(
        feedback["state"], assessment=assessment, manager=manager,
        evaluator=_dft_comparison_evaluator(runtime_adapters, effective_config, feedback["state"]))
    # Persist analysis products before asking the LLM what to do next. The
    # proposal must follow the recovered-round analysis, not precede its export.
    from analysis_layer.feedback.export_dft_products import export_dft_products
    export_dft_products(feedback["state"], effective_config.get("phase_diagram_directory"))
    from analysis_layer.phase.combined_phase_diagram import refresh_combined_phase_diagram
    refresh_combined_phase_diagram(feedback["state"], effective_config.get("phase_diagram_directory"))
    _save_runner_state(feedback["state"], state_path or effective_config.get("state_path"))
    return {**frame,
            "feedback": feedback,
    }

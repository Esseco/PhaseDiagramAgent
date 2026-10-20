"""Observe existing returns under their bound configuration before any decision."""

from copy import deepcopy
from phase_agent.runtime.turn_process import timed_call
from phase_agent.interface_compatibility import call_with_optimizations
from pathlib import Path

from phase_agent.configuration.runtime.build_effective_run_config import (
    build_effective_run_config,
)
from phase_agent.configuration.session.phase_cache_location import phase_cache_location
from phase_agent.tools.remote.transfer_bundle import has_registered_return
from phase_agent.tools.state.task_waiting import awaiting_task_result
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import (
    apply_scientific_feedback,
)
from phase_agent.analysis.phase.update_local_mlip_hull_pool import (
    update_local_mlip_hull_pool,
)
from phase_agent.tools.workflows.lifecycle_support import (
    _save_runner_state,
    _dft_comparison_evaluator,
)


def recover_saved_results(state, **dependencies):
    """Collect and analyze evidence only; never migrate, propose or dispatch work."""
    collector = dependencies.get("task_runner") or dependencies.get("result_collector")
    manager = dependencies.get("manager")
    bound = state.get("confirmed_config")
    version = state.get("confirmed_config_version")
    if collector is None or manager is None or not bound or not version:
        return state
    processed = set(state.get("processed_task_ids") or [])
    feedback_processed = set(state.get("feedback_processed_task_ids") or [])
    backlog = [
        row
        for row in state.get("tasks") or []
        if row.get("task_id") in processed - feedback_processed
        and row.get("stage")
        in {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}
        and row.get("status") in {"completed", "failed", "timeout", "cancelled"}
    ]
    returned = has_registered_return(state) or any(
        awaiting_task_result(row)
        and isinstance(row.get("result_path"), str)
        and Path(row["result_path"]).is_file()
        for row in state.get("tasks") or []
    )
    if not returned and not backlog:
        return state
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": version,
            "config": deepcopy(bound),
        },
    }
    config = build_effective_run_config(session, dependencies.get("run_config"))
    if state.get("active_model"):
        config["mlip"] = {
            **(config.get("mlip") or {}),
            **deepcopy(state["active_model"]),
        }
    evaluator = _dft_comparison_evaluator(dependencies, config, state)
    if hasattr(collector, "collect_results_with_report"):
        collected = timed_call(
            "saved_return_collection",
            collector.collect_results_with_report,
            state,
            category="processing",
        )
        results, report = collected["results"], collected["report"]
    else:
        results = collector.collect_results(state)
        report = {"recovered_count": len(results)}
    incoming_ids = {row.get("task_id") for row in results}
    results = [
        *results,
        *(row for row in backlog if row.get("task_id") not in incoming_ids),
    ]
    current = deepcopy(state)
    current["result_collection"] = deepcopy(report)
    if not results:
        return _complete_saved_comparisons(current, evaluator, manager, config, dependencies)
    reconciled = call_with_optimizations(
        reconcile_task_results, current, results, _optimizations={"_owned": True}
    )
    state_path = dependencies.get("state_path") or config.get("state_path")
    # Persist verified completion and budget settlement before expensive phase
    # recognition. A restart resumes outstanding feedback from these records.
    _save_runner_state(reconciled["state"], state_path)
    feedback = timed_call(
        "saved_scientific_feedback",
        call_with_optimizations,
        apply_scientific_feedback,
        reconciled["state"],
        results,
        category="processing",
        manager=manager,
        ledger_path=config.get("ledger_path"),
        phase_diagram_directory=config.get("phase_diagram_directory"),
        final_frame_mlip_evaluator=evaluator,
        active_model_version=(config.get("mlip") or {}).get("version")
        or (config.get("mlip") or {}).get("name"),
        phase_references=dependencies.get("phase_references"),
        _optimizations={"_owned": True},
        phase_identification_cache_path=dependencies.get("phase_identification_cache_path")
        or (phase_cache_location(state_path) if state_path else None),
    )
    current = call_with_optimizations(
        update_local_mlip_hull_pool,
        feedback["state"],
        config=config,
        path=config.get("branch_energy_pool_ledger_path"),
        _optimizations={"_owned": True},
    )
    current["result_collection"]["feedback_rejected"] = feedback.get("rejected") or []
    current = _complete_saved_comparisons(current, evaluator, manager, config, dependencies)
    _save_runner_state(current, state_path)
    return current


def _complete_saved_comparisons(state, evaluator, manager, config, dependencies):
    """Repair verified saved pairs without recollecting labels or settling costs."""
    completed = {
        row.get("task_id")
        for row in state.get("dft_mlip_comparisons") or []
        if row.get("status") == "completed"
    }
    ids = [
        row["task_id"]
        for row in state.get("dft_dataset_records") or []
        if row.get("task_id") not in completed
        and row.get("status") == "completed"
        and row.get("converged") is True
        and row.get("checks_passed") is True
        and row.get("training_ready") is True
    ]
    if not ids:
        return state
    from phase_agent.tools.workflows.refresh_dft_comparisons import refresh_dft_comparisons

    updated = refresh_dft_comparisons(
        state, assessment=None, evaluator=evaluator, manager=manager, task_ids=ids
    )
    if updated.get("dft_mlip_comparisons") != state.get("dft_mlip_comparisons"):
        from phase_agent.analysis.feedback.dft_result_products import export_dft_products

        export_dft_products(updated, config.get("phase_diagram_directory"))
        _save_runner_state(updated, dependencies.get("state_path") or config.get("state_path"))
    return updated

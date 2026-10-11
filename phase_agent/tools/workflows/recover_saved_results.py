"""Observe existing returns under their bound configuration before any decision."""

from copy import deepcopy
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
from phase_agent.tools.workflows.lifecycle_support import _save_runner_state


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
    if hasattr(collector, "collect_results_with_report"):
        collected = collector.collect_results_with_report(state)
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
        return current
    reconciled = reconcile_task_results(current, results)
    state_path = dependencies.get("state_path") or config.get("state_path")
    # Persist verified completion and budget settlement before expensive phase
    # recognition. A restart resumes outstanding feedback from these records.
    _save_runner_state(reconciled["state"], state_path)
    feedback = apply_scientific_feedback(
        reconciled["state"],
        results,
        manager=manager,
        ledger_path=config.get("ledger_path"),
        phase_diagram_directory=config.get("phase_diagram_directory"),
        active_model_version=(config.get("mlip") or {}).get("version")
        or (config.get("mlip") or {}).get("name"),
        phase_references=dependencies.get("phase_references"),
        phase_identification_cache_path=dependencies.get(
            "phase_identification_cache_path"
        )
        or (phase_cache_location(state_path) if state_path else None),
    )
    current = update_local_mlip_hull_pool(
        feedback["state"],
        config=config,
        path=config.get("branch_energy_pool_ledger_path"),
    )
    current["result_collection"]["feedback_rejected"] = feedback.get("rejected") or []
    _save_runner_state(current, state_path)
    return current

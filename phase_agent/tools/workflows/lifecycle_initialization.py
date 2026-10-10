"""Confirmed configuration and runtime dependency initialization."""

from copy import deepcopy
from phase_agent.configuration.runtime.build_effective_run_config import build_effective_run_config
from phase_agent.configuration.runtime.authorize_budget_extension import authorize_budget_extension
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.workflows.create_workflow_handlers import create_workflow_handlers
from phase_agent.tools.workflows.compact_action_history import compact_state_history
from phase_agent.tools.workflows.lifecycle_support import (
    _read_state_for_runner,
    _save_runner_state,
    _config_migration_block_message,
)


def _initialize_workflow(
    manager,
    phase_references,
    run_config,
    config_session,
    *,
    state=None,
    handlers=None,
    registry=None,
    agent_client=None,
    execution_mode="interactive",
    human_feedback=None,
    replay_record=None,
    recovered_results=None,
    max_steps=1,
    invocation_id=None,
    state_path=None,
    approval_directory=None,
    initial_long_term_advice=None,
    dispatcher=None,
    task_runner=None,
    result_collector=None,
    approve_budget_extension=False,
    **runtime_adapters,
):
    """Build the effective config and run bounded validated Agent actions."""
    snapshot = config_session.get("confirmed_snapshot") or {}
    if config_session.get("status") != "confirmed" or not snapshot:
        return {"status": "rejected", "reason": "configuration_not_confirmed", "submitted": False}
    effective_config = build_effective_run_config(config_session, run_config)
    effective_handlers = create_workflow_handlers(
        handlers,
        stage_registry=runtime_adapters.get("stage_registry"),
        candidates_provider=runtime_adapters.get("dft_candidates_provider"),
        qbc_evaluator=runtime_adapters.get("qbc_evaluator"),
    )
    effective_registry = registry or create_tool_registry(effective_handlers)
    context = {
        "manager": manager,
        "phase_references": phase_references,
        "effective_config": deepcopy(effective_config),
        "dispatcher": dispatcher,
        "agent_client": agent_client,
        **runtime_adapters,
    }
    automatic_results = []
    collection_report = None
    loaded_state = compact_state_history(
        _read_state_for_runner(state, state_path or effective_config.get("state_path"))
    )
    migration = authorize_budget_extension(
        loaded_state, snapshot, user_approved=approve_budget_extension
    )
    if migration["status"].startswith("rejected") or migration["status"] == "approval_required":
        return {
            "status": "rejected",
            "reason": migration["status"],
            "submitted": False,
            "config_version": snapshot["config_version"],
            "state": loaded_state,
            "migration": migration,
            "message": _config_migration_block_message(migration),
        }
    loaded_state = migration["state"]
    if loaded_state.get("active_model"):
        effective_config["mlip"] = {
            **(effective_config.get("mlip") or {}),
            **deepcopy(loaded_state["active_model"]),
        }
        context["effective_config"] = deepcopy(effective_config)
    if approve_budget_extension:
        if migration["status"] in {"extended", "empty_run_rebound"}:
            limits = deepcopy(effective_config.get("budgets") or {})
            loaded_state["budget_limits"] = limits
            total_limit = limits.get("total_relative_cost")
            if total_limit is not None:
                used = float(
                    (loaded_state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0
                )
                reserved = float(loaded_state.get("reserved_relative_cost", 0) or 0)
                loaded_state["budget_remaining"] = max(0.0, float(total_limit) - used - reserved)
            _save_runner_state(loaded_state, state_path or effective_config.get("state_path"))
            return {
                "status": "config_migrated",
                "state": loaded_state,
                "migration": {key: value for key, value in migration.items() if key != "state"},
                "from_config_version": migration.get("from"),
                "config_version": snapshot["config_version"],
                "submitted": False,
            }
        if migration["status"] == "unchanged":
            return {
                "status": "config_migration_not_needed",
                "state": loaded_state,
                "config_version": snapshot["config_version"],
                "submitted": False,
            }
    return {
        "_workflow_prepared": True,
        "manager": manager,
        "phase_references": phase_references,
        "run_config": run_config,
        "config_session": config_session,
        "state": state,
        "handlers": handlers,
        "registry": registry,
        "agent_client": agent_client,
        "execution_mode": execution_mode,
        "human_feedback": human_feedback,
        "replay_record": replay_record,
        "recovered_results": recovered_results,
        "max_steps": max_steps,
        "invocation_id": invocation_id,
        "state_path": state_path,
        "approval_directory": approval_directory,
        "initial_long_term_advice": initial_long_term_advice,
        "dispatcher": dispatcher,
        "task_runner": task_runner,
        "result_collector": result_collector,
        "approve_budget_extension": approve_budget_extension,
        "runtime_adapters": runtime_adapters,
        "snapshot": snapshot,
        "effective_config": effective_config,
        "effective_handlers": effective_handlers,
        "effective_registry": effective_registry,
        "context": context,
        "automatic_results": automatic_results,
        "collection_report": collection_report,
        "loaded_state": loaded_state,
    }

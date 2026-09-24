"""Create the backend-independent handlers available in every workflow."""

from analysis_layer.convergence.check_global_convergence import check_global_convergence
from execution_layer.workflows.create_active_learning_handlers import create_active_learning_handlers
from execution_layer.workflows.create_dft_selection_handler import create_dft_selection_handler
from execution_layer.dispatch.create_default_stage_registry import create_default_stage_registry
from execution_layer.state.restart_failed_task import restart_failed_task
from execution_layer.local.prepare_local_batch_files import prepare_local_batch_files


def create_workflow_handlers(
    extra_handlers=None, *, stage_registry=None, candidates_provider=None, qbc_evaluator=None
):
    handlers = {
        "check_convergence": _check_convergence,
        "pause_search": _pause_search,
        "restart_failed_task": _restart_failed_task,
        "prepare_local_batch_files": prepare_local_batch_files,
    }
    handlers.update(create_active_learning_handlers(stage_registry or create_default_stage_registry()))
    handlers["select_dft_candidates"] = create_dft_selection_handler(
        candidates_provider=candidates_provider, qbc_evaluator=qbc_evaluator
    )
    handlers.update(extra_handlers or {})
    return handlers


def _check_convergence(*, action, context):
    state = context.get("event_state") or {}
    rules = (context.get("effective_config") or {}).get("convergence") or {}
    return check_global_convergence(state, rules=_normalize_rules(rules))


def _pause_search(*, action, context):
    return {"status": "paused", "reason": action.get("reason") or "agent_requested_pause"}


def _restart_failed_task(*, action, context):
    target = (action.get("target_ids") or [None])[0]
    config = context.get("effective_config") or {}
    return restart_failed_task(
        context.get("event_state") or {}, target, budget_limits=config.get("budgets") or {},
        config_version=context.get("config_version"),
        max_retries=int((action.get("parameters") or {}).get("max_retries", 1)),
    )


def _normalize_rules(rules):
    aliases = {
        "hull_tolerance": "hull_change_tolerance",
        "stable_rounds": "stable_iterations",
        "coverage_threshold": "minimum_coverage_fraction",
        "dft_correction_tolerance": "final_frame_error_tolerance",
    }
    return {aliases.get(key, key): value for key, value in rules.items() if value is not None}

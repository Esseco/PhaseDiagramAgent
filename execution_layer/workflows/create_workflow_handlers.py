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
    from execution_layer.workflows.request_strategy_revision import request_strategy_revision
    handlers = {
        "adjust_strategy": request_strategy_revision,
        "check_convergence": _check_convergence,
        "pause_search": _pause_search,
        "restart_failed_task": _restart_failed_task,
        "prepare_local_batch_files": prepare_local_batch_files,
        "update_mlip": _update_mlip,
    }
    handlers.update(create_active_learning_handlers(stage_registry or create_default_stage_registry()))
    handlers["select_dft_candidates"] = create_dft_selection_handler(
        candidates_provider=candidates_provider, qbc_evaluator=qbc_evaluator
    )
    handlers.update(extra_handlers or {})
    return handlers


def _update_mlip(*, action, context):
    """Expose the existing model-update pipeline without bypassing its gates."""
    config = context.get("effective_config") or {}
    state = context.get("event_state") or {}
    parameters = action.get("parameters") or {}
    trigger = {**parameters, "action": parameters.get("action", "RETRAIN_MLIP")}
    input_only = parameters.get("prepare_inputs_only") is True and trigger["action"] == "RETRAIN_MLIP"
    if trigger["action"] == "RETRAIN_MLIP":
        if not input_only and (config.get("mlip_finetune") or {}).get("enabled") is not True:
            return {"status": "not_configured", "reason": "微调未启用；请先修订并确认配置。", "state": state}
        from analysis_layer.state.post_dft_assessment import post_dft_assessment
        assessment = post_dft_assessment(state, config)
        if assessment and assessment["status"] != "evaluated":
            return {"status": "not_configured", "reason": "本轮原模型误差评估尚未完成；先补齐模型与评估。", "state": state}
        from scientific_layer.dft.spin_acceptance import spin_standard_passed
        minimum = int(((config.get("mlip_finetune") or {}).get("training") or {}).get(
            "minimum_new_dft_records", ((config.get("qbc") or {}).get("retrain") or {}).get("minimum_new_dft_records", 10)))
        valid = {row.get("data_id") or row.get("task_id") for row in state.get("new_dft_records") or []
                 if row.get("status") == "completed" and row.get("converged") is True
                 and row.get("checks_passed") is True and row.get("energy") is not None
                 and spin_standard_passed(row)}
        if len(valid) < minimum:
            return {"status": "insufficient_new_data", "reason": f"新增合格数据 {len(valid)}/{minimum}；未训练。", "state": state}
    handler = context.get("model_update_handler")
    if input_only or (trigger["action"] == "RETRAIN_MLIP" and not callable(handler)
            and not callable(context.get("mlip_trainer"))):
        from execution_layer.workflows.prepare_remote_finetune import prepare_remote_finetune
        return prepare_remote_finetune(state, config)
    if not callable(handler):
        from execution_layer.workflows.create_model_update_handler import create_model_update_handler
        handler = create_model_update_handler(
            trainer=context.get("mlip_trainer"), validation_evaluator=context.get("mlip_validation_evaluator"),
            reevaluation_predictor=context.get("mlip_reevaluation_predictor"),
            historical_data_provider=context.get("historical_data_provider"),
            validation_data_provider=context.get("validation_data_provider"),
            candidate_provider=context.get("model_candidate_provider"))
    return handler(trigger=trigger, state=state, manager=context.get("manager"), config=config)


def _check_convergence(*, action, context):
    state = context.get("event_state") or {}
    if (state.get("model_refresh") or {}).get("partial"):
        return {"status": "not_converged", "reason": "partial_model_refresh_not_global_convergence_evidence"}
    rules = (context.get("effective_config") or {}).get("convergence") or {}
    result = check_global_convergence(state, rules=_normalize_rules(rules))
    if result.get("status") == "finished":
        from data_layer.memory.maybe_draft_converged_skill import maybe_draft_converged_skill
        result["knowledge_draft"] = maybe_draft_converged_skill(
            state, context.get("effective_config") or {}, result)
    return result


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

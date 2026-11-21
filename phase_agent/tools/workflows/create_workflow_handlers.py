"""Create the backend-independent handlers available in every workflow."""

from phase_agent.analysis.convergence.check_global_convergence import check_global_convergence
from phase_agent.tools.workflows.create_active_learning_handlers import (
    create_active_learning_handlers,
)
from phase_agent.tools.workflows.create_dft_selection_handler import create_dft_selection_handler
from phase_agent.tools.dispatch.create_default_stage_registry import create_default_stage_registry
from phase_agent.tools.state.restart_failed_task import restart_failed_task
from phase_agent.tools.local.prepare_local_batch_files import prepare_local_batch_files


def create_workflow_handlers(
    extra_handlers=None, *, stage_registry=None, candidates_provider=None, qbc_evaluator=None
):
    from phase_agent.tools.workflows.request_strategy_revision import request_strategy_revision

    handlers = {
        "adjust_strategy": request_strategy_revision,
        "check_convergence": _check_convergence,
        "pause_search": _pause_search,
        "restart_failed_task": _restart_failed_task,
        "prepare_local_batch_files": prepare_local_batch_files,
        "update_mlip": _update_mlip,
    }
    handlers.update(
        create_active_learning_handlers(stage_registry or create_default_stage_registry())
    )
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
    input_only = (
        parameters.get("prepare_inputs_only") is True and trigger["action"] == "RETRAIN_MLIP"
    )
    if trigger["action"] == "RETRAIN_MLIP":
        if not input_only and (config.get("mlip_finetune") or {}).get("enabled") is not True:
            return {
                "status": "not_configured",
                "reason": "微调未启用；请先修订并确认配置。",
                "state": state,
            }
        from phase_agent.evidence.post_dft_assessment import post_dft_assessment

        assessment = post_dft_assessment(state, config)
        if assessment and assessment["status"] != "evaluated":
            return {
                "status": "not_configured",
                "reason": "本轮原模型误差评估尚未完成；先补齐模型与评估。",
                "state": state,
            }
        from phase_agent.policies.workflow_policy import settings, new_dft_ids

        minimum = settings(config)["finetune_min_new_dft"]
        from copy import deepcopy

        config = deepcopy(config)
        config.setdefault("mlip_finetune", {}).setdefault("training", {})[
            "minimum_new_dft_records"
        ] = minimum
        valid = new_dft_ids(state, config)
        from phase_agent.tools.workflows.manual_training_coexistence import blocking_tasks

        if blocking_tasks(state, action):
            return {
                "status": "not_configured",
                "state": state,
                "reason": "fine_tune_waits_for_inflight_tasks",
            }
        if len(valid) < minimum:
            return {
                "status": "insufficient_new_data",
                "reason": f"新增合格数据 {len(valid)}/{minimum}；未训练。",
                "state": state,
            }
    handler = context.get("model_update_handler")
    if input_only or (
        trigger["action"] == "RETRAIN_MLIP"
        and not callable(handler)
        and not callable(context.get("mlip_trainer"))
    ):
        from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune

        return prepare_remote_finetune(
            state,
            config,
            training_action=action,
            transition_policy=parameters.get("post_training_transition"),
            transition_approval_reason=(
                parameters.get("post_training_transition_reason") or action.get("reason")
            ),
        )
    if not callable(handler):
        from phase_agent.tools.workflows.create_model_update_handler import (
            create_model_update_handler,
        )

        handler = create_model_update_handler(
            trainer=context.get("mlip_trainer"),
            validation_evaluator=context.get("mlip_validation_evaluator"),
            reevaluation_predictor=context.get("mlip_reevaluation_predictor"),
            historical_data_provider=context.get("historical_data_provider"),
            validation_data_provider=context.get("validation_data_provider"),
            candidate_provider=context.get("model_candidate_provider"),
        )
    return handler(trigger=trigger, state=state, manager=context.get("manager"), config=config)


def _check_convergence(*, action, context):
    state = context.get("event_state") or {}
    if (state.get("model_refresh") or {}).get("partial"):
        return {
            "status": "not_converged",
            "reason": "partial_model_refresh_not_global_convergence_evidence",
        }
    rules = (context.get("effective_config") or {}).get("convergence") or {}
    rules = {
        **rules,
        "stable_feedback_cycles": rules.get("stable_feedback_cycles", 1),
        "minimum_ground_state_dft_fraction": 0.6,
        "final_energy_mae_tolerance": 0.005,
        "final_force_rmse_tolerance": 0.05,
    }
    result = check_global_convergence(state, rules=_normalize_rules(rules))
    if result.get("status") == "finished":
        from phase_agent.persistence.memory.maybe_draft_converged_skill import (
            maybe_draft_converged_skill,
        )

        result["knowledge_draft"] = maybe_draft_converged_skill(
            state, context.get("effective_config") or {}, result
        )
    return result


def _pause_search(*, action, context):
    return {"status": "paused", "reason": action.get("reason") or "agent_requested_pause"}


def _restart_failed_task(*, action, context):
    targets = action.get("target_ids") or [None]
    config = context.get("effective_config") or {}
    parameters = action.get("parameters") or {}
    profile = parameters.get("repair_profile")
    if profile not in {
        None,
        "vasp_symmetry_single_gpu",
        "mixing_parameters",
        "dft_scf_diagnostic",
        "dft_recalculation",
        "mc_threshold_recovery",
    }:
        raise ValueError("unknown_retry_repair_profile")
    from phase_agent.tools.workflows.dft_failure_retry import (
        apply_retry_settings,
        mixing_retry_settings,
        scf_diagnostic_retry_settings,
        symmetry_retry_settings,
    )

    original_state = context.get("event_state") or {}
    state = original_state
    if profile == "dft_recalculation":
        from copy import deepcopy
        from phase_agent.agents.simulation.dft_recalculation import validate_action

        validate_action(action, original_state)
        state = deepcopy(original_state)
    if profile == "dft_scf_diagnostic":
        from copy import deepcopy
        strategies = parameters.get("strategies")
        if (not isinstance(strategies, dict) or set(strategies) != set(targets)
                or any(x not in {"nelm_200", "nelm_200_slow_mixing"} for x in strategies.values())
                or len(targets) != len(set(targets)) or not targets):
            raise ValueError("invalid_scf_diagnostic_scope")
        taskmap = {t.get("task_id"): t for t in original_state.get("tasks") or []}
        if any(t not in taskmap or taskmap[t].get("stage") != "dft_single_point"
               or taskmap[t].get("status") not in {"failed", "timeout"} for t in targets):
            raise ValueError("invalid_scf_diagnostic_target")
        state = deepcopy(original_state)
    if profile == "mc_threshold_recovery":
        from copy import deepcopy
        from phase_agent.tools.state.mc_threshold_retry import validate_mc_threshold_retry

        if parameters.get("prepare_inputs_after_approval"):
            raise ValueError("mc_retry_input_preparation_requires_separate_action")
        validate_mc_threshold_retry(original_state, targets)
        state = deepcopy(original_state)
    limits = config.get("budgets") or {}
    if profile == "mc_threshold_recovery":
        from phase_agent.tools.budget.check_budget import budget_is_hard

        if not budget_is_hard(limits):
            limits = deepcopy(limits)
            (limits.get("stage_limits", {}).get("deep_search") or {}).pop("max_tasks", None)
    prepared = []
    task_by_id = {t.get("task_id"): t for t in state.get("tasks") or []}
    for target in targets:
        original = task_by_id[target]
        strategy = parameters["strategies"][target] if profile == "dft_recalculation" else None
        remediation = (
            scf_diagnostic_retry_settings(original, strategy)
            if strategy in {"nelm_200", "nelm_200_slow_mixing"}
            else mixing_retry_settings(original)
            if strategy == "mixing_parameters"
            else symmetry_retry_settings(original)
            if strategy == "vasp_symmetry_single_gpu"
            else scf_diagnostic_retry_settings(original, parameters["strategies"][target])
            if profile == "dft_scf_diagnostic"
            else mixing_retry_settings(original)
            if profile == "mixing_parameters"
            else symmetry_retry_settings(original)
            if profile == "vasp_symmetry_single_gpu"
            else None
        )
        options = {"_owned": True} if profile in {"mc_threshold_recovery", "dft_scf_diagnostic", "dft_recalculation"} else {}
        result = restart_failed_task(
            state,
            target,
            budget_limits=limits,
            config_version=context.get("config_version"),
            max_retries=int(parameters.get("max_retries", 1)),
            **options,
        )
        if result.get("status") != "prepared":
            return (
                {**result, "state": original_state}
                if profile in {"mc_threshold_recovery", "dft_scf_diagnostic", "dft_recalculation"}
                else result
            )
        state = result["state"]
        if remediation is not None:
            apply_retry_settings(result["task"], *remediation, profile=profile)
        if profile == "dft_recalculation":
            result["task"]["retry_strategy"] = strategy
        prepared.append(result["task"])
    result = {"status": "prepared", "state": state, "tasks": prepared}
    if not parameters.get("prepare_inputs_after_approval"):
        return result
    from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches

    return {
        **prepare_dft_upload_batches(
            action={"target_ids": [t["task_id"] for t in prepared]},
            context={**context, "event_state": state},
        ),
        "retry_task_ids": [t["task_id"] for t in prepared],
        "retry_of": list(targets),
    }


def _normalize_rules(rules):
    aliases = {
        "hull_tolerance": "hull_change_tolerance",
        "stable_rounds": "stable_iterations",
        "coverage_threshold": "minimum_coverage_fraction",
        "dft_correction_tolerance": "final_frame_error_tolerance",
    }
    return {aliases.get(key, key): value for key, value in rules.items() if value is not None}

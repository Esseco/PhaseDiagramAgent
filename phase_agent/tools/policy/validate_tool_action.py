"""Enforce confirmed config, frozen parameters, budget, state and ownership."""

from phase_agent.tools.budget.check_budget import check_budget, budget_is_hard
from phase_agent.science.bohb.check_bohb_task_ownership import check_bohb_task_ownership
from phase_agent.configuration.schema.validate_strategy_adjustment import (
    validate_strategy_adjustment,
)
from phase_agent.configuration.schema.action_state_schema import (
    normalize_action,
    validate_action_schema,
)


def validate_tool_action(action: dict, state: dict, session: dict, registry: dict) -> dict:
    try:
        action = normalize_action(action)
    except (ValueError, TypeError) as error:
        return {"valid": False, "errors": [str(error)]}
    errors = []
    errors.extend(validate_action_schema(action))
    snapshot = session.get("confirmed_snapshot") if session.get("status") == "confirmed" else None
    if snapshot is None:
        errors.append("configuration_not_confirmed")
        return {"valid": False, "errors": errors}
    config = snapshot["config"]
    from phase_agent.tools.dispatch.scientific_operations import (
        operation_binding_errors,
    )

    model = state.get("active_model") or config.get("mlip") or {}
    errors.extend(
        operation_binding_errors(
            action,
            config_version=snapshot["config_version"],
            model_version=state.get("active_model_version")
            or model.get("version")
            or model.get("name"),
        )
    )
    if action.get("scientific_decision") is not None:
        from phase_agent.decisions.agent.scientific_decisions import (
            scientific_decision_errors,
        )

        errors.extend(
            scientific_decision_errors(
                action,
                {
                    "config_version": snapshot["config_version"],
                    "model_version": state.get("active_model_version")
                    or model.get("version")
                    or model.get("name"),
                },
                {"post_dft_assessment": action.get("post_dft_review")},
            )
        )
    tool = action.get("tool")
    bound_version = state.get("confirmed_config_version")
    if bound_version is not None and bound_version != snapshot["config_version"]:
        errors.append("config_version_mismatch")
    if tool not in registry or tool not in set(
        (config.get("agent") or {}).get("allowed_tools") or []
    ):
        errors.append("tool_not_allowed")
    elif not callable(registry[tool].get("handler")):
        errors.append("tool_handler_not_configured")
    if any(key in action for key in ("energy", "ehull", "score", "converged")):
        errors.append("agent_supplied_numeric_result")
    parameters = action.get("parameters") or {}
    requested_dft_tasks = 0
    if tool == "select_dft_candidates" and isinstance(parameters.get("decisions"), list):
        requested_dft_tasks = sum(
            isinstance(row, dict) and row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
            for row in parameters["decisions"]
        )
    elif tool == "run_calculation_stage" and (action.get("stage") or parameters.get("stage")) in {
        "dft_single_point",
        "dft_relax",
    }:
        requested_dft_tasks = 1
    if requested_dft_tasks:
        from phase_agent.decisions.qbc_selection.dft_round_policy import (
            dft_task_limit,
            round_dft_tasks,
        )

        try:
            if len(round_dft_tasks(state, config)) + requested_dft_tasks > dft_task_limit(config):
                errors.append("dft_round_task_limit")
        except ValueError as error:
            errors.append(f"invalid_dft_round_task_limit:{error}")
    from phase_agent.tools.policy.branch_conditions import branch_entry_errors

    errors.extend(branch_entry_errors(action, state))
    from phase_agent.tools.policy.adaptive_branch_count import generation_review_errors

    errors.extend(generation_review_errors(action, state))
    if tool == "generate_branches":
        from phase_agent.tools.policy.region_sampling import is_region, CONTROLS

        if is_region(action):
            if parameters.get("sampling_controls") != CONTROLS:
                errors.append("region_sampling_controls_not_frozen")
            preview = parameters.get("region_preview") or {}
            if not preview.get("ledger_fingerprint"):
                errors.append("region_sampling_requires_metadata_preview")
            if (
                preview.get("model_version") != state.get("active_model_version")
                or preview.get("config_version") != snapshot["config_version"]
            ):
                errors.append("region_sampling_preview_version_changed")
        from phase_agent.decisions.agent.generation_plan import (
            configured_generation_strategies,
            disabled_generation_allocations,
            validate_generation_plan,
        )

        if "generation_plan" in parameters:
            try:
                validate_generation_plan(parameters)
            except ValueError as error:
                errors.append(f"invalid_generation_plan:{error}")
        disabled = disabled_generation_allocations(
            parameters, configured_generation_strategies(config)
        )
        if disabled:
            errors.append(f"disabled_generation_strategies:{','.join(disabled)}")
    from phase_agent.tools.policy.workflow_policy import action_errors

    errors.extend(action_errors(action, state, config))
    supplied_paths = _flatten_paths(parameters)
    for path in config.get("frozen_parameters") or []:
        if any(
            item == path or item.startswith(path + ".") or path.endswith("." + item)
            for item in supplied_paths
        ):
            errors.append(f"frozen_parameter_override:{path}")
    if tool == "run_calculation_stage":
        stage = action.get("stage") or parameters.get("stage")
        if len(action.get("target_ids") or []) != 1:
            errors.append("calculation_stage_requires_one_structure_target")
        if stage not in set((config.get("calculation") or {}).get("enabled_stages") or []):
            errors.append("calculation_stage_not_enabled")
        if stage in {"dft_single_point", "dft_relax"}:
            confirmed_dft = (config.get("dft") or {}).get("parameters") or {}
            if any(
                key not in confirmed_dft or confirmed_dft[key] != value
                for key, value in parameters.items()
            ):
                errors.append("dft_parameter_override")
    if tool == "restart_failed_task":
        targets = action.get("target_ids") or []
        retryable = {
            row.get("task_id")
            for row in state.get("tasks") or []
            if row.get("task_id") and row.get("status") in {"failed", "timeout"}
        }
        if len(targets) != 1 or targets[0] not in retryable:
            errors.append("retry_target_not_failed_task")
    task_key = action.get("task_key")
    if tool not in {"check_convergence", "pause_search"} and not task_key:
        errors.append("formal_tool_requires_task_key")
    effective = state.get("effective_decisions") or {}
    if (
        task_key
        and task_key in effective
        and effective[task_key].get("status")
        in {"reserved", "pending", "running", "completed", "reconciliation_required"}
    ):
        errors.append("duplicate_effective_decision")
    budget = action.get("budget", 0.0)
    if not isinstance(budget, (int, float)) or budget < 0:
        errors.append("invalid_budget")
    else:
        from phase_agent.tools.policy.adaptive_branch_count import (
            budget_limits_for_action,
        )

        branch_limits = budget_limits_for_action(action, config["budgets"])
        check = check_budget(
            state,
            {"stage": action.get("stage"), "tasks": 1, "relative_cost": budget},
            branch_limits,
        )
        errors.extend(check["reasons"])
        total_limit = (config.get("budgets") or {}).get("total_relative_cost")
        used = ((state.get("budget_usage") or {}).get("total_relative_cost") or 0) + (
            state.get("reserved_relative_cost") or 0
        )
        if (
            budget_is_hard(branch_limits)
            and total_limit is not None
            and used + budget > total_limit
        ):
            errors.append("reserved_total_cost_limit")
    if tool == "select_dft_candidates" and registry.get(tool, {}).get("owner") != "decision_layer":
        errors.append("dft_selection_owner_invalid")
    if tool == "allocate_mc_bohb":
        forbidden = {
            "branch_id",
            "branch_ids",
            "selected_branches",
            "branch_budgets",
            "budget_level",
            "promotion",
        } & set(parameters)
        if forbidden:
            errors.append("bohb_owned_fields_forbidden:" + ",".join(sorted(forbidden)))
        mc_budget = parameters.get("mc_budget")
        if isinstance(mc_budget, bool) or not isinstance(mc_budget, int) or mc_budget <= 0:
            errors.append("invalid_mc_budget")
        maximum_mc_budget = (config.get("round_strategy") or {}).get("maximum_mc_budget")
        if (
            isinstance(mc_budget, int)
            and not isinstance(mc_budget, bool)
            and maximum_mc_budget is not None
            and mc_budget > int(maximum_mc_budget)
            and budget_is_hard(config.get("budgets"))
        ):
            errors.append("mc_budget_exceeds_configured_maximum")
        exploration = parameters.get("exploration_fraction")
        minimum = (config.get("round_strategy") or {}).get("minimum_exploration_fraction", 0.1)
        if (
            not isinstance(exploration, (int, float))
            or isinstance(exploration, bool)
            or not minimum <= exploration <= 1
        ):
            errors.append("invalid_exploration_fraction")
    ownership = check_bohb_task_ownership(state.get("active_round"), action)
    if not ownership["allowed"]:
        errors.append(ownership["reason"])
    if tool == "adjust_strategy":
        if parameters.get("request_configuration_revision") is True:
            from phase_agent.tools.workflows.request_strategy_revision import (
                _known_path,
            )

            patch = parameters.get("patch") or {}
            if (
                not isinstance(patch, dict)
                or not patch
                or any(not _known_path(config, key) for key in patch)
            ):
                errors.append("invalid_configuration_revision_request")
        else:
            adjustment = validate_strategy_adjustment(
                parameters.get("patch") or {}, config, bounds=parameters.get("bounds")
            )
            if adjustment["requires_user_confirmation"]:
                errors.append("hard_constraint_change_requires_confirmation")
            errors.extend(f"strategy_adjustment_rejected:{item}" for item in adjustment["rejected"])
    return {
        "valid": not errors,
        "errors": errors,
        "config_version": snapshot["config_version"],
        "budget_warnings": check.get("warnings", [])
        if isinstance(budget, (int, float)) and budget >= 0
        else [],
    }


def _flatten_paths(value, prefix=""):
    paths = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_flatten_paths(child, path))
    return paths

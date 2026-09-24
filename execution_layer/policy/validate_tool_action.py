"""Enforce confirmed config, frozen parameters, budget, state and ownership."""

from execution_layer.budget.check_budget import check_budget
from scientific_layer.bohb.check_bohb_task_ownership import check_bohb_task_ownership
from config_layer.schema.validate_strategy_adjustment import validate_strategy_adjustment
from config_layer.schema.action_state_schema import normalize_action, validate_action_schema


def validate_tool_action(action: dict, state: dict, session: dict, registry: dict) -> dict:
    action = normalize_action(action)
    errors = []
    errors.extend(validate_action_schema(action))
    snapshot = session.get("confirmed_snapshot") if session.get("status") == "confirmed" else None
    if snapshot is None:
        errors.append("configuration_not_confirmed")
        return {"valid": False, "errors": errors}
    config = snapshot["config"]; tool = action.get("tool")
    bound_version = state.get("confirmed_config_version")
    if bound_version is not None and bound_version != snapshot["config_version"]:
        errors.append("config_version_mismatch")
    if tool not in registry or tool not in set((config.get("agent") or {}).get("allowed_tools") or []):
        errors.append("tool_not_allowed")
    if any(key in action for key in ("energy", "ehull", "score", "converged")):
        errors.append("agent_supplied_numeric_result")
    parameters = action.get("parameters") or {}
    supplied_paths = _flatten_paths(parameters)
    for path in config.get("frozen_parameters") or []:
        if any(item == path or item.startswith(path + ".") or path.endswith("." + item) for item in supplied_paths):
            errors.append(f"frozen_parameter_override:{path}")
    if tool == "run_calculation_stage":
        stage = action.get("stage")
        if stage not in set((config.get("calculation") or {}).get("enabled_stages") or []):
            errors.append("calculation_stage_not_enabled")
        if stage in {"dft_single_point", "dft_relax"}:
            confirmed_dft = (config.get("dft") or {}).get("parameters") or {}
            if any(key not in confirmed_dft or confirmed_dft[key] != value for key, value in parameters.items()):
                errors.append("dft_parameter_override")
    task_key = action.get("task_key")
    if tool not in {"check_convergence", "pause_search"} and not task_key:
        errors.append("formal_tool_requires_task_key")
    effective = state.get("effective_decisions") or {}
    if task_key and task_key in effective and effective[task_key].get("status") in {"reserved", "pending", "running", "completed"}:
        errors.append("duplicate_effective_decision")
    budget = action.get("budget", 0.0)
    if not isinstance(budget, (int, float)) or budget < 0:
        errors.append("invalid_budget")
    else:
        check = check_budget(state, {"stage": action.get("stage"), "tasks": 1, "relative_cost": budget}, config["budgets"])
        errors.extend(check["reasons"])
        total_limit = (config.get("budgets") or {}).get("total_relative_cost")
        used = ((state.get("budget_usage") or {}).get("total_relative_cost") or 0) + (state.get("reserved_relative_cost") or 0)
        if total_limit is not None and used + budget > total_limit:
            errors.append("reserved_total_cost_limit")
    if tool == "select_dft_candidates" and registry.get(tool, {}).get("owner") != "decision_layer":
        errors.append("dft_selection_owner_invalid")
    if tool == "allocate_mc_bohb":
        forbidden = {"branch_id", "branch_ids", "selected_branches", "branch_budgets", "budget_level", "promotion"} & set(parameters)
        if forbidden:
            errors.append("bohb_owned_fields_forbidden:" + ",".join(sorted(forbidden)))
        mc_budget = parameters.get("mc_budget")
        if isinstance(mc_budget, bool) or not isinstance(mc_budget, int) or mc_budget <= 0:
            errors.append("invalid_mc_budget")
        exploration = parameters.get("exploration_fraction")
        minimum = ((config.get("round_strategy") or {}).get("minimum_exploration_fraction", .1))
        if not isinstance(exploration, (int, float)) or isinstance(exploration, bool) or not minimum <= exploration <= 1:
            errors.append("invalid_exploration_fraction")
    ownership = check_bohb_task_ownership(state.get("active_round"), action)
    if not ownership["allowed"]:
        errors.append(ownership["reason"])
    if tool == "adjust_strategy":
        adjustment = validate_strategy_adjustment(parameters.get("patch") or {}, config, bounds=parameters.get("bounds"))
        if adjustment["requires_user_confirmation"]:
            errors.append("hard_constraint_change_requires_confirmation")
        errors.extend(f"strategy_adjustment_rejected:{item}" for item in adjustment["rejected"])
    return {"valid": not errors, "errors": errors, "config_version": snapshot["config_version"]}


def _flatten_paths(value, prefix=""):
    paths = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key); paths.add(path); paths.update(_flatten_paths(child, path))
    return paths

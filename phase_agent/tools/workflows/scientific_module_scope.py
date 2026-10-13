"""Narrow branch-to-package scope; no new strategy, model switch or HPC submission."""

from copy import deepcopy
import math


def attach_module_scope(action, state, config):
    if (state.get("scientific_decision_policy") or {}).get("schema_version") != 1:
        return action
    if action.get("tool") != "generate_branches":
        return action
    preference = (action.get("scientific_decision") or {}).get("module_execution")
    if (
        preference or (state.get("scientific_decision_policy") or {}).get("mlip_execution")
    ) == "stepwise":
        return action
    params = action.get("parameters") or {}
    preview = params.get("generation_cost_preview") or {}
    amount = preview.get("relax_mc_cost_upper")
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
        return action
    # Old preflight includes one segment. Three times it is a conservative
    # envelope; exact diagnosis still has to fit, or preparation is blocked.
    params["module_execution_scope"] = {
        "schema_version": 1,
        "operation": "prepare_relax_mc_batch",
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "branch_upper": preview["selected_branch_upper"],
        "relative_cost_upper": 3 * amount,
        "maximum_mc_segments": 3,
        "submit_authorized": False,
        "mc_policy": deepcopy(config.get("mc_policy") or {}),
    }
    action["parameters"] = params
    return action


def prepare_generated_module(action, result, context):
    scope = (action.get("parameters") or {}).get("module_execution_scope")
    if not scope or not result.get("registered"):
        return result
    state = result["state"]
    from phase_agent.tools.relax_mc_batch.plan import attach_preview
    from phase_agent.tools.relax_mc_batch.prepare import prepare_batch
    from phase_agent.tools.policy.validate_tool_action import validate_tool_action

    branches = sorted({r["branch_id"] for r in result["registered"]})
    config = context["effective_config"]
    why = None
    upper = scope.get("relative_cost_upper")
    if type(upper) not in (int, float) or not math.isfinite(upper) or upper < 0:
        why = "module_scope_cost_limit_invalid"
    elif type(scope.get("branch_upper")) is not int or scope["branch_upper"] < 0:
        why = "module_scope_branch_limit_invalid"
    elif scope.get("submit_authorized") is not False:
        why = "module_scope_cannot_authorize_submission"
    elif scope.get("config_version") != context.get("config_version") or scope.get(
        "model_version"
    ) != state.get("active_model_version"):
        why = "module_scope_version_changed"
    elif scope.get("maximum_mc_segments") != 3:
        why = "module_scope_segment_limit_changed"
    elif scope.get("mc_policy") != (config.get("mc_policy") or {}):
        why = "module_scope_rules_changed"
    elif len(branches) > scope.get("branch_upper", 0):
        why = "module_scope_branch_limit_exceeded"
    child = {
        "tool": "prepare_local_batch_files",
        "target_ids": branches,
        "parameters": {"mode": "relax_mc_inputs"},
        "task_key": str(action.get("task_key")) + ":relax-mc",
        "reason": "Prepare only the registered branches under the approved module envelope",
    }
    try:
        if why:
            raise ValueError(why)
        child_context = {**context, "event_state": state}
        child = attach_preview(child, state, child_context, config)
        if child["budget"] > scope["relative_cost_upper"]:
            raise ValueError("module_scope_cost_limit_exceeded")
        session = {
            "status": "confirmed",
            "confirmed_snapshot": {"config": config, "config_version": context["config_version"]},
        }
        check = validate_tool_action(
            child, state, session, {"prepare_local_batch_files": {"handler": prepare_batch}}
        )
        if not check["valid"]:
            raise ValueError(";".join(check["errors"]))
        prepared = prepare_batch(action=child, context=child_context)
        if prepared.get("status") not in {"prepared", "already_prepared"}:
            raise ValueError(prepared.get("reason") or prepared.get("status"))
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
        # Generation is already complete. Never retry it to repair packaging.
        state.setdefault("module_preparation_reports", []).append(
            {
                "parent_task_key": action.get("task_key"),
                "branch_ids": branches,
                "status": "blocked",
                "reason": str(error),
                "scope": deepcopy(scope),
            }
        )
        return {
            **result,
            "module_preparation": {"status": "blocked", "reason": str(error)},
            "reason": "Branch generation completed; module packaging blocked: " + str(error),
        }
    state = prepared.get("state") or state
    for task in state.get("tasks") or []:
        if task.get("relax_mc_bundle_id") == prepared.get("bundle_id"):
            task["scientific_parent_task_key"] = action.get("task_key")
    state.setdefault("module_preparation_reports", []).append(
        {
            "parent_task_key": action.get("task_key"),
            "branch_ids": branches,
            "bundle_id": prepared.get("bundle_id"),
            "status": prepared["status"],
            "scope": deepcopy(scope),
        }
    )
    return {
        **result,
        **prepared,
        "state": state,
        "summary": result["summary"],
        "registered": result["registered"],
        "module_preparation": {"status": prepared["status"], "scope": deepcopy(scope)},
        "submitted": False,
    }

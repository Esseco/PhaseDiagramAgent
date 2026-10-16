"""Configurable limits for the single evidence-driven scientific workflow."""

from copy import deepcopy
import math


def settings(config):
    workflow = config.get("workflow") or {}
    legacy = (config.get("agent") or {}).get("adaptive") or {}
    training = (config.get("mlip_finetune") or {}).get("training") or {}
    minimum = workflow.get(
        "finetune_min_new_dft",
        legacy.get(
            "N_min",
            training.get(
                "minimum_new_dft_records",
                ((config.get("qbc") or {}).get("retrain") or {}).get("minimum_new_dft_records", 10),
            ),
        ),
    )
    maximum = workflow.get("finetune_max_new_dft", legacy.get("N_max"))
    baseline = (config.get("run") or {}).get("total_quota", config.get("total_quota", 900))
    batch = workflow.get("branch_batch_size")
    if batch is None:
        batch = legacy.get("branch_batch_size")
        if batch is None and any(
            key in legacy for key in ("branch_batch_fraction", "branch_batch_max")
        ):
            ratio = legacy.get("branch_batch_fraction", 0.4)
            cap = legacy.get("branch_batch_max", 300)
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
                raise ValueError("workflow_legacy_branch_fraction_invalid")
            if type(cap) is not int or cap <= 0:
                raise ValueError("workflow_legacy_branch_max_invalid")
            batch = min(cap, max(1, math.ceil(baseline * ratio)))
        if batch is None:
            batch = baseline
    if type(batch) is not int or batch <= 0:
        raise ValueError("workflow.branch_batch_size_must_be_positive_integer")
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("workflow.finetune_min_new_dft_must_be_positive_integer")
    if maximum is not None and (type(maximum) is not int or maximum < minimum):
        raise ValueError("workflow.finetune_max_new_dft_requires_integer_ge_minimum")
    return {
        "branch_batch_size": batch,
        "finetune_min_new_dft": minimum,
        "finetune_max_new_dft": maximum,
    }


def normalize_workflow_state(state, *, _owned=False):
    """Archive scheduling metadata, preserving real jobs and frozen scientific approvals."""
    current = state if _owned else deepcopy(state)
    legacy = {}
    for key in (
        "search_mode",
        "search_mode_review",
        "search_mode_reviewed_stages",
        "adaptive",
    ):
        if key in current:
            legacy[key] = current.pop(key)
    pending = current.get("pending_execution_policies") or {}
    if "__search_mode__" in pending:
        legacy["switch_proposal"] = pending.pop("__search_mode__")
    if legacy:
        current.setdefault("workflow_migration_history", []).append(legacy)
    current["workflow_version"] = "unified-v1"
    return current


def active_tasks(state):
    from phase_agent.tools.state.task_waiting import awaiting_task_result

    rows = {
        row.get("task_id", str(index)): row
        for index, row in enumerate(state.get("pending_tasks") or [])
    }
    rows.update(
        {row.get("task_id", str(index)): row for index, row in enumerate(state.get("tasks") or [])}
    )
    return [
        row
        for row in rows.values()
        if awaiting_task_result(row)
        or row.get("status") in {"pending", "running", "submitted", "unknown", "submission_unknown"}
    ]


def new_dft_ids(state, config):
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    model = state.get("active_model") or config.get("mlip") or {}
    version = (
        state.get("active_model_version")
        or model.get("version")
        or (config.get("calculation") or {}).get("mlip_version")
        or model.get("name")
    )
    ids = set()
    for row in state.get("new_dft_records") or []:
        energy = row.get("energy")
        identity = row.get("data_id") or row.get("task_id")
        if (
            identity
            and row.get("status") == "completed"
            and row.get("converged") is True
            and row.get("checks_passed") is True
            and isinstance(energy, (int, float))
            and not isinstance(energy, bool)
            and math.isfinite(energy)
            and spin_standard_passed(row)
            and row.get("model_version") in {None, version}
        ):
            ids.add(identity)
    return ids


def branch_batch_errors(action, config):
    from phase_agent.tools.policy.region_sampling import is_region, errors

    if is_region(action):
        return errors(action.get("parameters") or {})
    params = action.get("parameters") or {}
    if params.get("generation_plan"):
        rows = params["generation_plan"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return ["branch_generation_plan_invalid"]
        counts = [row.get("quota") for row in rows]
    elif params.get("quotas") is not None:
        if not isinstance(params["quotas"], dict):
            return ["branch_quotas_invalid"]
        counts = list(params["quotas"].values())
    else:
        counts = [params.get("total_quota")]
    if counts == [None]:
        return ["branch_requires_explicit_quota"]
    if (
        not counts
        or any(type(count) is not int or count < 0 for count in counts)
        or sum(counts) <= 0
    ):
        return ["branch_quota_requires_nonnegative_integer_and_positive_total"]
    total = sum(counts)
    if "total_quota" in params and (
        type(params["total_quota"]) is not int or params["total_quota"] != total
    ):
        return ["branch_total_quota_mismatch"]
    from phase_agent.tools.policy.adaptive_branch_count import is_adaptive_generation

    if is_adaptive_generation(action):
        if total < 100:
            return ["branch_batch_below_minimum_100"]
        return ["branch_batch_exceeds_limit"] if total > 1000 else []
    return ["branch_batch_exceeds_limit"] if total > settings(config)["branch_batch_size"] else []


def action_errors(action, state, config):
    policy = settings(config)
    tool = action.get("tool") or action.get("action_type")
    params = action.get("parameters") or {}
    errors = branch_batch_errors(action, config) if tool == "generate_branches" else []
    if tool == "update_mlip" and params.get("action", "RETRAIN_MLIP") == "RETRAIN_MLIP":
        if len(new_dft_ids(state, config)) < policy["finetune_min_new_dft"]:
            errors.append("finetune_min_new_dft_not_reached")
        if active_tasks(state):
            errors.append("fine_tune_waits_for_inflight_tasks")
    stage = action.get("stage") or params.get("stage")
    direct_dft = tool == "run_calculation_stage" and stage in {
        "dft_single_point",
        "dft_relax",
    }
    from phase_agent.tools.budget.check_budget import budget_is_hard

    if (
        budget_is_hard(config.get("budgets"))
        and (tool == "select_dft_candidates" or direct_dft)
        and policy["finetune_max_new_dft"] is not None
    ):
        if direct_dft:
            selected = set(action.get("target_ids") or [])
        else:
            decisions = params.get("decisions") or []
            if not isinstance(decisions, list) or any(
                not isinstance(row, dict) for row in decisions
            ):
                return errors + ["dft_decisions_require_object_array"]
            selected = {
                row.get("candidate_id")
                for row in decisions
                if row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
            }
        if not selected:
            return errors
        inflight = {
            row.get("structure_id") or row.get("object_id") or row.get("task_id")
            for row in active_tasks(state)
            if row.get("stage") in {"dft_single_point", "dft_relax"}
        }
        if (
            len(new_dft_ids(state, config)) + len(inflight | selected)
            > policy["finetune_max_new_dft"]
        ):
            errors.append("dft_accumulation_exceeds_finetune_max")
    return errors


def needs_human(action, config):
    tool = action.get("tool") or action.get("action_type")
    params = action.get("parameters") or {}
    if tool == "generate_branches":
        system = config.get("system_config") or config.get("system") or {}
        cap = (
            (system.get("H_generation") or {}).get("first_round_max_det_H")
            or (config.get("generation_options") or {}).get("max_det_H")
            or (config.get("generation_actions") or {}).get("first_round_max_det_H")
        )
        caps = [params["max_det_H"]] if "max_det_H" in params else []
        caps.extend(row["max_det_H"] for row in params.get("regions") or [])
        caps.extend(
            row["max_det_H"] for row in params.get("generation_plan") or [] if "max_det_H" in row
        )
        return bool(
            caps and (cap is None or any(type(value) is not int or value > cap for value in caps))
        )
    if tool == "select_dft_candidates":
        return any(
            isinstance(row, dict) and row.get("action") == "DFT_RELAX"
            for row in params.get("decisions") or []
        )
    if (
        tool == "run_calculation_stage"
        and (action.get("stage") or params.get("stage")) == "dft_relax"
    ):
        return True
    return tool == "update_mlip" and params.get("action") == "ACTIVATE_CANDIDATE_MODEL"


def initial_direction_errors(action, state, assessment):
    """Validate the confirmed cold-start route; never choose an action for the LLM."""
    if not assessment or assessment.get("status") != "evaluated":
        return []
    if (
        state.get("model_activation_history")
        or state.get("last_trained_dataset_version")
        or (state.get("active_model") or {}).get("dataset_version")
    ):
        return []
    if assessment.get("eligible_structures", 0) < assessment.get("minimum_new_dft_records", 10):
        return []
    if (action.get("tool") or action.get("action_type")) != "update_mlip":
        return ["first_round_requires_initial_finetune_proposal"]
    return []

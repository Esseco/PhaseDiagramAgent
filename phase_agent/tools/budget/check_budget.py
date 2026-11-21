"""只检查预算，不提前扣账。"""


def budget_is_hard(limits):
    policy = (limits or {}).get("enforcement", "advisory")
    if policy not in {"advisory", "hard"}:
        raise ValueError("budget_enforcement_must_be_advisory_or_hard")
    return policy == "hard"


def check_budget(state: dict, request: dict, limits: dict) -> dict:
    from math import isfinite

    usage = state.get("budget_usage", {})
    reasons = []
    value = request.get("relative_cost", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        reasons.append("invalid_relative_cost")
    active_reservations = [
        item
        for item in (state.get("budget_reservations") or {}).values()
        if item.get("status") in {"reserved", "submitted", "running"}
    ]
    stage = request.get("stage")
    if stage:
        stage_limit = limits.get("stage_limits", {}).get(stage)
        if stage_limit is None:
            reasons.append("unknown_stage")
        else:
            stage_usage = usage.get("stages", {}).get(stage, {})
            reserved_stage = [item for item in active_reservations if item.get("stage") == stage]
            if (
                stage not in {"dft_single_point", "dft_relax"}
                and stage_limit.get("max_tasks") is not None
                and stage_usage.get("tasks", 0) + len(reserved_stage) + int(request.get("tasks", 1))
                > stage_limit["max_tasks"]
            ):
                reasons.append("stage_task_limit")
    llm = request.get("llm_usage")
    if llm:
        llm_usage = usage.get("llm", {})
        llm_limits = limits["llm_limits"]
        for field, limit_key in (
            ("calls", "max_calls"),
            ("input_tokens", "max_input_tokens"),
            ("output_tokens", "max_output_tokens"),
            ("cost", "max_cost"),
        ):
            limit = llm_limits.get(limit_key)
            if limit is not None and llm_usage.get(field, 0) + llm.get(field, 0) > limit:
                reasons.append(f"llm_{field}_limit")
        iteration = str(request.get("iteration", 0))
        if (
            llm_usage.get("calls_by_iteration", {}).get(iteration, 0) + llm.get("calls", 0)
            > llm_limits["max_calls_per_iteration"]
        ):
            reasons.append("llm_iteration_call_limit")
    hard = budget_is_hard(limits)
    mandatory = {"unknown_stage", "invalid_relative_cost"}
    if stage not in {"dft_single_point", "dft_relax"}:
        mandatory.add("stage_task_limit")
    errors = reasons if hard else [reason for reason in reasons if reason in mandatory]
    return {
        "allowed": not errors,
        "reasons": errors,
        "warnings": [reason for reason in reasons if reason not in errors],
        "request": request,
        "enforcement": "hard" if hard else "advisory",
    }

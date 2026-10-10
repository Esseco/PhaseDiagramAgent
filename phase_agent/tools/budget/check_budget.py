"""只检查预算，不提前扣账。"""


def check_budget(state: dict, request: dict, limits: dict) -> dict:
    usage = state.get("budget_usage", {})
    reasons = []
    cost = float(request.get("relative_cost", 0.0))
    active_reservations = [
        item
        for item in (state.get("budget_reservations") or {}).values()
        if item.get("status") in {"reserved", "submitted", "running"}
    ]
    reserved_total = sum(
        float(item.get("reserved_cost", item.get("relative_cost", 0)))
        for item in active_reservations
    )
    if (
        usage.get("total_relative_cost", 0.0) + reserved_total + cost
        > limits["total_relative_cost"]
    ):
        reasons.append("total_cost_limit")
    stage = request.get("stage")
    if stage:
        stage_limit = limits.get("stage_limits", {}).get(stage)
        if stage_limit is None:
            reasons.append("unknown_stage")
        else:
            stage_usage = usage.get("stages", {}).get(stage, {})
            reserved_stage = [item for item in active_reservations if item.get("stage") == stage]
            if (
                stage_limit.get("max_tasks") is not None
                and stage_usage.get("tasks", 0) + len(reserved_stage) + int(request.get("tasks", 1))
                > stage_limit["max_tasks"]
            ):
                reasons.append("stage_task_limit")
            reserved_stage_cost = sum(
                float(item.get("reserved_cost", item.get("relative_cost", 0)))
                for item in reserved_stage
            )
            if (
                stage_limit.get("max_cost") is not None
                and stage_usage.get("cost", 0.0) + reserved_stage_cost + cost
                > stage_limit["max_cost"]
            ):
                reasons.append("stage_cost_limit")
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
    return {"allowed": not reasons, "reasons": reasons, "request": request}

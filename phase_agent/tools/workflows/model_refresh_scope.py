"""Explicit limits for the existing one-wave refresh supplementation policy."""

from copy import deepcopy


def approved_refresh_scope(plan, config_version):
    return {
        "plan_checksum": plan["checksum"],
        "model_version": plan["new_model_version"],
        "config_version": config_version,
        "eligible_structure_ids": [
            row["structure_id"] for row in plan["candidates"] if row["operation"] == "predict"
        ],
        "maximum_count": plan["maximum_supplemental_count"],
        "maximum_cost": plan["maximum_supplemental_cost"],
        "remaining_waves": 1,
        "tool": "prepare_local_batch_files",
        "submit_authorized": False,
    }


def refresh_scope_errors(refresh, config_version):
    scope = refresh.get("approved_scope")
    if not scope:
        return ["缺少已批准的补充范围，需单独审阅本批方案"]
    plan = refresh["plan"]
    rows = refresh.get("supplemental_candidates") or []
    errors = []
    if scope.get("plan_checksum") != plan.get("checksum") or scope.get(
        "model_version"
    ) != refresh.get("new_model_version"):
        errors.append("刷新方案或模型版本已变化")
    if scope.get("config_version") != config_version:
        errors.append("配置版本已变化")
    if scope.get("remaining_waves") != 1 or refresh.get("wave") != 1:
        errors.append("已批准的补充次数已用完或轮次不匹配")
    ids = [row["structure_id"] for row in rows]
    identities = [(row["structure_id"], row.get("structure_sha256")) for row in rows]
    eligible = {
        (row["structure_id"], row.get("structure_sha256"))
        for row in plan["candidates"]
        if row["operation"] == "predict"
    }
    sources = [
        (
            row["structure_id"],
            row.get("approved_input_structure_sha256", row.get("structure_sha256")),
        )
        for row in rows
    ]
    if (
        len(identities) != len(set(identities))
        or not set(ids).issubset(scope.get("eligible_structure_ids") or [])
        or not set(sources).issubset(eligible)
        or any(not digest for _, digest in identities)
    ):
        errors.append("补充目标超出已批准范围或重复")
    if len(rows) > scope.get("maximum_count", 0):
        errors.append("补充数量超过上限")
    costs = [row.get("relative_cost") for row in rows]
    import math

    if any(
        not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0 for cost in costs
    ):
        errors.append("补充成本缺失或无效")
    elif sum(costs) > scope.get("maximum_cost", 0):
        errors.append("补充成本超过上限")
    return errors


def consume_refresh_scope(refresh):
    refresh["approved_scope"] = {
        **deepcopy(refresh["approved_scope"]),
        "remaining_waves": 0,
    }


def is_refresh_planning_hold(state, key, reservation):
    """An approved future allowance is not evidence of an unfinished action."""
    import math
    from phase_agent.tools.workflows.model_refresh_state import current_model_refresh

    refresh = current_model_refresh(state)
    plan = refresh.get("plan") or {}
    scope = refresh.get("approved_scope") or {}
    amount = reservation.get("reserved_cost")
    return bool(
        refresh.get("status") in {"waiting_results", "supplement_ready"}
        and refresh.get("approved_by")
        and refresh.get("wave") in {0, 1}
        and plan.get("checksum")
        and key == refresh.get("supplemental_reservation_key")
        and key == f"model-refresh:{plan['checksum']}:supplement-envelope"
        and reservation.get("task_key") == key
        and reservation.get("status") == "reserved"
        and not reservation.get("reconciliation_required")
        and reservation.get("stage") == "relax_and_feature"
        and reservation.get("config_version")
        == scope.get("config_version")
        == state.get("confirmed_config_version")
        and reservation.get("model_version")
        == scope.get("model_version")
        == plan.get("new_model_version")
        == refresh.get("new_model_version")
        and scope.get("plan_checksum") == plan["checksum"]
        and scope.get("remaining_waves") == 1
        and scope.get("maximum_count") == plan.get("maximum_supplemental_count")
        and scope.get("tool") == "prepare_local_batch_files"
        and scope.get("submit_authorized") is False
        and isinstance(amount, (int, float))
        and not isinstance(amount, bool)
        and math.isfinite(amount)
        and amount > 0
        and amount == scope.get("maximum_cost") == plan.get("maximum_supplemental_cost")
    )

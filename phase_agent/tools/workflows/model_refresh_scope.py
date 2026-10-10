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
    if len(ids) != len(set(ids)) or not set(ids).issubset(
        scope.get("eligible_structure_ids") or []
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
    refresh["approved_scope"] = {**deepcopy(refresh["approved_scope"]), "remaining_waves": 0}

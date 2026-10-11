"""Attach the same frozen allocation preview to initial model MC proposals."""

from copy import deepcopy
from phase_agent.decisions.strategy.estimate_branch_mc_budget import estimate_branch_mc_budget


def prepare_mc_preview(action, state, context, config):
    if action.get("tool") != "allocate_mc_bohb":
        return action
    manager = (context or {}).get("manager")
    if manager is None:
        raise ValueError("缺少MC分支台账，不能生成可审批分配方案")
    result = deepcopy(action)
    params = result.setdefault("parameters", {})
    version = params.get("hull_reference_version") or state.get("current_branch_hull_version")
    pool = (state.get("branch_hull_batches") or {}).get(version)
    ids = list(dict.fromkeys(result.get("target_ids") or []))
    branches = manager.data.get("branches") or {}
    if not pool or not ids or any(key not in branches for key in ids):
        raise ValueError("MC候选或冻结Relax凸包缺失，请先完成回收与候选规划")
    steps = params.get("mc_budget")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("MC总步数必须为正整数，相对成本不能作为步数使用")
    preview = estimate_branch_mc_budget(
        [{**branches[key], "branch_id": key} for key in ids],
        pool,
        state,
        config,
        step_limit=steps,
        seed=int(params.get("seed", (config.get("run") or {}).get("seed", 42))),
    )
    if preview.get("missing_branch_ids"):
        raise ValueError("所选MC分支缺少有效Ehull，不能生成可审批方案")
    if preview.get("selected_branch_count", 0) <= 0:
        raise ValueError("当前MC没有可分配任务：核对已有活动MC、步数下限和候选资格后再提出方案")
    if preview.get("round_kind"):
        params["round_kind"] = preview["round_kind"]
    if preview.get("round_kind") == "second":
        params["first_round_source_checksum"] = preview["first_round_source_checksum"]
    params["hull_reference_version"] = version
    params["phase_diagram_version"] = preview["phase_diagram_version"]
    params["budget_preview"] = preview
    # A model cannot silently decide to reduce the requested coverage.
    if result.get("decision_source") != "explicit_user_instruction":
        params.pop("compression_choice", None)
    result["budget"] = preview["estimated_relative_cost"]
    result["expected_cost"] = preview["estimated_relative_cost"]
    result["expected_purpose"] = (
        f"本次覆盖{preview['selected_branch_count']}个分支，合计{preview['requested_steps']}步；"
        f"完整范围{preview['full_plan_branch_count']}个分支、{preview['full_plan_steps']}步。"
        + (
            "当前步数不足以覆盖完整方案，请选择提高步数或明确缩小分支范围。"
            if preview.get("compression_requires_user_choice")
            else "批准后仅准备所展示的MC任务与输入。"
        )
    )
    return result

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
    params.setdefault("seed", config.get("seed", (config.get("run") or {}).get("seed", 42)))
    version = params.get("hull_reference_version") or state.get("current_branch_hull_version")
    pool = (state.get("branch_hull_batches") or {}).get(version)
    ids = list(dict.fromkeys(result.get("target_ids") or []))
    branches = manager.data.get("branches") or {}
    if not pool or not ids or any(key not in branches for key in ids):
        raise ValueError("MC候选或冻结Relax凸包缺失，请先完成回收与候选规划")
    from phase_agent.science.mc.second_round_state import first_round_source

    source = first_round_source(state, pool.get("model_version"))
    original_ids = set(ids)
    if source:
        if set(ids) - set(source["branch_ids"]):
            raise ValueError("追加MC只能从上一段实际保留的分支中选择，请修正计划。")
        ids = [key for key in source["branch_ids"] if key in branches]
    else:
        from phase_agent.analysis.phase.branch_relax_hull import rank_relaxed_branches
        from phase_agent.science.mc.continuation_policy import effective_mc_policy
        from phase_agent.science.mc.schedule_tiered_mc import _hb_protected

        ranked, _ = rank_relaxed_branches(
            [{**row, "branch_id": key} for key, row in branches.items()],
            pool,
            uncertainty_weight=float((config.get("bohb") or {}).get("uncertainty_weight", 1)),
            phase_diagram=(state.get("phase_diagrams") or {}).get("mlip") or {},
        )
        policy = effective_mc_policy(config)
        ids += [
            row["branch_id"]
            for row in ranked
            if row["branch_id"] not in original_ids and _hb_protected(row, policy)
        ]
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
        branch_count=params.get("branch_count"),
        exploration_fraction=params.get("exploration_fraction"),
    )
    missing = set(preview.get("missing_branch_ids") or [])
    if missing - original_ids:
        ids = [key for key in ids if key not in missing - original_ids]
        preview = estimate_branch_mc_budget(
            [{**branches[key], "branch_id": key} for key in ids],
            pool,
            state,
            config,
            step_limit=steps,
            seed=int(params.get("seed", (config.get("run") or {}).get("seed", 42))),
            branch_count=params.get("branch_count"),
            exploration_fraction=params.get("exploration_fraction"),
        )
    if preview.get("missing_branch_ids"):
        raise ValueError(
            f"所选MC分支中有{len(preview['missing_branch_ids'])}个缺少与当前相图匹配的有效弛豫记录，"
            f"其余{preview['candidate_count'] - len(preview['missing_branch_ids'])}个具备有效记录；"
            "需先核对缺失分支，尚未生成MC输入，不能将缺失Ehull视为0。"
        )
    if preview.get("selected_branch_count", 0) <= 0:
        if preview.get("status") == "continuation_pool_below_minimum":
            raise ValueError("有效MC追加候选不足10个，请先扩展Branch并完成Relax；未生成重复任务。")
        raise ValueError("当前MC没有可分配任务：核对已有活动MC、步数下限和候选资格后再提出方案")
    result["target_ids"] = ids
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
    selection = preview.get("selection") or {}
    if selection.get("round_kind") == "continuation":
        exploration_count = selection.get("actual_exploration_count", 0)
        result["expected_purpose"] += (
            f" 高/中档{preview['selected_branch_count'] - exploration_count}个，"
            f"低档/探索档{exploration_count}个；保护候选{selection.get('protected_count', 0)}个。"
            f" 高档{selection.get('high_required_count', 0)}个已全部保留，不受数量建议限制。"
            + (
                "高/中档不足，本次提高探索比例补充有效候选。"
                if selection.get("exploration_fill_count", 0)
                else ""
            )
        )
    return result

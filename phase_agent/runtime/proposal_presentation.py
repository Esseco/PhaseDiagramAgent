"""Human-readable proposal labels; presentation never changes allocations."""

TOOL_LABELS = {
    "generate_branches": "生成候选结构",
    "prepare_local_batch_files": "准备计算输入文件",
    "allocate_mc_bohb": "安排 MC 搜索",
    "select_dft_candidates": "选择 DFT 验证结构",
    "run_calculation_stage": "运行本次计算",
    "restart_failed_task": "重试失败任务",
    "check_convergence": "检查搜索进展与收敛",
    "pause_search": "暂停搜索",
}
STRATEGY_LABELS = {
    "coverage": ("基础覆盖", "广泛采样允许的结构空间"),
    "composition": ("Na 含量补充", "增加不同 Na 含量的候选"),
    "periodic_extension": ("超胞补充", "增加不同周期超胞的候选"),
    "competing_phase": ("相间补充", "增加其他允许相的候选"),
    "tm_ordering": ("TM 排布补充", "增加过渡金属占位排列的候选"),
}


def concise_generation_lines(params, cost):
    plan = params.get("generation_plan") or []
    total = params.get("total_quota")
    lines = ["建议：生成候选结构"]
    lines.append(
        f"计划采样 {total} 个候选，按已确认的相、Na 含量和超胞边界生成。"
        if total is not None
        else "按已确认的边界生成候选结构，数量以保存方案为准。"
    )
    if plan:
        lines.append("")
        for row in plan:
            label, meaning = STRATEGY_LABELS.get(
                row["strategy"], (row["strategy"], row.get("reason") or "按保存方案采样")
            )
            target = "全部允许相" if row["phase"] == "all" else row["phase"]
            limits = []
            if row.get("na_min") is not None:
                limits.append(f"Na/O₂ {row['na_min']:g}–{row['na_max']:g}")
            if row.get("max_det_H") is not None:
                limits.append(f"det(H) ≤ {row['max_det_H']}")
            scope = "，" + "，".join(limits) if limits else ""
            lines.append(f"- {label} {row['quota']} 个（{target}{scope}）：{meaning}。")
        lines.append("")
    elif isinstance(params.get("quotas"), dict):
        lines.append(
            "采样分配："
            + "、".join(
                f"{STRATEGY_LABELS.get(k, (k, ''))[0]} {v} 个"
                for k, v in params["quotas"].items()
                if v
            )
        )
    cap = params.get("max_det_H")
    lines.append(
        f"超胞上限：det(H) ≤ {cap}。" if cap is not None else "超胞范围：按已确认的 H 边界。"
    )
    selected = params.get("batch_size")
    count = params.get("initial_states_per_branch")
    selection = (
        f"去重后最多选入 {selected} 个 branch（结构分支）"
        if selected is not None
        else "去重后按配置选择 branch（结构分支）"
    )
    initial = f"每个最多生成 {min(count, 3)} 个初态" if type(count) is int else "初态数量按配置"
    lines.append(f"{selection}，{initial}；实际数量取决于合法候选和去重结果。")
    estimated = cost.get("estimated_total_cost")
    if estimated is not None and estimated != 0:
        lines.append(f"本次预计相对预算：{estimated}（不是耗时）。")
    lines.append("本次只生成结构，不运行 Relax、MC 或 DFT；本地生成仍会耗时。")
    return lines

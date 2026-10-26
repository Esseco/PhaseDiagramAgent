"""Readable default replies; detailed responses retain the original record."""

import re


def approval_reply(proposal, state, *, state_path=None, config=None):
    if not proposal:
        from phase_agent.runtime.plan_queries import pending_plan_reply

        return pending_plan_reply(state)

    from phase_agent.runtime.review_presentation import review_card

    pending = state.get("pending_execution_policies") or {}
    match = next(
        ((key, row) for key, row in pending.items() if row.get("agent_proposal") == proposal),
        None,
    )
    card = review_card(*match) if match else review_card("", {"agent_proposal": proposal})
    purpose = str(
        proposal.get("expected_purpose")
        or proposal.get("reason")
        or card.get("scope")
        or "核对并准备下一步操作"
    )
    purpose = re.sub(r"config-\d+-[a-f0-9]+", "已确认配置", purpose)
    params = (
        (proposal.get("raw_action") or {}).get("parameters")
        or proposal.get("action_parameters")
        or {}
    )
    tool = (proposal.get("raw_action") or {}).get("tool") or proposal.get("recommended_action")
    dft_preview = params.get("dft_input_preview") or {}
    if (
        tool == "select_dft_candidates"
        and dft_preview
        and (not isinstance(params.get("decisions"), list) or not params["decisions"])
    ):
        count = dft_preview.get("task_count")
        scope = f"原预览记录 {count} 个 DFT 输入" if count is not None else "原 DFT 预览"
        return (
            scope + "，但缺少可执行的选点决策明细，当前不可批准。\n"
            "请核对已保存的原选点并补齐方案；未生成任务或提交计算。"
        )
    preview = params.get("relax_mc_preview") or {}
    workload = preview.get("workload") or {}
    result = "待确认：" + card["title"] + "。"
    if (
        (proposal.get("raw_action") or {}).get("tool") or proposal.get("recommended_action")
    ) == "select_dft_candidates":
        result += card["scope"] + "。"
    reason = proposal.get("reason") or (proposal.get("raw_action") or {}).get("reason")
    if reason and reason != purpose:
        result += str(reason).rstrip("。") + "。"
    scope_lines = []
    if tool == "generate_branches" and params.get("sampling_mode") != "region_adaptive":
        from phase_agent.runtime.proposal_presentation import concise_generation_lines

        scope_lines = concise_generation_lines(params, {})[1:]
    elif tool == "generate_branches":
        from phase_agent.runtime.proposal_presentation import region_scope_lines

        scope_lines = region_scope_lines(params)
    elif tool == "allocate_mc_bohb":
        from phase_agent.runtime.proposal_presentation import mc_scope_lines

        scope_lines = mc_scope_lines(params)
    elif "supplemental_relax_parameters" in params:
        settings = params["supplemental_relax_parameters"]
        count = len((proposal.get("raw_action") or {}).get("target_ids") or [])
        result = f"待确认：为 {count} 个单点候选追加弛豫输入；首批结果保留。"
        scope_lines = [
            f"设置：最多 {settings.get('relax_steps', 500)} 步，{settings.get('optimizer', 'LBFGS')}；未收敛结果排除，不自动延长或重试。"
        ]
    from phase_agent.tools.policy.training_input_action import is_training_input_action

    if is_training_input_action(proposal.get("raw_action") or {"tool": tool, "parameters": params}):
        result = "待确认：微调模型，生成超算训练提交文件。"
        scope_lines.append(
            "批准后仅生成训练输入与提交脚本，由你上传提交；不本机训练、不自动激活模型。"
        )
    jobs = workload.get("gpu_jobs_upper_by_wave")
    budget = (
        f"最多 {sum(jobs)} 个 GPU 计算作业，另 1 个控制作业" if jobs else "计算量待任务展开后估算"
    )
    if "supplemental_relax_parameters" in params:
        budget = f"本批 {count} 个弛豫任务；预计相对成本 {(proposal.get('estimated_cost') or {}).get('estimated_total_cost', '未记录')}"
    if workload:
        count = workload.get("relax_structure_tasks")
        waves = workload.get("mc_task_upper") or []
        result = f"弛豫 {count} 个结构，最多 {len(waves)} 段 MC。"
        if preview.get("bootstrap_first_mc"):
            result += "MC 按回收后的凸包结果选择；缺少有效凸包或 Na 端点时暂停。"
    runtime = (proposal.get("estimated_cost") or {}).get("runtime_budget") or {}
    seconds = runtime.get("serial_seconds")
    budget += (
        f"；串行累计约 {seconds / 3600:.2g} 小时，不含排队。"
        if seconds is not None
        else "；耗时暂无可靠估计。"
    )
    if (
        tool == "generate_branches"
        and params.get("sampling_mode") == "region_adaptive"
        and not params.get("regions")
    ):
        budget = "暂不新增branch；后续搜索工作量另行提案，不执行计算。"
    if proposal.get("recommended_action") == "check_convergence":
        budget = "仅检查已有数据，不新增计算任务。"
    from phase_agent.runtime.evidence_reference_presentation import (
        evidence_reference_lines,
    )

    evidence_lines = [
        line.replace("依据提醒：", "需注意：", 1)
        for line in evidence_reference_lines(proposal.get("raw_action") or {})
        if line.startswith("依据提醒：")
    ]
    boundary = card.get("approval_boundary") or "按本方案执行"
    from phase_agent.tools.workflows.generation_artifacts import proposal_artifacts

    artifacts = proposal_artifacts(proposal, state)
    from phase_agent.runtime.review_file_presentation import file_location_lines

    artifact_lines = []
    if artifacts.get("text"):
        artifact_lines = file_location_lines("Branch 分布与构成文件", artifacts["text"])
    from phase_agent.tools.workflows.dft_selection_artifacts import (
        proposal_artifacts as dft_artifacts,
    )

    dft_details = dft_artifacts(proposal, state)
    if dft_details:
        artifact_lines.extend(file_location_lines("DFT 选点文件", dft_details["text"]))
    from phase_agent.runtime.handoff_presentation import planned_handoff_lines

    path_lines = planned_handoff_lines(proposal, state, state_path, config=config)
    scope_warning = []
    next_step = "下一步：回复‘同意’、‘拒绝’或提出修改。确认范围：" + boundary
    if artifacts.get("scope_mismatch"):
        scope_warning.append(
            "需注意：本批目标与该模型最近生成的 branch 完全无交集；请先核对概况文件中的范围。"
        )
        next_step = "下一步：先修正 branch 范围后再审批。"
    elif artifacts.get("scope_review_unavailable"):
        scope_warning.append("需注意：尚未生成可核对的 branch 概况文件。")
        next_step = "下一步：先补齐 branch 生成概况和范围核对，再审批。"
    shared = (proposal.get("raw_action") or {}).get("_shared_dft_snapshot")
    if shared is not None:
        artifact_lines.append(
            "引用DFT："
            + "、".join(shared["source_projects"])
            + f"；本方案固定 {len(shared['record_ids'])} 条记录，新增数据下次审批纳入。"
        )
    from phase_agent.runtime.review_presentation import revision_lines

    revisions = (
        [f"方案修订 {card['revision']}", *revision_lines(match[1])]
        if match and card["revision"]
        else []
    )
    return "\n".join(
        [
            "本次：" + purpose,
            *revisions,
            "结果：" + result,
            *[line for line in scope_lines if line],
            "预算：" + budget,
            *evidence_lines,
            *artifact_lines,
            *scope_warning,
            *path_lines,
            next_step,
        ]
    )


def concise_body(reply, state):
    from phase_agent.runtime.response_preferences import detailed_response

    if detailed_response():
        return str(reply)
    hidden = (
        "方案：",
        "已保存方案：",
        "预计相对成本：",
        "依据提醒：",
        "依据：",
        "审批范围：",
        "审批边界：",
        "本次仅查看",
        "完整记录：",
        "详细记录：",
        "当前项目状态：",
        "累计任务：",
        "剩余相对预算",
    )
    lines = []
    budgets = []
    for line in str(reply).splitlines():
        line = line.strip()
        if not line or line.startswith(hidden):
            continue
        if line.startswith("诊断提示："):
            if "missing Na endpoints" in line:
                line = "结果：缺少 Na 端点或有效凸包时，MC 将暂停。"
            else:
                # Keep unresolved warnings visible; never hide a possible blocker.
                line = line.replace("诊断提示：", "需注意：", 1)
        if line.startswith(("耗时粗估：", "训练耗时：", "预算：", "GPU 占用统计：")):
            if "暂无足够" in line:
                line = "耗时暂无可靠估计。"
            budgets.append(line.removeprefix("预算：").removeprefix("耗时粗估："))
            continue
        if line.startswith("封装估算："):
            jobs = re.search(r"GPU 计算作业最多 (\d+) 个", line)
            if jobs:
                budgets.insert(0, f"最多 {jobs[1]} 个 GPU 计算作业，另 1 个控制作业。")
            line = "结果：" + line.removeprefix("封装估算：").split("；GPU", 1)[0] + "。"
        line = re.sub(r"(?:sha256:)?[a-f0-9]{32,64}", "", line)
        line = re.sub(r"config-\d+-[a-f0-9]+", "已确认配置", line)
        if line not in lines:
            lines.append(line)
    # A status query can include the full approval card. Keep its exact scope
    # and one review instruction, instead of printing two competing next steps.
    next_steps = [line for line in lines if line.startswith("下一步：")]
    scoped = next((line for line in next_steps if "确认范围：" in line), None)
    if scoped and len(next_steps) > 1:
        boundary = scoped.split("确认范围：", 1)[1]
        merged = "下一步：查看方案后回复“同意”“拒绝”或提出修改。确认范围：" + boundary
        if any("本次查询不会批准" in line for line in next_steps):
            merged += "本次查询不会批准。"
        index = next(i for i, line in enumerate(lines) if line.startswith("下一步："))
        lines = [line for line in lines if not line.startswith("下一步：")]
        lines.insert(index, merged)
    if state.get("pending_execution_policies") and not any(
        line.startswith("本次：") for line in lines
    ):
        lines.insert(
            0,
            "本次："
            + (
                "未生成有效新方案，原待审方案保留。"
                if state.get("pending_execution_policies")
                and any(
                    line.startswith(
                        ("本轮未执行：", "本轮失败：", "结果：本轮未执行：", "结果：本轮失败：")
                    )
                    for line in lines
                )
                else "已保存输入或计算方案，等待确认。"
                if state.get("pending_execution_policies")
                else "已检查当前项目状态。"
            ),
        )
    if budgets:
        lines.append("预算：" + "；".join(dict.fromkeys(budgets)))
    return "\n".join(lines)

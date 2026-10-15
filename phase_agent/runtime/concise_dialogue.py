"""Readable default replies; detailed responses retain the original record."""

import re


def approval_reply(proposal, state):
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
        proposal.get("action_parameters")
        or (proposal.get("raw_action") or {}).get("parameters")
        or {}
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
    evidence = (proposal.get("raw_action") or {}).get("evidence_reference_check") or {}
    jobs = workload.get("gpu_jobs_upper_by_wave")
    budget = (
        f"最多 {sum(jobs)} 个 GPU 计算作业，另 1 个控制作业" if jobs else "计算量待任务展开后估算"
    )
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
    if proposal.get("recommended_action") == "check_convergence":
        budget = "仅检查已有数据，不新增计算任务。"
    from phase_agent.runtime.evidence_reference_presentation import evidence_reference_lines

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
    if artifacts:
        artifact_lines = file_location_lines("Branch", artifacts["json"])
    from phase_agent.tools.workflows.dft_selection_artifacts import (
        proposal_artifacts as dft_artifacts,
    )

    dft_details = dft_artifacts(proposal, state)
    if dft_details:
        artifact_lines.extend(file_location_lines("DFT 选点", dft_details["json"]))
    return "\n".join(
        [
            "本次：" + purpose,
            "结果：" + result,
            "下一步：回复‘同意’或提出修改。确认范围：" + boundary,
            "预算：" + budget,
            *evidence_lines,
            *artifact_lines,
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
        "当前状态：",
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
    if not any(line.startswith("本次：") for line in lines):
        lines.insert(
            0,
            "本次："
            + (
                "已保存输入或计算方案，等待确认。"
                if state.get("pending_execution_policies")
                else "已检查当前项目状态。"
            ),
        )
    lines.append(
        "预算：" + ("；".join(dict.fromkeys(budgets)) if budgets else "本次未提供新的计算预算。")
    )
    return "\n".join(lines)

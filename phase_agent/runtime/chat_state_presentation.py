"""Read-only conversation state presentation, separate from HTTP/workflow."""

from phase_agent.tools.state.task_waiting import active_pending_tasks


def waiting_training_job(state):
    jobs = [
        job
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("status") in {"inputs_prepared", "submitted", "running"}
        and not job.get("activated")
        and job.get("original_model_version") == state.get("active_model_version")
    ]
    return jobs[-1] if jobs else None


def brief_chat_state(state, *, configuring=False):
    """One factual state line, without re-listing files or scientific evidence."""
    if configuring:
        return "当前：配置修订中，等待确认。"
    returned = [
        job.get("returned_results")
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("status") == "results_received" and not job.get("activated")
    ]
    if returned:
        if any((row or {}).get("status") == "metadata_required" for row in returned):
            return "当前：微调结果已回收，模型清单待补齐；未激活。"
        return "当前：微调完成，仍在使用旧模型。"
    refresh = state.get("model_refresh") or {}
    if refresh and refresh.get("status") != "completed":
        return (
            "当前：新模型结构刷新方案待确认。"
            if refresh.get("status") == "approval_required"
            else f"当前：新模型结构刷新第{int(refresh.get('wave', 0)) + 1}批，等待输入准备或结果回传。"
        )
    if state.get("pending_dft_recovery_question"):
        return "当前：DFT 部分结果已回收，等待是否继续回收的确认。"
    pending = state.get("pending_execution_policies") or {}
    if pending:
        names = {
            "update_mlip": "微调训练输入方案",
            "select_dft_candidates": "DFT 输入方案",
            "allocate_mc_bohb": "MC 分配方案",
            "prepare_local_batch_files": "输入准备方案",
            "generate_branches": "branch 生成方案",
        }
        if len(pending) == 1:
            proposal = next(iter(pending.values())).get("agent_proposal") or {}
            tool = proposal.get("recommended_action") or (proposal.get("raw_action") or {}).get(
                "tool"
            )
            return f"当前：{names.get(tool, '操作方案')}待确认。"
        return f"当前：{len(pending)} 个方案待确认。"
    training = waiting_training_job(state)
    if training:
        return (
            "当前：微调输入已准备，等待超算提交与训练结果。"
            if training.get("status") == "inputs_prepared"
            else "当前：微调训练等待结果回传。"
        )
    tasks = state.get("tasks") or []
    active = active_pending_tasks(tasks)
    if active:
        names = {
            "deep_search": "MC",
            "relax_and_feature": "Relax",
            "dft_relax": "DFT",
            "dft_single_point": "DFT",
        }
        counts = {}
        for row in active:
            name = names.get(row.get("stage"), "任务")
            counts[name] = counts.get(name, 0) + 1
        return (
            "当前："
            + "、".join(f"{name} {count} 个待完成" for name, count in counts.items())
            + "。"
        )
    if state.get("pending_mc_regeneration"):
        return "当前：MC 重生成计划待确认。"
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment

    assessment = post_dft_assessment(state, state.get("confirmed_config") or {})
    if assessment:
        if assessment["status"] != "evaluated":
            return "当前：DFT 回收结束，本轮原模型误差评估待补齐。"
        return "当前：DFT 本轮误差已评估，等待微调、补DFT、新branch或收敛/停止建议。"
    failed = sum(row.get("status") in {"failed", "timeout"} for row in tasks)
    if failed:
        return f"当前：无运行中任务，{failed} 个任务失败/超时待处理。"
    completed_mc = [
        row
        for row in tasks
        if row.get("stage") == "deep_search" and row.get("status") == "completed"
    ]
    if completed_mc:
        return f"当前：MC 已完成 {len(completed_mc)} 个任务，等待下一步评估。"
    return "当前：无待运行任务，等待下一步指令。"


def epoch_stage_line(state, *, configuring=False):
    """Display model epoch independently of MC/DFT sub-rounds."""
    from phase_agent.analysis.state.model_epoch import model_epoch

    version = state.get("active_model_version") or (state.get("active_model") or {}).get("version")
    epoch = model_epoch(state, version) or "epoch0"
    stage = brief_chat_state(state, configuring=configuring).removeprefix("当前：").rstrip("。")
    if (
        not configuring
        and not state.get("tasks")
        and not state.get("pending_execution_policies")
        and not state.get("branches")
        and not state.get("generation_history")
        and "无待运行任务" in stage
    ):
        stage = "初始搜索，等待首轮方案"
    return f"{epoch} · {stage}"


def format_epoch_reply(reply, state, *, configuring=False):
    """Show the epoch once, removing only an identical repeated stage label."""
    header = epoch_stage_line(state, configuring=configuring)
    stage = header.split(" · ", 1)[-1].rstrip("。")
    lines = []
    for line in str(reply).splitlines():
        if line.startswith("当前状态：") and line.split("：", 1)[1].rstrip("。") == stage:
            continue
        lines.append(line)
    return header + "\n" + "\n".join(lines)

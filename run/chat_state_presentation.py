"""Read-only conversation state presentation, separate from HTTP/workflow."""
from execution_layer.state.task_waiting import active_pending_tasks

def brief_chat_state(state, *, configuring=False):
    """One factual state line, without re-listing files or scientific evidence."""
    if configuring:
        return "当前：配置修订中，等待确认。"
    if state.get("pending_dft_recovery_question"):
        return "当前：DFT 部分结果已回收，等待是否继续回收的确认。"
    pending = state.get("pending_execution_policies") or {}
    if pending:
        names = {"select_dft_candidates": "DFT 输入方案", "allocate_mc_bohb": "MC 分配方案",
                 "prepare_local_batch_files": "输入准备方案", "generate_branches": "branch 生成方案"}
        if len(pending) == 1:
            proposal = next(iter(pending.values())).get("agent_proposal") or {}
            tool = proposal.get("recommended_action") or (proposal.get("raw_action") or {}).get("tool")
            return f"当前：{names.get(tool, '操作方案')}待确认。"
        return f"当前：{len(pending)} 个方案待确认。"
    tasks = state.get("tasks") or []
    active = active_pending_tasks(tasks)
    if active:
        names = {"deep_search": "MC", "relax_and_feature": "Relax", "dft_relax": "DFT", "dft_single_point": "DFT"}
        counts = {}
        for row in active:
            name = names.get(row.get("stage"), "任务")
            counts[name] = counts.get(name, 0) + 1
        return "当前：" + "、".join(f"{name} {count} 个待完成" for name, count in counts.items()) + "。"
    if state.get("pending_mc_regeneration"):
        return "当前：MC 重生成计划待确认。"
    from analysis_layer.state.post_dft_assessment import post_dft_assessment
    assessment = post_dft_assessment(state, state.get("confirmed_config") or {})
    if assessment:
        if assessment["status"] != "evaluated":
            return "当前：DFT 回收结束，本轮原模型误差评估待补齐。"
        return "当前：DFT 本轮误差已评估，等待微调或新 branch 方案。"
    failed = sum(row.get("status") in {"failed", "timeout"} for row in tasks)
    if failed:
        return f"当前：无运行中任务，{failed} 个任务失败/超时待处理。"
    completed_mc = [row for row in tasks if row.get("stage") == "deep_search" and row.get("status") == "completed"]
    if completed_mc:
        return f"当前：MC 已完成 {len(completed_mc)} 个任务，等待下一步评估。"
    return "当前：无待运行任务，等待下一步指令。"

"""Factual progress display; no workflow execution, analysis or state mutation."""
from collections import Counter
from analysis_layer.state.model_epoch import model_epoch_label
from execution_layer.state.task_waiting import awaiting_task_result
from run.chat_state_presentation import brief_chat_state, waiting_training_job
from run.response_preferences import detailed_response

ACTION_NAMES = {
    "update_mlip": "生成微调训练输入", "generate_branches": "生成 branch",
    "allocate_mc_bohb": "分配 MC", "prepare_local_batch_files": "准备计算输入",
    "select_dft_candidates": "筛选并生成 DFT 输入", "reevaluate_candidates": "刷新结构能量",
    "check_convergence": "检查收敛", "pause_search": "暂停搜索",
    "adjust_strategy": "修订策略",
}
STAGE_NAMES = {"relax_and_feature": "Relax", "deep_search": "MC",
               "dft_single_point": "DFT 单点", "dft_relax": "DFT 优化"}


def format_progress(state):
    model = state.get("active_model_version") or "未记录"
    pending = state.get("pending_execution_policies") or {}
    tasks = state.get("tasks") or []
    training = waiting_training_job(state)
    waiting = [row for row in tasks if awaiting_task_result(row)]
    waived = [row for row in tasks if row.get("status") in {"pending", "running", "submitted", "unknown"}
              and not awaiting_task_result(row)]
    lines = [f"当前项目状态：{model_epoch_label(state, model)}。",
             brief_chat_state(state)]
    from execution_layer.local.recover_remote_training import inspect_training_results, training_result_message, INACTIVE
    returned = [inspect_training_results(job) for job in (state.get("remote_finetune_jobs") or {}).values()
                if job.get("status") not in INACTIVE and not job.get("activated")]
    returned = [row for row in returned if row is not None]
    if returned:
        return "\n".join([lines[0], *[training_result_message(row) for row in returned],
                          "本次仅检查回传文件，未登记、分析或激活；说“继续”由Agent回收。"])
    if len(pending) == 1:
        record_id, record = next(iter(pending.items()))
        proposal = record.get("agent_proposal") or {}
        action = proposal.get("raw_action") or {}
        tool = action.get("tool") or proposal.get("recommended_action")
        name = ACTION_NAMES.get(tool, tool or "待审批操作")
        lines += [f"待审批方案 1 个：{name}。",
                  "下一步：查看该方案后回复“同意”“拒绝”或提出修改；本次查询不会批准。"]
        if detailed_response():
            lines.append(f"方案编号：{record_id}。")
            purpose = proposal.get("expected_purpose") or action.get("expected_purpose")
            if purpose:
                lines.append(f"目的：{purpose}")
    elif pending:
        lines += [f"待审批方案 {len(pending)} 个。", "下一步：先明确审批对象，不要笼统批准。"]
    elif state.get("pending_dft_recovery_question"):
        lines.append("下一步：回复“继续回收”或“不再回收”，确认是否等待剩余 DFT。")
    elif training:
        lines.append("下一步：上传完整训练轮目录，在inputs内提交 GPU.sh（旧布局在轮根）；回传该轮results，再说“继续”。模型仍需验证和单独激活审批。"
                     if training.get("status") == "inputs_prepared" else
                     "下一步：回传训练结果后说“继续”，检查通过后再单独批准模型激活。")
        lines.append(f"训练目录：{training.get('directory') or '未记录'}。")
    elif waiting:
        lines.append("下一步：回传对应 results 后说“继续”；尚未回传的任务继续等待。")
    else:
        lines.append("下一步：说“继续”，由 Agent 检查已有结果并提出下一步方案，不等于批准。")
    by_stage = Counter(STAGE_NAMES.get(row.get("stage"), row.get("stage") or "未分类")
                       for row in waiting)
    completed = sum(row.get("status") == "completed" for row in tasks)
    failed = sum(row.get("status") in {"failed", "timeout"} for row in tasks)
    wait_text = "、".join(f"{name} {count}" for name, count in by_stage.items()) or "0"
    lines.append(f"累计任务：完成 {completed}、失败/超时 {failed}；仍需回收 {wait_text}；未回传但不再等待 {len(waived)}。")
    remaining = state.get("budget_remaining")
    if isinstance(remaining, (float, int)) and not isinstance(remaining, bool):
        lines.append(f"剩余相对预算约 {remaining:.1f}。")
    if detailed_response():
        lines.append(f"配置：{state.get('confirmed_config_version') or '未记录'}。")
        for method, snapshot in (state.get("phase_diagrams") or {}).items():
            lines.append(f"{method} 相图：{snapshot.get('version') or '未记录'}；文件：{snapshot.get('csv_path') or '未记录'}。")
        lines.append(f"预留相对成本：{float(state.get('reserved_relative_cost') or 0):.1f}。")
    lines.append("仅查看已记录状态；未回收、分析或执行任务。")
    return "\n".join(lines)

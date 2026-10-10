"""Factual progress display; no workflow execution, analysis or state mutation."""

from collections import Counter
import json
from pathlib import Path
from phase_agent.analysis.state.model_epoch import model_epoch_label
from phase_agent.tools.state.task_waiting import awaiting_task_result
from phase_agent.runtime.chat_state_presentation import brief_chat_state, waiting_training_job
from phase_agent.runtime.response_preferences import detailed_response

ACTION_NAMES = {
    "update_mlip": "生成微调训练输入",
    "generate_branches": "生成 branch",
    "allocate_mc_bohb": "分配 MC",
    "prepare_local_batch_files": "准备计算输入",
    "select_dft_candidates": "筛选并生成 DFT 输入",
    "reevaluate_candidates": "刷新结构能量",
    "check_convergence": "检查收敛",
    "pause_search": "暂停搜索",
    "adjust_strategy": "修订策略",
}
STAGE_NAMES = {
    "relax_and_feature": "Relax",
    "deep_search": "MC",
    "dft_single_point": "DFT 单点",
    "dft_relax": "DFT 优化",
}


def format_progress(state):
    model = state.get("active_model_version") or "未记录"
    pending = state.get("pending_execution_policies") or {}
    tasks = state.get("tasks") or []
    training = waiting_training_job(state)
    waiting = [row for row in tasks if awaiting_task_result(row)]
    waived = [
        row
        for row in tasks
        if row.get("status") in {"pending", "running", "submitted", "unknown"}
        and not awaiting_task_result(row)
    ]
    lines = [f"当前项目状态：{model_epoch_label(state, model)}。", brief_chat_state(state)]
    from phase_agent.tools.local.recover_remote_training import (
        inspect_training_results,
        training_result_message,
        INACTIVE,
    )

    returned = [
        inspect_training_results(job)
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("status") not in INACTIVE | {"results_received"} and not job.get("activated")
    ]
    returned = [row for row in returned if row is not None]
    handoffs = [
        job.get("training_handoff")
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("training_handoff")
        and job.get("status") not in INACTIVE
        and not job.get("activated")
    ]
    if handoffs:
        reasons = []
        for job in (state.get("remote_finetune_jobs") or {}).values():
            handoff = job.get("training_handoff")
            if not handoff or job.get("status") in INACTIVE or job.get("activated"):
                continue
            from phase_agent.runtime.training_reply import concise_training_reply

            concise = concise_training_reply(handoff, state)
            if concise:
                reasons.append(concise)
                continue
            if handoff.get("stage") in {
                "awaiting_direction_approval",
                "validation_prerequisites_required",
                "agent_review_required",
            }:
                version = handoff.get("candidate_model_version")
                candidate = (state.get("candidate_models") or {}).get(version) or {}
                review = candidate.get("agent_review") or {}
                if review:
                    reasons.append("当前阶段：微调后策略评估，尚未切换模型。")
                    reasons.append("Agent建议：" + str(review.get("reason") or "")[:220])
                    if handoff.get("stage") == "awaiting_direction_approval":
                        reasons.append("下一步：回复‘同意’批准已展示的方向和计划，或‘拒绝’取消。")
                    else:
                        reasons.append(
                            "下一步：回复‘继续’，由Agent补齐或校验具体方案；不会自动执行计算。"
                        )
                    continue
                reasons.append(
                    "当前阶段：微调已完成，等待Agent评估下一步。\n下一步：回复‘继续’，比较换模型、补DFT或核查方案。"
                )
                continue
            report_path = Path(job["directory"]) / "results" / "cv_baseline_review.json"
            if (
                handoff.get("stage") == "validation_prerequisites_required"
                and report_path.is_file()
            ):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                saved = job.get("returned_results") or {}
                if report.get("training_fingerprint") == saved.get("fingerprint"):
                    reasons.append(
                        f"模型清单已齐。K折对比报告：{report_path}；匹配结构 {report['structures']}，力分量 {report['force_components']}。"
                    )
                    for name, unit in (("energy_per_atom", "meV/atom"), ("forces", "meV/Å")):
                        old, new = report["old"].get(name), report["new"].get(name)
                        if old and new and old.get("mae") is not None:
                            reasons.append(
                                f"{name} MAE/RMSE：旧模型 {old['mae'] * 1000:.2f}/{old['rmse'] * 1000:.2f} → K折 {new['mae'] * 1000:.2f}/{new['rmse'] * 1000:.2f} {unit}。"
                            )
                    reasons.append(
                        "下一步方案："
                        + report["next_proposal"]
                        + " K折报告不等同于最终模型独立测试，未激活。"
                    )
                    continue
            reasons.append(handoff["reason"])
        return "\n".join([lines[0], *reasons, "本次仅查看状态，未执行或批准。"])
    if returned:
        return "\n".join(
            [
                lines[0],
                *[training_result_message(row) for row in returned],
                "本次仅检查回传文件，未登记、分析或激活；说“继续”由Agent回收。",
            ]
        )
    recovered = [
        job.get("returned_results") or {}
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("status") == "results_received" and not job.get("activated")
    ]
    if recovered:
        lines.append("已登记微调回收结果；下一步检查模型清单和验证记录，无需重复生成训练输入。")
        for result in recovered:
            issues = result.get("issues") or []
            if issues:
                lines.append("当前缺项：" + "；".join(str(issue) for issue in issues) + "。")
            if result.get("kfold_metrics"):
                lines.append("交叉验证指标已登记。")
        if any(result.get("issues") for result in recovered):
            lines.append(
                "执行方案：补齐模型清单，无需重训。回复‘继续’，生成 GPU_manifest.sh 和 collect_training_results.py；上传至超算本轮原 inputs 目录，执行 sbatch GPU_manifest.sh，完成后回传 results/models.json，再回复‘继续’检查验证条件。"
            )
        else:
            lines.append(
                "执行方案：回复‘继续’，检查独立验证数据与标准并准备验证作业；验证通过后单独审批激活。"
            )
        if pending:
            lines.append(
                f"另有待审批方案 {len(pending)} 个；该方案不代表微调尚未回收。重新训练方案需核对必要性后再处理。"
            )
        lines.append("本次仅查看记录，未重复训练、批准或激活模型。")
        return "\n".join(lines)
    if len(pending) == 1:
        record_id, record = next(iter(pending.items()))
        proposal = record.get("agent_proposal") or {}
        action = proposal.get("raw_action") or {}
        tool = action.get("tool") or proposal.get("recommended_action")
        name = ACTION_NAMES.get(tool, tool or "待审批操作")
        lines += [
            f"待审批方案 1 个：{name}。",
            "下一步：查看该方案后回复“同意”“拒绝”或提出修改；本次查询不会批准。",
        ]
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
        lines.append(
            "下一步：上传完整训练轮目录，在inputs内提交 GPU.sh（旧布局在轮根）；回传该轮results，再说“继续”。模型仍需验证和单独激活审批。"
            if training.get("status") == "inputs_prepared"
            else "下一步：回传训练结果后说“继续”，检查通过后再单独批准模型激活。"
        )
        lines.append(f"训练目录：{training.get('directory') or '未记录'}。")
    elif waiting:
        lines.append("下一步：回传对应 results 后说“继续”；尚未回传的任务继续等待。")
    else:
        lines.append("下一步：说“继续”，由 Agent 检查已有结果并提出下一步方案，不等于批准。")
    by_stage = Counter(
        STAGE_NAMES.get(row.get("stage"), row.get("stage") or "未分类") for row in waiting
    )
    completed = sum(row.get("status") == "completed" for row in tasks)
    failed = sum(row.get("status") in {"failed", "timeout"} for row in tasks)
    wait_text = "、".join(f"{name} {count}" for name, count in by_stage.items()) or "0"
    lines.append(
        f"累计任务：完成 {completed}、失败/超时 {failed}；仍需回收 {wait_text}；未回传但不再等待 {len(waived)}。"
    )
    remaining = state.get("budget_remaining")
    if isinstance(remaining, (float, int)) and not isinstance(remaining, bool):
        lines.append(f"剩余相对预算约 {remaining:.1f}。")
    if detailed_response():
        lines.append(f"配置：{state.get('confirmed_config_version') or '未记录'}。")
        for method, snapshot in (state.get("phase_diagrams") or {}).items():
            lines.append(
                f"{method} 相图：{snapshot.get('version') or '未记录'}；文件：{snapshot.get('csv_path') or '未记录'}。"
            )
        lines.append(f"预留相对成本：{float(state.get('reserved_relative_cost') or 0):.1f}。")
    lines.append("仅查看已记录状态；未回收、分析或执行任务。")
    return "\n".join(lines)

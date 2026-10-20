"""Read-only recovered-round metrics before the next scientific proposal."""

from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment
from phase_agent.analysis.state.model_epoch import model_epoch_label


def _metric_pair(metrics, unit):
    if metrics.get("mae") is None or metrics.get("rmse") is None:
        return "暂不可用（缺少有效同帧配对）"
    return f"{1000 * metrics['mae']:.5g}/{1000 * metrics['rmse']:.5g} {unit}"


def reply_assessment(result, review_action):
    """History is evidence only for an explicit recovery/review in this turn."""
    rows = [result, *(result.get("events") or [])]
    if any(row.get("final_action") for row in rows):
        return None
    if not (
        review_action.get("post_dft_review")
        or result.get("dft_recovery_question")
        or result.get("recovered_count")
        or (result.get("manual_wait") or {}).get("recovered_count")
        # The dialogue entry recovers files before the workflow starts, so its
        # workflow recovered_count can be zero despite a verified DFT return.
        or any(
            counts.get("completed") or counts.get("failed")
            for stage, counts in (result.get("verified_recovery") or {}).items()
            if stage in {"dft_single_point", "dft_relax"}
        )
    ):
        return None
    state = result.get("state") or {}
    return post_dft_assessment(state, state.get("confirmed_config") or {})


def post_dft_lines(state, *, assessment=None):
    if assessment is None:
        assessment = post_dft_assessment(state, state.get("confirmed_config") or {})
    if not assessment:
        return ""
    metrics = assessment["metrics"]
    energy, forces = metrics["energy_per_atom"], metrics["forces"]
    label = model_epoch_label(state, assessment["scope"].get("model_version"))
    lines = [
        f"结果回收与分析：{label}",
        f"本轮 DFT：回收 {assessment['recovered_tasks']}/{assessment['expected_tasks']}，合格配对 {assessment['paired_structures']}。",
        f"成功 {assessment['successful_tasks']}、失败 {assessment['failed_tasks']}、已取消 {assessment.get('cancelled_tasks', 0)}；取消项不计DFT验证，不再回收。",
        f"原模型 {assessment['scope'].get('model_version')}：能量 MAE/RMSE {_metric_pair(energy, 'meV/atom')}；受力 MAE/RMSE {_metric_pair(forces, 'meV/Å')}。"
        + (
            "配对不完整，仅统计已有合格配对；缺失项不计为零。"
            if assessment["status"] != "evaluated"
            else ""
        ),
    ]
    export = next(
        (
            row
            for row in (state.get("dft_result_exports") or {}).values()
            if all(
                row.get("round_scope", {}).get(key) == value
                for key, value in assessment["scope"].items()
            )
        ),
        None,
    )
    if export:
        lines.append(
            f"数据：`{export['directory']}`（comparisons：对角线与误差；training：训练数据；diagnostics：磁矩）。"
        )
        plots = export.get("parity_plots") or {}
        if plots.get("status") == "completed":
            lines.append(
                f"对角线图（MAE/RMSE，能量meV/atom、受力meV/Å）：`{export['directory']}/comparisons/plots`。"
            )
        elif plots.get("reason"):
            lines.append(f"对角线图未生成：{plots['reason']}")
        if (plots.get("diagnostics") or {}).get("warning"):
            lines.append("受力检查提醒：" + plots["diagnostics"]["warning"])
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("csv_path"):
        lines.append(f"MLIP 预测相图：`{snapshot['csv_path']}`；DFT 用于检验与微调。")
    else:
        lines.append("MLIP 预测相图：尚无已保存 CSV；DFT 用于检验与微调。")
    return "\n".join(lines)


def post_dft_review_lines(action, *, verbose=False, finetune_enabled=None):
    review = (action or {}).get("post_dft_review")
    if not isinstance(review, dict):
        return ""
    fields = (
        ("round_findings", "本轮发现与收益"),
        ("error_assessment", "误差判断"),
        ("coverage_assessment", "覆盖判断"),
        ("limitations", "数据局限"),
        ("finetune_assessment", "微调取舍"),
        ("search_assessment", "搜索取舍"),
        ("dft_assessment", "补DFT取舍"),
        ("convergence_assessment", "收敛与停止"),
        ("reference_assessment", "参考意见"),
    )
    if not verbose:
        fields = (
            ("round_findings", "本轮发现"),
            ("finetune_assessment", "微调取舍"),
            ("dft_assessment", "补DFT取舍"),
            ("search_assessment", "新branch取舍"),
            ("convergence_assessment", "收敛与停止"),
            ("limitations", "数据局限"),
        )
    recommendation = {
        "now": "建议微调",
        "defer": "建议暂缓微调",
        "insufficient_evidence": "证据不足，暂不能判断是否微调",
    }.get(review.get("finetune_recommendation"))
    lines = []
    choice = {
        "search": "新增branch",
        "finetune": "直接微调",
        "supplement_dft": "补充DFT",
        "convergence": "评估收敛",
        "stop": "停止",
        "revise_strategy": "修订执行配置",
    }.get(review.get("choice"))
    if choice:
        from phase_agent.tools.policy.training_input_action import is_training_input_action

        if is_training_input_action(action):
            choice = "微调：生成超算训练提交文件"
        lines.append(f"首选建议：{choice}。")
    for key, label in fields:
        if review.get(key):
            if key == "finetune_assessment" and recommendation:
                lines.append(f"微调取舍：{recommendation}。原因：{review[key]}")
            else:
                lines.append(f"{label}：{review[key]}")
    if review.get("finetune_recommendation") == "now" and finetune_enabled is False:
        lines.append(
            "执行条件：批准后仅生成超算训练提交文件；当前未训练，不修改模型启用或激活配置。"
        )
    return "本轮总结：\n" + "\n".join(lines)

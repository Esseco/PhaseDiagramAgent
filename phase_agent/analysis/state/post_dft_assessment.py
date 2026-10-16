"""Round-scoped evidence for the decision after DFT recovery, not a retrain rule."""

from phase_agent.analysis.state.dft_round_status import dft_recovery_rounds
from phase_agent.analysis.feedback.dft_comparison_tables import (
    build_comparison_tables,
    comparison_metrics,
)


def post_dft_assessment(state, config=None):
    if any(
        task.get("stage") in {"dft_single_point", "dft_relax"}
        and task.get("status")
        in {"pending", "running", "submitted", "unknown", "submission_unknown"}
        for task in state.get("tasks") or []
    ):
        return None
    config = config or {}
    model = config.get("mlip") or {}
    version = state.get("active_model_version") or model.get("version") or model.get("name")
    consumed = set(state.get("post_dft_decided_rounds") or [])
    rounds = [
        row
        for row in dft_recovery_rounds(state)
        if row["recovered_tasks"]
        and not row["pending_task_ids"]
        and not row["waived_task_ids"]
        and row["resolved_tasks"] == row["expected_tasks"]
        and row["scope_key"] not in consumed
        and (not version or row["scope"].get("model_version") == version)
    ]
    if not rounds:
        return None
    row = rounds[-1]
    ids = set(row["recovered_task_ids"])
    records = [
        item for item in state.get("dft_dataset_records") or [] if item.get("task_id") in ids
    ]
    comparisons = [
        item for item in state.get("dft_mlip_comparisons") or [] if item.get("task_id") in ids
    ]
    energies, forces = build_comparison_tables(records, comparisons, round_name=row["scope_key"])
    metrics = comparison_metrics(energies, forces, row["scope"])
    eligible = [
        item
        for item in records
        if item.get("status") == "completed"
        and item.get("converged") is True
        and item.get("checks_passed") is True
        and item.get("training_ready") is True
    ]
    paired = sum(item.get("comparison_status") == "completed" for item in energies)
    from phase_agent.tools.policy.workflow_policy import settings

    minimum = settings(config)["finetune_min_new_dft"]
    export = next(
        (
            item
            for item in (state.get("dft_result_exports") or {}).values()
            if item.get("round_scope") == row["scope"]
        ),
        {},
    )
    return {
        **row,
        "status": "evaluated" if eligible and paired == len(eligible) else "evaluation_incomplete",
        "parity_plots": export.get("parity_plots") or {},
        "remote_finetune_jobs": [
            {
                key: job.get(key)
                for key in (
                    "directory",
                    "status",
                    "original_model_version",
                    "submitted",
                    "activated",
                )
            }
            for job in (state.get("remote_finetune_jobs") or {}).values()
        ],
        "eligible_structures": len(eligible),
        "failure_review": {
            "failure_fraction": row["failed_tasks"] / row["recovered_tasks"]
            if row["recovered_tasks"]
            else None,
            "majority_failed": row["failed_tasks"] > row["recovered_tasks"] * 0.5,
            "instruction": "个别DFT失败直接放弃；超过50%结合共同错误汇报用户，不自动重试或修改参数。未回传不算失败。",
        },
        "paired_structures": paired,
        "metrics": metrics,
        "minimum_new_dft_records": minimum,
        "mlip_phase_diagram": {
            key: ((state.get("phase_diagrams") or {}).get("mlip") or {}).get(key)
            for key in (
                "status",
                "version",
                "model_version",
                "excluded_structures",
                "reason",
                "csv_path",
            )
        },
        "finetune_enabled": (config.get("mlip_finetune") or {}).get("enabled") is True,
        "decision_requirement": "比较直接微调、已有结构池补DFT、新branch、收敛/停止四路，给出首选和其他方向暂不选的理由。人工经验将过大的受力误差视为重要微调依据；数据不足仍比较两条路径，补DFT与新branch按包含后续计算的预期增量收益和成本比较，覆盖和误差不作硬门槛。微调未启用只是执行条件。",
        "instruction": "根据本轮与历史结果、原模型能量/力MAE及RMSE、相图变化、覆盖、收益、代表性、预算和记忆决策；不硬编码科学结论。区分科学收敛与预算/条件暂停，无新稳定相不等于收敛。参数错误不丢弃有效分析，不重复派发旧DFT，不自动训练或激活模型。",
    }

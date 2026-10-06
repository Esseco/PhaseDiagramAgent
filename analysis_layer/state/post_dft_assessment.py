"""Round-scoped evidence for the decision after DFT recovery, not a retrain rule."""
from analysis_layer.state.dft_round_status import dft_recovery_rounds
from analysis_layer.feedback.dft_comparison_tables import build_comparison_tables, comparison_metrics


def post_dft_assessment(state, config=None):
    config = config or {}
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name") or state.get("active_model_version")
    consumed = set(state.get("post_dft_decided_rounds") or [])
    rounds = [row for row in dft_recovery_rounds(state)
              if row["recovered_tasks"] and not row["pending_task_ids"]
              and row["scope_key"] not in consumed
              and (not version or row["scope"].get("model_version") == version)]
    if not rounds:
        return None
    row = rounds[-1]
    ids = set(row["recovered_task_ids"])
    records = [item for item in state.get("dft_dataset_records") or [] if item.get("task_id") in ids]
    comparisons = [item for item in state.get("dft_mlip_comparisons") or [] if item.get("task_id") in ids]
    energies, forces = build_comparison_tables(records, comparisons, round_name=row["scope_key"])
    metrics = comparison_metrics(energies, forces, row["scope"])
    eligible = [item for item in records if item.get("status") == "completed"
                and item.get("converged") is True and item.get("checks_passed") is True
                and item.get("training_ready") is True]
    paired = sum(item.get("comparison_status") == "completed" for item in energies)
    training = (config.get("mlip_finetune") or {}).get("training") or {}
    minimum = training.get("minimum_new_dft_records", ((config.get("qbc") or {}).get("retrain") or {}).get("minimum_new_dft_records", 10))
    return {**row, "status": "evaluated" if eligible and paired == len(eligible) else "evaluation_incomplete",
            "eligible_structures": len(eligible), "paired_structures": paired, "metrics": metrics,
            "minimum_new_dft_records": minimum,
            "finetune_enabled": (config.get("mlip_finetune") or {}).get("enabled") is True,
            "instruction": "本轮DFT回收已结束：先分析本轮原模型的能量/受力MAE、RMSE、相/Na覆盖和记忆，再决定微调或生成新branch。缺失误差不能视为零，先说明并补齐模型/评估接口；数据数量门槛不是科学误差阈值。不得重复派发本轮旧DFT，不自动训练或激活模型。"}

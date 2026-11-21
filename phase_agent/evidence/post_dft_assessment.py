"""Round-scoped evidence for the decision after DFT recovery, not a retrain rule."""

from phase_agent.evidence.dft_round_status import dft_recovery_rounds
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

    tasks_by_id = {t.get("task_id"): t for t in state.get("tasks") or [] if t.get("task_id")}
    gs_hashes = {
        item.get("structure_sha256")
        for batch in (state.get("remote_dft_sampling") or {}).values()
        for item in batch.get("selected") or []
        if item.get("selection_mode") == "phase_diagram_gs" and item.get("structure_sha256")
    }

    def _is_gs_task(task_id):
        task = tasks_by_id.get(task_id) or {}
        if task.get("is_ground_state") is True or task.get("selection_mode") == "phase_diagram_gs":
            return True
        sha = task.get("input_structure_sha256")
        return bool(sha and sha in gs_hashes)

    gs_ids = {
        item.get("task_id")
        for item in records
        if _is_gs_task(item.get("task_id"))
        or item.get("is_ground_state") is True
        or item.get("selection_mode") == "phase_diagram_gs"
    }
    gs_records = [item for item in records if item.get("task_id") in gs_ids]
    gs_comparisons = [item for item in comparisons if item.get("task_id") in gs_ids]
    sample_records = [item for item in records if item.get("task_id") not in gs_ids]
    sample_comparisons = [item for item in comparisons if item.get("task_id") not in gs_ids]
    gs_e, gs_f = build_comparison_tables(
        gs_records, gs_comparisons, round_name=row["scope_key"] + ":GS"
    )
    sample_e, sample_f = build_comparison_tables(
        sample_records, sample_comparisons, round_name=row["scope_key"] + ":sample"
    )
    gs_metrics = comparison_metrics(gs_e, gs_f, row["scope"])
    sample_metrics = comparison_metrics(sample_e, sample_f, row["scope"])
    gs_paired = sum(item.get("comparison_status") == "completed" for item in gs_e)
    sample_paired = sum(item.get("comparison_status") == "completed" for item in sample_e)
    eligible = [
        item
        for item in records
        if item.get("status") == "completed"
        and item.get("converged") is True
        and item.get("checks_passed") is True
        and item.get("training_ready") is True
    ]
    paired = sum(item.get("comparison_status") == "completed" for item in energies)
    from phase_agent.policies.workflow_policy import settings

    minimum = settings(config)["finetune_min_new_dft"]
    untrained_valid_ids = {
        item.get("task_id") or item.get("structure_id")
        for item in state.get("dft_dataset_records") or []
        if isinstance(item, dict)
        and item.get("status") == "completed"
        and item.get("converged") is True
        and item.get("checks_passed") is True
        and item.get("training_ready") is True
        and (not version or item.get("model_version") == version)
    } - {None}
    conv = config.get("convergence") or {}
    energy_tol = float(conv.get("final_energy_mae_tolerance") or 0.005)
    force_tol = float(conv.get("final_force_rmse_tolerance") or 0.05)

    def _metric_val(metric_dict, name, stat):
        entry = (metric_dict or {}).get(name)
        return entry.get(stat) if isinstance(entry, dict) else None

    gs_e_mae = _metric_val(gs_metrics, "energy_per_atom", "mae")
    gs_f_rmse = _metric_val(gs_metrics, "forces", "rmse")
    sample_e_mae = _metric_val(sample_metrics, "energy_per_atom", "mae")
    sample_f_rmse = _metric_val(sample_metrics, "forces", "rmse")
    gs_error_high = bool(
        (gs_e_mae is not None and gs_e_mae > energy_tol)
        or (gs_f_rmse is not None and gs_f_rmse > force_tol)
    )
    sample_error_high = bool(
        (sample_e_mae is not None and sample_e_mae > energy_tol)
        or (sample_f_rmse is not None and sample_f_rmse > force_tol)
    )
    untrained_count = len(untrained_valid_ids)
    data_sufficient = untrained_count >= minimum
    urgent_gs_finetune = bool(gs_error_high and untrained_count >= min(minimum, 20))
    gs_converged = bool(gs_paired > 0 and not gs_error_high)
    gs_only_mode = bool(conv.get("gs_only_convergence_mode", True))
    if urgent_gs_finetune or (
        (gs_error_high or (sample_error_high and not gs_only_mode)) and data_sufficient
    ):
        route_hint = "finetune"
    elif gs_converged and gs_only_mode:
        route_hint = "explore_or_converge"
    elif gs_error_high or sample_error_high:
        route_hint = "accumulate_dft_or_explore"
    else:
        route_hint = "explore_or_converge"
    from phase_agent.analysis.phase.mp_global_ehull import (
        CompetingPhaseRepo,
        compute_gs_global_dft_ehull,
    )

    mp_cfg = config.get("mp_competing_phases") or {}
    competing_entries = list(state.get("mp_competing_dft_entries") or [])
    repo_path = mp_cfg.get("shared_repo_path")
    if not competing_entries and mp_cfg.get("enabled", True) and repo_path:
        elements = {str(el) for rec in gs_records for el in (rec.get("composition") or {})}
        if elements:
            try:
                competing_entries = CompetingPhaseRepo(repo_path).lookup(
                    elements,
                    (config.get("dft") or {}).get("parameters") or {},
                    max_arity=int(mp_cfg.get("max_arity", 2)),
                )
            except (OSError, ValueError):
                competing_entries = []
    if competing_entries and gs_records:
        gs_global_dft_ehull = compute_gs_global_dft_ehull(gs_records, competing_entries)
    else:
        gs_global_dft_ehull = {"status": "not_configured_or_missing_competing_phases"}
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
        "untrained_valid_dft_count": untrained_count,
        "post_dft_decision_signals": {
            "energy_mae_tolerance": energy_tol,
            "force_rmse_tolerance": force_tol,
            "gs_energy_mae": gs_e_mae,
            "gs_force_rmse": gs_f_rmse,
            "sample_energy_mae": sample_e_mae,
            "sample_force_rmse": sample_f_rmse,
            "gs_error_high": gs_error_high,
            "sample_error_high": sample_error_high,
            "gs_converged": gs_converged,
            "gs_only_convergence_mode": gs_only_mode,
            "data_sufficient_for_finetune": data_sufficient,
            "urgent_gs_finetune": urgent_gs_finetune,
            "recommended_route_hint": route_hint,
            "sampling_buffer_target": "60_to_100_frames_kmeans_pp_fps",
            "convergence_rule": "本轮GS基态能量与受力误差均低于阈值且给定边界探索充分即可触发终验静态计算并收敛，减少不必要迭代。",
        },
        "failure_review": {
            "failure_fraction": row["failed_tasks"] / row["recovered_tasks"]
            if row["recovered_tasks"]
            else None,
            "majority_failed": row["failed_tasks"] > row["recovered_tasks"] * 0.5,
            "instruction": "少量DFT失败直接排除；单批失败超过10个或占已回传计算超过20%时交由Simulation调查并报告，不自动重试或修改参数。未回传不算失败。",
        },
        "paired_structures": paired,
        "metrics": metrics,
        "gs_paired_structures": gs_paired,
        "gs_metrics": gs_metrics,
        "gs_global_dft_ehull": gs_global_dft_ehull,
        "sample_paired_structures": sample_paired,
        "sample_metrics": sample_metrics,
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

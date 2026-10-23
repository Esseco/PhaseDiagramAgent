"""Validate a concrete DFT input plan before human approval."""

from copy import deepcopy
import math
from phase_agent.science.qbc.build_candidate_metrics import build_qbc_candidate_metrics
from phase_agent.decisions.qbc_selection.dft_round_policy import dft_task_limit, round_dft_tasks
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def stale_dft_preview(proposal):
    tool = proposal.get("recommended_action") or (proposal.get("raw_action") or {}).get("tool")
    if tool != "select_dft_candidates":
        return False
    params = (
        proposal.get("action_parameters")
        or (proposal.get("raw_action") or {}).get("parameters")
        or {}
    )
    preview = params.get("dft_input_preview") or {}
    return preview.get("schema_version") != 2 or not preview.get("selected_structures")


def preview_dft_inputs(action, state, config):
    decisions = (action.get("parameters") or {}).get("decisions")
    if (
        not isinstance(decisions, list)
        or not decisions
        or any(not isinstance(row, dict) for row in decisions)
    ):
        return None, "DFT 方案为空或格式无效；请重新提出具体结构优化方案。"
    selected = [row for row in decisions if row.get("action") in {"DFT_RELAX", "DFT_SINGLE_POINT"}]
    if not selected:
        return None, "未选择实际 DFT 结构；不能生成输入文件。"
    try:
        max_tasks = dft_task_limit(config)
    except ValueError as error:
        return None, f"DFT 数量配置无效：{error}"
    prior = round_dft_tasks(state, config)
    if len(selected) + len(prior) > max_tasks:
        return None, (
            f"本轮 DFT 任务超过上限（单点与优化合计最多 {max_tasks} 个，"
            f"已登记 {len(prior)} 个，本次 {len(selected)} 个），请按需要减少选点后重新预览。"
        )
    qbc = deepcopy(config.get("qbc") or {})
    qbc["selection_policy"] = deepcopy((config.get("dft") or {}).get("selection") or {})
    qbc["budget_limits"] = deepcopy(config.get("budgets") or {})
    candidates = state.get("qbc_candidates") or []
    candidate_map = {row.get("candidate_id"): row for row in candidates}
    selected_rows = [candidate_map.get(row.get("candidate_id")) or {} for row in selected]
    available_phases = {row.get("phase") for row in candidates if row.get("phase")}
    selected_phases = {row.get("phase") for row in selected_rows if row.get("phase")}
    warnings = []
    if available_phases - selected_phases:
        warnings.append(
            "未覆盖相："
            + "、".join(sorted(available_phases - selected_phases))
            + "；请核对LLM取舍理由。"
        )
    metrics = build_qbc_candidate_metrics(candidates)["metrics"]
    try:
        validation = validate_dft_agent_decisions(
            {"decisions": decisions, "source": action.get("decision_source", "llm_agent")},
            metrics,
            state,
            config=qbc,
            config_version=state.get("confirmed_config_version"),
            remaining_budget=float("inf"),
        )
    except (KeyError, TypeError, ValueError) as error:
        return None, f"DFT 成本或筛选配置不完整：{error}"
    warnings.extend(
        reason
        for row in validation.get("accepted") or []
        for reason in row.get("budget_warnings") or []
    )
    accepted = [
        row for row in validation["accepted"] if row["action"] in {"DFT_RELAX", "DFT_SINGLE_POINT"}
    ]
    relax_rejected = [
        row
        for row in validation["rejected"]
        if row.get("reason") == "dft_relax_reserved_for_limited_geometry_checks"
    ]
    if relax_rejected:
        fraction = float(qbc["selection_policy"].get("max_relax_fraction", 0.10))
        return None, (
            f"当前配置只允许最多 {fraction:.0%} 的 DFT 候选做结构优化，"
            f"本次 {len(selected)} 个优化候选中 {len(relax_rejected)} 个被限制。"
            "请按单点优先策略重新分配，不能自动转换计算类型。未生成任务。"
        )
    if not validation["valid"] or validation["rejected"] or len(accepted) != len(selected):
        return None, "DFT 方案未通过筛选/预算校验，需修订后再批准：" + str(
            validation.get("errors") or validation.get("rejected")
        )
    from phase_agent.tools.workflows.dft_coverage_review import coverage_errors

    missing_review = coverage_errors(action, candidates)
    if missing_review:
        return None, "DFT 的 D/H 独立疑点未完整审阅；增加或替换代表，或明确暂缓理由：" + "; ".join(
            missing_review
        )
    total = sum(row["relative_cost"] for row in accepted)
    if not math.isfinite(total) or total <= 0:
        return None, "DFT 成本尚未确定，不能以零成本批准。"
    return {
        "schema_version": 2,
        "task_count": len(accepted),
        "relative_cost": total,
        "warnings": warnings,
        "workload": [
            {
                "stage": stage,
                "tasks": sum(row["action"] == name for row in accepted),
                "initial_cost": sum(
                    row["relative_cost"] for row in accepted if row["action"] == name
                ),
            }
            for name, stage in (
                ("DFT_SINGLE_POINT", "dft_single_point"),
                ("DFT_RELAX", "dft_relax"),
            )
            if any(row["action"] == name for row in accepted)
        ],
        "single_point_count": sum(row["action"] == "DFT_SINGLE_POINT" for row in accepted),
        "relax_count": sum(row["action"] == "DFT_RELAX" for row in accepted),
        "candidate_ids": [row["candidate_id"] for row in accepted],
        "selected_structures": [
            {
                "candidate_id": row.get("candidate_id"),
                "x_Na_per_O2": row.get("x_Na_per_O2"),
                "phase": row.get("phase"),
                "ehull": row.get("predicted_Ehull"),
                "ehull_unit": "eV/atom",
            }
            for row in selected_rows
        ],
        "qbc_available_count": sum(
            not str((row.get("qbc") or {}).get("status", "")).startswith(("not_", "failed"))
            and any(
                isinstance((row.get("qbc") or {}).get(key), (int, float))
                for key in ("f_std_max", "force_max_atom_disagreement", "energy_per_atom_std")
            )
            for row in selected_rows
        ),
        "reasons": [row.get("reason") for row in selected],
        "calculation_type": "单点/结构优化",
    }, None

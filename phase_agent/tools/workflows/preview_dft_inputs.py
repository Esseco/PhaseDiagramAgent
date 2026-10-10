"""Validate a concrete DFT input plan before human approval."""

from copy import deepcopy
import math
from phase_agent.science.qbc.build_candidate_metrics import build_qbc_candidate_metrics
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
    total = sum(row["relative_cost"] for row in accepted)
    selection = (config.get("dft") or {}).get("selection") or {}
    single_points = [row for row in accepted if row["action"] == "DFT_SINGLE_POINT"]
    max_points = int(selection.get("single_point_max_per_round", 100))
    max_point_cost = float(selection.get("single_point_cost_per_round", 5000))
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name") or state.get("active_model_version")
    prior = [
        row
        for row in state.get("tasks") or []
        if row.get("stage") == "dft_single_point"
        and row.get("model_version") == version
        and row.get("generation_cycle") == len(state.get("generation_history") or [])
        and row.get("status") != "cancelled"
    ]
    if (
        len(single_points) + len(prior) > max_points
        or sum(row["relative_cost"] for row in single_points)
        + sum(float(row.get("planned_relative_cost") or 0) for row in prior)
        > max_point_cost
    ):
        return (
            None,
            f"本轮单点方案超过上限（最多 {max_points} 个、相对成本 {max_point_cost:g}），请按相/Na/QBC分层调整后重新预览。",
        )
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

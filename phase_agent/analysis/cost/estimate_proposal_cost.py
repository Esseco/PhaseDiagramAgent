"""Describe proposed workload using initial estimates calibrated by actual costs."""

from __future__ import annotations
from phase_agent.analysis.cost.calibrate_relative_cost import calibrate_relative_cost


def estimate_proposal_cost(action, state):
    tool = action.get("tool")
    params = action.get("parameters") or {}
    stages = []
    if tool == "prepare_local_batch_files" and params.get("mode") == "model_refresh_inputs":
        preview = params.get("refresh_preview") or {}
        return {
            "workload": [
                {
                    "stage": "relax_and_feature",
                    "tasks": len(preview.get("candidates") or []),
                    "maximum_supplemental_tasks": preview.get("maximum_supplemental_count"),
                    "estimated_cost": float(action.get("budget", 0)),
                }
            ],
            "estimated_total_cost": float(action.get("budget", 0)),
            "cost_unit": "relative_cost",
            "calibration": preview.get("cost_calibration"),
        }
    if tool == "allocate_mc_bohb":
        preview = params.get("budget_preview") or {}
        stages.append(
            {
                "stage": "deep_search",
                "tasks": preview.get("selected_branch_count", "MC-selected"),
                "mc_steps": preview.get("requested_steps", int(params.get("mc_budget", 0))),
                "initial_cost": float(action.get("budget", 0) or params.get("mc_budget", 0)),
            }
        )
    elif tool == "prepare_local_batch_files" and params.get("mode") == "relax_inputs":
        stages.append(
            {
                "stage": "relax_and_feature",
                "tasks": "existing legal structures",
                "initial_cost": float(action.get("budget", 0) or 0),
            }
        )
    elif tool == "select_dft_candidates":
        decisions = params.get("decisions") or []
        if not isinstance(decisions, list) or any(not isinstance(row, dict) for row in decisions):
            return {
                "workload": [],
                "estimated_total_cost": None,
                "error": "invalid_dft_decisions_format",
                "cost_unit": "relative_cost",
            }
        for name, stage in (("DFT_SINGLE_POINT", "dft_single_point"), ("DFT_RELAX", "dft_relax")):
            count = sum(row.get("action") == name for row in decisions)
            if count:
                base = _stage_unit(state, stage)
                stages.append({"stage": stage, "tasks": count, "initial_cost": count * base})
        preview = params.get("dft_input_preview") or {}
        if preview.get("relative_cost") is not None:
            stages = preview.get("workload") or [
                {
                    "stage": "dft_relax",
                    "tasks": preview["task_count"],
                    "initial_cost": preview["relative_cost"],
                }
            ]
    elif tool == "run_calculation_stage":
        stages.append(
            {
                "stage": action.get("stage"),
                "tasks": max(1, len(action.get("target_ids") or [])),
                "initial_cost": float(action.get("budget", 0) or 0),
            }
        )
    total = 0.0
    for row in stages:
        factor, samples = calibrate_relative_cost(state, row["stage"])
        if tool == "select_dft_candidates" and params.get("dft_input_preview"):
            factor = 1.0  # Preview already uses the configured scientific cost model.
        row["calibration_factor"] = factor
        row["history_samples"] = samples
        row["estimated_cost"] = float(row["initial_cost"]) * factor
        total += row["estimated_cost"]
    limits = state.get("budget_limits") or {}
    limit = limits.get("total_relative_cost")
    used = float((state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
    reserved = float(state.get("reserved_relative_cost", 0) or 0)
    return {
        "workload": stages,
        "estimated_total_cost": total,
        "cost_unit": limits.get("cost_unit", "relative_cost"),
        "used_cost": used,
        "reserved_cost": reserved,
        "remaining_before": None if limit is None else max(0.0, float(limit) - used - reserved),
        "remaining_after": None
        if limit is None
        else max(0.0, float(limit) - used - reserved - total),
        "calibration": "30% initial estimate + 70% historical actual/planned ratio",
    }


def _stage_unit(state, stage):
    return float(
        (((state.get("budget_limits") or {}).get("stage_limits") or {}).get(stage) or {}).get(
            "task_cost", 0
        )
    )

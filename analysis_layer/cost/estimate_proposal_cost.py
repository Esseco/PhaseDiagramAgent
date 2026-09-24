"""Describe proposed workload using initial estimates calibrated by actual costs."""

from __future__ import annotations


def estimate_proposal_cost(action, state):
    tool = action.get("tool"); params = action.get("parameters") or {}
    stages = []
    if tool == "allocate_mc_bohb":
        stages.append({"stage": "deep_search", "tasks": "BOHB-selected", "mc_steps": int(params.get("mc_budget", 0)), "initial_cost": float(action.get("budget", 0) or params.get("mc_budget", 0))})
    elif tool == "select_dft_candidates":
        decisions = params.get("decisions") or []
        for name, stage in (("DFT_SINGLE_POINT", "dft_single_point"), ("DFT_RELAX", "dft_relax")):
            count = sum(row.get("action") == name for row in decisions)
            if count:
                base = _stage_unit(state, stage)
                stages.append({"stage": stage, "tasks": count, "initial_cost": count * base})
    elif tool == "run_calculation_stage":
        stages.append({"stage": action.get("stage"), "tasks": max(1, len(action.get("target_ids") or [])), "initial_cost": float(action.get("budget", 0) or 0)})
    total = 0.0
    for row in stages:
        factor, samples = _calibration(state, row["stage"])
        row["calibration_factor"] = factor; row["history_samples"] = samples
        row["estimated_cost"] = float(row["initial_cost"]) * factor
        total += row["estimated_cost"]
    limits = state.get("budget_limits") or {}
    limit = limits.get("total_relative_cost")
    used = float((state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
    reserved = float(state.get("reserved_relative_cost", 0) or 0)
    return {"workload": stages, "estimated_total_cost": total, "cost_unit": limits.get("cost_unit", "relative_cost"), "used_cost": used, "reserved_cost": reserved, "remaining_before": None if limit is None else max(0.0, float(limit) - used - reserved), "remaining_after": None if limit is None else max(0.0, float(limit) - used - reserved - total), "calibration": "30% initial estimate + 70% historical actual/planned ratio"}


def _stage_unit(state, stage):
    return float((((state.get("budget_limits") or {}).get("stage_limits") or {}).get(stage) or {}).get("task_cost", 0))


def _calibration(state, stage):
    ratios = [float(row["actual_cost"]) / float(row["planned_cost"]) for row in state.get("cost_history", []) if row.get("stage") == stage and float(row.get("planned_cost", 0) or 0) > 0 and row.get("status") == "completed"]
    if not ratios: return 1.0, 0
    observed = sum(ratios[-10:]) / len(ratios[-10:])
    return max(.25, min(4.0, .3 + .7 * observed)), len(ratios[-10:])

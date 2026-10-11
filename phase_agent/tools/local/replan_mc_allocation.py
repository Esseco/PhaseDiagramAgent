"""Recompute an unsubmitted MC wave using the confirmed Relax/hull policy."""

from copy import deepcopy

from phase_agent.decisions.strategy.estimate_branch_mc_budget import estimate_branch_mc_budget
from phase_agent.tools.local.regenerate_mc_inputs import _approved_mc_preview


def replan_mc_allocation(state, manager, config):
    previous, _ = _approved_mc_preview(state)
    keys = {row["task_key"] for row in previous["allocations"]}
    old = [row for row in state.get("tasks", []) if row.get("task_key") in keys]
    if not old or any(row.get("status") != "pending" or row.get("job_id") for row in old):
        raise ValueError("只能重新规划尚未提交的当前 MC；已完成结果须保留")
    version = state.get("current_branch_hull_version")
    pool = (state.get("branch_hull_batches") or {}).get(version)
    if not pool or any(row.get("model_version") != pool["model_version"] for row in old):
        raise ValueError("当前 MC 与 Relax 凸包模型不一致，不能替换")
    clean = deepcopy(state)
    clean["tasks"] = [row for row in clean.get("tasks", []) if row.get("task_key") not in keys]
    clean.setdefault("tiered_mc_state", {})["segments"] = [
        row
        for row in clean.get("tiered_mc_state", {}).get("segments", [])
        if row.get("task_key") not in keys
    ]
    ids = sorted(
        {
            row["branch_id"]
            for row in pool["records"]
            if row.get("stage") in (None, "relax_and_feature")
        }
    )
    branches = manager.data.get("branches") or {}
    if not ids or any(key not in branches for key in ids):
        raise ValueError("Relax 分支台账不完整，不能重新规划 MC")
    preview = estimate_branch_mc_budget(
        [{**branches[key], "branch_id": key} for key in ids],
        pool,
        clean,
        config,
        step_limit=float("inf"),
        seed=int((config.get("run") or {}).get("seed", 42)),
    )
    if preview.get("missing_branch_ids") or not preview.get("allocations"):
        raise ValueError("MC 候选缺少有效凸包依据，不能重新规划")
    steps = preview["requested_steps"]
    maximum = (config.get("round_strategy") or {}).get("maximum_mc_budget")
    limits = ((config.get("budgets") or {}).get("stage_limits") or {}).get("deep_search") or {}
    if maximum is not None and steps > maximum:
        raise ValueError(
            f"完整 MC 需要 {steps} 步，超过配置上限 {maximum}；请先修订预算，不自动缩成单任务"
        )
    if limits.get("max_tasks") is not None and len(preview["allocations"]) > limits["max_tasks"]:
        raise ValueError("完整 MC 超过任务数量上限；请先修订预算，不自动截断")
    if (
        limits.get("max_cost") is not None
        and preview["estimated_relative_cost"] > limits["max_cost"]
    ):
        raise ValueError("完整 MC 超过成本上限；请先修订预算，不自动截断")
    # Freeze a finite, serializable budget; the legacy cost value is never reused as steps.
    preview.update(step_limit=steps, target_steps=steps, candidate_branch_ids=ids)
    return preview

"""幂等记录一个预算评估结果和累计成本。"""

from copy import deepcopy

from .calculate_bohb_loss import calculate_bohb_loss


def record_bohb_result(state: dict, branch: dict, result: dict, *, scope_id: str, budget: int, seed: int, objective: dict) -> dict:
    updated = deepcopy(state)
    if updated.get("scope_id") not in {None, scope_id}:
        raise ValueError("不能混用不同 MLIP、凸包或候选范围的 BOHB 记录")
    updated["scope_id"] = scope_id
    scope = updated.get("scope", {})
    for field in ("model_version", "hull_reference_version"):
        if scope.get(field) is not None and result.get(field) != scope[field]:
            raise ValueError(f"结果 {field} 与固定 BOHB scope 不一致")
    task_key = result.get("task_key") or f"{scope_id}:{branch['branch_id']}:{budget}:{seed}"
    records = updated.setdefault("observations", [])
    existing = next((item for item in records if item["task_key"] == task_key), None)
    if existing and existing.get("status") == "completed":
        return updated
    loss = calculate_bohb_loss(branch, result, objective=objective)
    previous_budget = max([item.get("budget", 0) for item in records if item.get("branch_id") == branch["branch_id"] and item.get("status") == "completed"], default=0)
    incremental_budget = max(0, budget - previous_budget)
    status = result.get("status", "failed")
    actual_steps = result.get("actual_mc_steps")
    charged_budget = (min(incremental_budget, max(0, int(actual_steps))) if actual_steps is not None
                      else incremental_budget if status in {"completed", "failed"} else 0)
    actual_gpu = result.get("actual_gpu_core_hours")
    actual_cost = result.get("actual_cost")
    estimated_cost = result.get("estimated_cost")
    record = {"task_key": task_key, "branch_id": branch["branch_id"], "budget": budget, "requested_max_mc_steps": result.get("requested_max_mc_steps", incremental_budget), "actual_mc_steps": actual_steps, "patience_steps": result.get("patience_steps"), "min_improvement": result.get("min_improvement"), "stop_reason": result.get("stop_reason"), "restart_mode": result.get("restart_mode", "new_segment_from_structure"), "strict_chain_resume": False, "incremental_budget": incremental_budget, "charged_budget": charged_budget, "released_budget": max(0, incremental_budget-charged_budget), "seed": seed, "status": status, "loss": loss.get("loss"), "group": loss.get("group"), "loss_details": loss, "checkpoint": result.get("checkpoint"), "actual_cost": actual_cost, "estimated_cost": estimated_cost, "actual_gpu_core_hours": actual_gpu, "failure": result.get("error"), "bohb_features": branch.get("bohb_features"), "model_version": result.get("model_version"), "hull_reference_version": result.get("hull_reference_version")}
    if existing:
        records[records.index(existing)] = record
    else:
        records.append(record)
    if charged_budget:
        updated["consumed_mc_budget"] = float(updated.get("consumed_mc_budget", 0)) + charged_budget
    if isinstance(actual_gpu, (int, float)):
        updated["actual_gpu_core_hours"] = float(updated.get("actual_gpu_core_hours", 0)) + float(actual_gpu)
    return updated

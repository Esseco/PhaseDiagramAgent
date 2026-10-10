"""在一致能量基准下计算一次性搜索收益。"""

from __future__ import annotations

from typing import Any


def calculate_search_reward(
    previous_snapshot: dict[str, Any],
    current_snapshot: dict[str, Any],
    *,
    batch_id: str,
    task_ids: list[str],
    actual_cost: float,
    reward_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 Ehull 降低和新稳定结构计收益；批次及任务只计一次。"""
    state = reward_state or {"accounted_batches": [], "accounted_tasks": []}
    if previous_snapshot.get("energy_basis_id") != current_snapshot.get("energy_basis_id"):
        raise ValueError("不能比较不同能量基准或 MLIP/DFT 相图")
    if batch_id in state["accounted_batches"] or any(
        task in state["accounted_tasks"] for task in task_ids
    ):
        return {"status": "already_accounted", "reward": 0.0, "state": state}
    old = {item["record_id"]: item for item in previous_snapshot.get("entries", [])}
    improvement = 0.0
    new_stable = 0
    for item in current_snapshot.get("entries", []):
        before = old.get(item["record_id"])
        if before is not None:
            improvement += max(0.0, float(before["ehull"]) - float(item["ehull"]))
        elif item.get("is_stable"):
            new_stable += 1
    cost = float(actual_cost)
    if cost < 0:
        raise ValueError("actual_cost 不能为负")
    reward = improvement + new_stable
    state["accounted_batches"].append(batch_id)
    state["accounted_tasks"].extend(
        task for task in task_ids if task not in state["accounted_tasks"]
    )
    return {
        "status": "accounted",
        "batch_id": batch_id,
        "ehull_improvement": improvement,
        "new_stable_entries": new_stable,
        "actual_cost": cost,
        "reward": reward,
        "reward_per_cost": reward / cost if cost > 0 else None,
        "state": state,
    }

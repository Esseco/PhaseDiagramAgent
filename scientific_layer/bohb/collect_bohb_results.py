"""回收 pending BOHB 任务，移除活动项并幂等计费。"""

from .record_bohb_result import record_bohb_result


def collect_bohb_results(state: dict, candidates: list[dict], results: list[dict], *, config: dict) -> dict:
    current = dict(state)
    current["observations"] = list(state.get("observations", []))
    current["pending_tasks"] = list(state.get("pending_tasks", []))
    by_id = {item["branch_id"]: item for item in candidates}
    for result in results:
        task_key = result.get("task_key")
        pending = next((item for item in current["pending_tasks"] if item["task_key"] == task_key), None)
        if pending is None:
            if any(item["task_key"] == task_key for item in current["observations"]):
                continue
            raise KeyError(f"未知 BOHB 任务：{task_key}")
        merged = {**pending, **result}
        if merged.get("status") in {"pending", "running"}:
            current["pending_tasks"][current["pending_tasks"].index(pending)] = merged
            continue
        current["pending_tasks"].remove(pending)
        current = record_bohb_result(current, by_id[merged["branch_id"]], merged, scope_id=current["scope_id"], budget=int(merged["budget"]), seed=int(merged["seed"]), objective=config["objective"])
    return current

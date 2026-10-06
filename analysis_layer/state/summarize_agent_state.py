"""生成供规则或 LLM 阅读的最小状态摘要。"""
from execution_layer.state.task_waiting import active_pending_tasks


def summarize_agent_state(state: dict) -> dict:
    tasks = state.get("tasks", [])
    status_counts = {}
    phase_counts = {"identified": 0, "unknown": 0}
    for item in tasks:
        status_counts[item.get("status", "unknown")] = status_counts.get(item.get("status", "unknown"), 0) + 1
        if item.get("status") == "completed" and item.get("stage") in {
                "relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}:
            identified = (item.get("outputs") or {}).get("phase_identification") or {}
            key = "identified" if identified.get("status") == "identified" else "unknown"
            phase_counts[key] += 1
    stale = []
    for collection in ("energy_records", "feature_records", "surrogate_predictions", "phase_diagrams"):
        values = state.get(collection, [])
        if isinstance(values, dict):
            values = values.values()
        stale.extend({"collection": collection, "record_id": item.get("record_id") or item.get("id") or item.get("version"), "refresh_for_model_version": item.get("refresh_for_model_version")} for item in values if isinstance(item, dict) and item.get("validity") == "stale")
    return {
        "iteration": state.get("iteration", 0),
        "remaining_budget": state.get("remaining_budget"),
        "known_target_ids": sorted(set(state.get("known_target_ids", []))),
        "coverage_gaps": state.get("coverage_gaps", []),
        "candidate_ids": [item.get("candidate_id") for item in state.get("candidates", []) if item.get("candidate_id")],
        "active_task_keys": [item.get("task_key") for item in active_pending_tasks(tasks)],
        "waived_dft_wait_task_ids": [item.get("task_id") for item in tasks if item.get("recovery_wait_waived") is True],
        "completed_task_keys": [item.get("task_key") for item in tasks if item.get("status") == "completed"],
        "score_results": state.get("score_results", {}),
        "allowed_actions": state.get("allowed_actions", ["generate", "run_stage", "select_dft", "wait"]),
        "budget_usage": state.get("budget_usage", {}),
        "budget_reservations": state.get("budget_reservations", {}),
        "task_status_counts": status_counts,
        "active_model_version": state.get("active_model_version"),
        "phase_identification_counts": phase_counts,
        "stale_or_refresh_required": stale,
        "unknown_results": [item.get("task_key") for item in tasks if item.get("status") in {"unknown", "not_configured"}],
        "deferred_tasks": [item.get("task_key") for item in tasks if item.get("status") == "deferred"],
        "low_yield_tasks": [item.get("task_key") for item in tasks if item.get("status") == "low_yield"],
    }

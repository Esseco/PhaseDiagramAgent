"""确定性的基础调度规则。"""


def choose_rule_action(summary: dict, *, config=None) -> dict:
    settings = {"default_generation_strategy": "coverage", "generation_budget": 1.0, "calculation_budget": 1.0}
    settings.update(config or {})
    if summary.get("active_task_keys"):
        return {"action": "wait", "target_ids": [], "budget": 0.0, "reason": "tasks_in_progress", "source": "rule"}
    if summary.get("coverage_gaps"):
        target = summary["coverage_gaps"][0].get("region_id")
        return {"action": "generate", "target_ids": [target] if target else [], "strategy": settings["default_generation_strategy"], "budget": settings["generation_budget"], "reason": "largest_known_coverage_gap", "source": "rule"}
    if summary.get("candidate_ids"):
        return {"action": "run_stage", "target_ids": [summary["candidate_ids"][0]], "budget": settings["calculation_budget"], "reason": "next_available_candidate", "source": "rule"}
    return {"action": "wait", "target_ids": [], "budget": 0.0, "reason": "no_actionable_candidate", "source": "rule"}

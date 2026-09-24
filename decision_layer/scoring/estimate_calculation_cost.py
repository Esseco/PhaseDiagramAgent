"""按规模和预算估计相对计算成本。"""

from __future__ import annotations

from typing import Any
from config_layer.defaults.default_budget_rules import default_budget_rules
from execution_layer.budget.estimate_stage_cost import estimate_stage_cost


def estimate_calculation_cost(task: dict[str, Any], *, config=None) -> dict[str, Any]:
    defaults = default_budget_rules()
    settings = {**defaults["cost_model"], "stage_factors": {name: row["task_cost"] for name, row in defaults["stage_limits"].items()}}
    settings.update(config or {})
    missing = [key for key in ("stage", "atom_count") if task.get(key) is None]
    factor = settings["stage_factors"].get(task.get("stage"))
    if factor is None:
        missing.append("stage_factor")
    if missing:
        return _unknown(missing, task, settings)
    initial = task.get("initial_state_count", 1)
    budget = task.get("search_budget", 1.0)
    model = dict(settings)
    if config and "atom_exponent" in config and "stage_exponents" not in config:
        model["stage_exponents"] = {}
    estimate = estimate_stage_cost(task["stage"], atom_count=task["atom_count"], initial_state_count=initial, mc_steps=task.get("mc_steps"), budgets={"cost_model": model, "stage_limits": {task["stage"]: {"task_cost": factor}}})
    score = estimate["value"] * float(budget)
    return {"status": "completed", "score": score, "components": {"stage_factor": factor, "size_factor": estimate["size_factor"], "initial_state_count": initial, "search_budget": budget}, "basis": settings, "missing": [], "evidence": task, "unit": "relative_cost"}


def _unknown(missing, evidence, basis):
    return {"status": "unknown", "score": None, "components": {}, "basis": basis, "missing": sorted(set(missing)), "evidence": evidence, "unit": "relative_cost"}

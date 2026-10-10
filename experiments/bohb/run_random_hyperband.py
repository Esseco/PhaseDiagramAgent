"""用于基线比较的随机 Hyperband 多 bracket 实现。"""

import math
import random

from phase_agent.science.bohb.record_bohb_result import record_bohb_result
from phase_agent.science.bohb.validate_bohb_scope import validate_bohb_scope


def run_random_hyperband(candidates: list[dict], *, config: dict, total_mc_budget: int, seed: int, evaluator) -> dict:
    scope = validate_bohb_scope(config, candidates)
    state = {"scope_id": scope.get("scope_id"), "scope": dict(config.get("scope") or {}), "observations": [], "pending_tasks": [], "consumed_mc_budget": 0, "iteration": 0}
    if scope["status"] != "completed":
        return state
    levels, eta = list(config["budget_levels"]), int(config["eta"])
    rng = random.Random(seed)
    unused = list(candidates)
    for bracket in reversed(range(len(levels))):
        start_budget = levels[len(levels) - 1 - bracket]
        count = min(len(unused), max(1, math.ceil((len(levels) / (bracket + 1)) * eta**bracket)))
        if not count:
            break
        active = rng.sample(unused, count)
        used_ids = {item["branch_id"] for item in active}
        unused = [item for item in unused if item["branch_id"] not in used_ids]
        previous = 0
        for budget in [item for item in levels if item >= start_budget]:
            completed = []
            for offset, branch in enumerate(active):
                increment = budget - previous
                if state["consumed_mc_budget"] + increment > total_mc_budget:
                    return state
                action = {"budget": budget, "incremental_budget": increment, "seed": seed + bracket * 10000 + budget * 100 + offset, "task_key": f"hb:{bracket}:{branch['branch_id']}:{budget}", "model_version": config["scope"]["mlip_version"], "hull_reference_version": config["scope"]["hull_reference_version"]}
                result = {**action, **evaluator(branch=branch, action=action)}
                state = record_bohb_result(state, branch, result, scope_id=scope["scope_id"], budget=budget, seed=action["seed"], objective=config["objective"])
                record = state["observations"][-1]
                if record["status"] == "completed" and record["loss"] is not None:
                    completed.append((record["loss"], branch))
            completed.sort(key=lambda item: (item[0], item[1]["branch_id"]))
            active = [item[1] for item in completed[: max(1, len(completed) // eta)]]
            previous = budget
            if not active:
                break
    return state

"""用简单权重为各 branch 生成策略分配固定总配额。"""

from __future__ import annotations

from typing import Any


def compute_strategy_efficiency(
    state: dict[str, Any], existing_branches: list[dict[str, Any]]
) -> dict[str, float]:
    """Compute per-strategy near-hull yield (near_hull_branches / total_branches)."""
    by_bid = {row.get("branch_id"): row for row in existing_branches or [] if row.get("branch_id")}
    strategy_by_task: dict[str, str] = {}
    for rec in state.get("action_records") or []:
        raw = (rec.get("agent_proposal") or {}).get("raw_action") or rec.get("final_action") or {}
        if (raw.get("tool") or raw.get("action_type")) != "generate_branches":
            continue
        tk = raw.get("task_key")
        plan = (raw.get("parameters") or {}).get("generation_plan") or []
        if tk and plan and isinstance(plan[0], dict) and plan[0].get("strategy"):
            strategy_by_task[tk] = str(plan[0]["strategy"])

    counts: dict[str, list[int]] = {}
    for gen in state.get("generation_history") or []:
        tk = gen.get("task_key")
        strat = strategy_by_task.get(tk)
        if not strat:
            continue
        bucket = counts.setdefault(strat, [0, 0])
        for bid in gen.get("new_branch_ids") or []:
            bucket[0] += 1
            b = by_bid.get(bid) or {}
            ehull = b.get("current_model_ehull_ev_per_atom")
            if (
                b.get("is_hull_ground_state")
                or b.get("near_degenerate")
                or (isinstance(ehull, (int, float)) and ehull <= 0.020)
            ):
                bucket[1] += 1
    return {strat: round(hits / total, 4) for strat, (total, hits) in counts.items() if total > 0}


def choose_generation_strategy(
    metrics: dict[str, Any], *, total_quota: int, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """根据覆盖矩阵缺口、边际收益和历史效率动态分配配额。"""
    strategies = [
        "coverage",
        "composition",
        "competing_phase",
        "tm_ordering",
        "periodic_extension",
    ]
    if isinstance(total_quota, bool) or not isinstance(total_quota, int) or total_quota < 0:
        raise ValueError("total_quota 必须是非负整数")
    settings = {
        "minimum_per_strategy": 0,
        "base_weights": {
            "coverage": 1.0,
            "composition": 1.0,
            "competing_phase": 1.0,
            "tm_ordering": 1.0,
            "periodic_extension": 0.4,
        },
    }
    settings.update(config or {})
    weights = dict(settings["base_weights"])
    weights["coverage"] *= 1 + float(metrics.get("coverage_gap", 0.0))

    matrix = metrics.get("coverage_and_marginal_matrix") or []
    if matrix:
        unexplored = sum(1 for c in matrix if c.get("exploration_status") == "unexplored")
        high_gain = sum(1 for c in matrix if c.get("exploration_status") == "high_gain")
        if unexplored > 0:
            weights["coverage"] *= 1.0 + 0.5 * unexplored
            weights["competing_phase"] *= 1.0 + 0.3 * unexplored
        elif high_gain > 0:
            weights["composition"] *= 1.0 + 0.5 * high_gain
            weights["tm_ordering"] *= 1.0 + 0.4 * high_gain
        if int(metrics.get("near_degenerate_count") or 0) > 0:
            weights["periodic_extension"] *= 1.5

    historical_efficiency = dict(metrics.get("strategy_efficiency", {}))
    if "tm_mutation" in historical_efficiency and "tm_ordering" not in historical_efficiency:
        historical_efficiency["tm_ordering"] = historical_efficiency["tm_mutation"]
    for name, efficiency in historical_efficiency.items():
        if name in weights:
            weights[name] *= max(0.0, 1 + float(efficiency))
    minimum = (
        settings["minimum_per_strategy"]
        if total_quota >= len(strategies) * settings["minimum_per_strategy"]
        else 0
    )
    quotas = {name: minimum for name in strategies}
    remaining = total_quota - sum(quotas.values())
    total_weight = sum(weights.values()) or len(strategies)
    raw = {name: remaining * weights[name] / total_weight for name in strategies}
    for name in strategies:
        quotas[name] += int(raw[name])
    left = total_quota - sum(quotas.values())
    order = sorted(strategies, key=lambda name: (-(raw[name] - int(raw[name])), name))
    for name in order[:left]:
        quotas[name] += 1
    return {
        "quotas": quotas,
        "decision_basis": {
            "metrics": metrics,
            "weights": weights,
            "total_quota": total_quota,
            "minimum_per_strategy": minimum,
        },
    }

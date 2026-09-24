"""按策略收益更新下一轮配额和计算预算。"""

from __future__ import annotations

from typing import Any

from decision_layer.strategy.choose_generation_strategy import choose_generation_strategy


def update_search_policy(
    policy: dict[str, Any],
    rewards: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """采用平滑收益率，不使用强化学习；始终保留探索配额。"""
    settings = {
        "smoothing": 0.5,
        "minimum_exploration_fraction": 0.1,
        "budget_growth": 1.1,
        "budget_shrink": 0.9,
    }
    settings.update(config or {})
    efficiency = dict(policy.get("strategy_efficiency", {}))
    grouped = {}
    for item in rewards:
        if item.get("status") != "accounted":
            continue
        strategy = item.get("strategy", "coverage")
        grouped.setdefault(strategy, []).append(
            float(item.get("reward_per_cost") or 0.0)
        )
    for strategy, values in grouped.items():
        observed = sum(values) / len(values)
        old = efficiency.get(strategy, 0.0)
        efficiency[strategy] = (1 - settings["smoothing"]) * old + settings[
            "smoothing"
        ] * observed
    total_quota = int(policy.get("total_quota", 0))
    minimum = (
        max(1, int(total_quota * settings["minimum_exploration_fraction"] / 5))
        if total_quota >= 5
        else 0
    )
    allocation = choose_generation_strategy(
        {
            "coverage_gap": policy.get("coverage_gap", 0.0),
            "strategy_efficiency": efficiency,
        },
        total_quota=total_quota,
        config={"minimum_per_strategy": minimum},
    )
    positive = any(value > 0 for value in efficiency.values())
    old_budget = float(policy.get("calculation_budget", 0.0))
    budget = old_budget * (
        settings["budget_growth"] if positive else settings["budget_shrink"]
    )
    return {
        **policy,
        "quotas": allocation["quotas"],
        "strategy_efficiency": efficiency,
        "calculation_budget": budget,
        "decision_basis": {
            "rewards": rewards,
            "settings": settings,
            "allocation": allocation["decision_basis"],
        },
    }

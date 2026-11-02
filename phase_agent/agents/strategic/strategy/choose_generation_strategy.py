"""用简单权重为各 branch 生成策略分配固定总配额。"""

from __future__ import annotations

from typing import Any


def choose_generation_strategy(
    metrics: dict[str, Any], *, total_quota: int, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """根据覆盖缺口和历史效率分配配额，并保留每种策略的探索份额。"""
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
        "minimum_per_strategy": 1,
        "base_weights": {name: 1.0 for name in strategies},
    }
    settings.update(config or {})
    weights = dict(settings["base_weights"])
    weights["coverage"] *= 1 + float(metrics.get("coverage_gap", 0.0))
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

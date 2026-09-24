"""按策略分数分配整数配额，并保留随机探索。"""

from __future__ import annotations

from typing import Any


def allocate_generation_quotas(scores: dict[str, dict[str, Any]], *, total_quota: int, exploration_fraction: float = 0.2) -> dict[str, Any]:
    if total_quota < 0 or not 0 <= exploration_fraction <= 1:
        raise ValueError("total_quota 或 exploration_fraction 非法")
    strategies = sorted(scores)
    exploration = min(total_quota, round(total_quota * exploration_fraction))
    quotas = {name: 0 for name in strategies}
    for index in range(exploration):
        if strategies:
            quotas[strategies[index % len(strategies)]] += 1
    remaining = total_quota - exploration
    values = {name: max(0.0, float(item["score"])) if item.get("score") is not None else 0.0 for name, item in scores.items()}
    denominator = sum(values.values())
    if denominator == 0 and strategies:
        values = {name: 1.0 for name in strategies}
        denominator = len(strategies)
    raw = {name: remaining * values[name] / denominator for name in strategies}
    for name in strategies:
        quotas[name] += int(raw[name])
    left = total_quota - sum(quotas.values())
    order = sorted(strategies, key=lambda name: (-(raw[name] - int(raw[name])), name))
    for name in order[:left]:
        quotas[name] += 1
    return {"status": "completed", "quotas": quotas, "components": {"scores": scores, "exploration_quota": exploration}, "basis": {"total_quota": total_quota, "exploration_fraction": exploration_fraction}, "missing": [name for name, item in scores.items() if item.get("score") is None]}

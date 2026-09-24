"""评价一种 branch 生成策略的预期价值。"""

from __future__ import annotations

from typing import Any


def score_generation_strategy(strategy: str, metrics: dict[str, Any], *, weights: dict[str, float] | None = None) -> dict[str, Any]:
    config = {"coverage": 0.4, "historical_reward": 0.4, "novelty": 0.2}
    config.update(weights or {})
    values = {"coverage": metrics.get("coverage_gap"), "historical_reward": metrics.get("reward_per_cost"), "novelty": metrics.get("novelty")}
    available = {key: float(value) for key, value in values.items() if value is not None}
    missing = [key for key, value in values.items() if value is None]
    if not available:
        return {"status": "unknown", "score": None, "components": {}, "basis": config, "missing": missing, "evidence": {"strategy": strategy}}
    denominator = sum(config[key] for key in available)
    score = sum(config[key] * value for key, value in available.items()) / denominator
    return {"status": "completed", "score": score, "components": available, "basis": config, "missing": missing, "evidence": {"strategy": strategy}}

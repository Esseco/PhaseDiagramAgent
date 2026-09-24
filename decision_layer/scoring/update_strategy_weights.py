"""用已确认动作收益缓慢更新策略权重。"""


def update_strategy_weights(weights: dict[str, float], rewards: dict[str, float | None], *, config=None) -> dict:
    settings = {"learning_rate": 0.2, "minimum_weight": 0.05, "maximum_weight": 0.8, "exploration_strategy": "random_exploration"}
    settings.update(config or {})
    known = {key: float(value) for key, value in rewards.items() if value is not None}
    if not known:
        return {"status": "unknown", "score": None, "components": {}, "basis": settings, "missing": ["confirmed_strategy_rewards"], "evidence": rewards, "weights": dict(weights)}
    updated = dict(weights)
    low, high = min(known.values()), max(known.values())
    for key, reward in known.items():
        target = 0.5 if high == low else (reward - low) / (high - low)
        updated[key] = min(settings["maximum_weight"], max(settings["minimum_weight"], (1 - settings["learning_rate"]) * updated.get(key, 0.0) + settings["learning_rate"] * target))
    exploration = settings["exploration_strategy"]
    if exploration in updated:
        updated[exploration] = max(settings["minimum_weight"], updated[exploration])
    total = sum(updated.values())
    updated = {key: value / total for key, value in updated.items()} if total else updated
    return {"status": "completed", "score": None, "components": {"known_rewards": known}, "basis": settings, "missing": [key for key, value in rewards.items() if value is None], "evidence": rewards, "weights": updated}

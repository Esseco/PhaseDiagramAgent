"""组合已确认的反馈收益与实际成本。"""


def score_action_reward(data: dict, *, config=None) -> dict:
    settings = {"hull_weight": 1.0, "validation_weight": 1.0}
    settings.update(config or {})
    missing = [key for key in ("actual_cost",) if data.get(key) is None]
    rewards = {key: data.get(key) for key in ("hull_improvement", "validation_improvement") if data.get(key) is not None}
    if not rewards:
        missing.append("confirmed_reward")
    if missing:
        return {"status": "unknown", "score": None, "components": {}, "basis": settings, "missing": missing, "evidence": data}
    reward = settings["hull_weight"] * float(rewards.get("hull_improvement", 0.0)) + settings["validation_weight"] * float(rewards.get("validation_improvement", 0.0))
    cost = float(data["actual_cost"])
    return {"status": "completed", "score": reward / cost if cost > 0 else None, "components": {"reward": reward, "actual_cost": cost, **rewards}, "basis": settings, "missing": [] if cost > 0 else ["positive_actual_cost"], "evidence": data}

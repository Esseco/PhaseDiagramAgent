"""组合代理预测、凸包邻近度和不确定性。"""

from __future__ import annotations


def score_branch_potential(data: dict, *, config=None) -> dict:
    settings = {"weights": {"predicted_low_energy": 0.5, "near_hull": 0.3, "uncertainty": 0.2}, "energy_scale": 0.1}
    settings.update(config or {})
    values = {}
    if data.get("predicted_energy_above_hull") is not None:
        ehull = max(0.0, float(data["predicted_energy_above_hull"]))
        values["predicted_low_energy"] = 1.0 / (1.0 + ehull / settings["energy_scale"])
        values["near_hull"] = values["predicted_low_energy"]
    if data.get("uncertainty") is not None:
        values["uncertainty"] = min(1.0, max(0.0, float(data["uncertainty"])))
    if not values:
        return {"status": "unknown", "score": None, "components": {}, "basis": settings, "missing": ["predicted_energy_above_hull", "uncertainty"], "evidence": data}
    weights = settings["weights"]
    total = sum(weights[key] for key in values)
    score = sum(weights[key] * value for key, value in values.items()) / total
    missing = [key for key in weights if key not in values]
    return {"status": "partial" if missing else "completed", "score": score, "components": values, "basis": settings, "missing": missing, "evidence": data}

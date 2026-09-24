"""评估继续 MC 搜索的边际价值。"""

from __future__ import annotations


def score_additional_mc_value(data: dict, *, config=None) -> dict:
    settings = {"weights": {"recent_improvement": 0.5, "unsearched_fraction": 0.3, "uncertainty": 0.2}, "improvement_scale": 0.02}
    settings.update(config or {})
    values = {}
    if data.get("recent_energy_improvement") is not None:
        values["recent_improvement"] = min(1.0, max(0.0, float(data["recent_energy_improvement"]) / settings["improvement_scale"]))
    for key in ("unsearched_fraction", "uncertainty"):
        if data.get(key) is not None:
            values[key] = min(1.0, max(0.0, float(data[key])))
    if not values:
        return {"status": "unknown", "score": None, "components": {}, "basis": settings, "missing": list(settings["weights"]), "evidence": data}
    weights = settings["weights"]
    total = sum(weights[key] for key in values)
    missing = [key for key in weights if key not in values]
    return {"status": "partial" if missing else "completed", "score": sum(weights[key] * value for key, value in values.items()) / total, "components": values, "basis": settings, "missing": missing, "evidence": data}

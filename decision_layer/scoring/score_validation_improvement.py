"""在同一独立验证集和指标下比较新旧模型。"""


def score_validation_improvement(before: dict, after: dict) -> dict:
    keys = ("validation_data_version", "metric", "unit")
    mismatched = [key for key in keys if not before.get(key) or before.get(key) != after.get(key)]
    if mismatched or before.get("value") is None or after.get("value") is None:
        missing = mismatched + [key for key in ("before.value", "after.value") if (before if key.startswith("before") else after).get("value") is None]
        return {"status": "unknown", "score": None, "components": {}, "basis": {key: before.get(key) for key in keys}, "missing": missing, "evidence": {"before": before, "after": after}}
    improvement = float(before["value"]) - float(after["value"])
    return {"status": "completed", "score": improvement, "components": {"absolute_error_reduction": improvement, "relative_error_reduction": improvement / float(before["value"]) if before["value"] else None}, "basis": {key: before[key] for key in keys}, "missing": [], "evidence": {"old_model": before.get("model_version"), "new_model": after.get("model_version")}}

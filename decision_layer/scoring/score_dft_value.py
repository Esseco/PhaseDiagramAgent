"""评估 DFT 标注价值；QBC 仅作一个组成项。"""

from __future__ import annotations


def score_dft_value(data: dict, *, config=None) -> dict:
    settings = {"weights": {"hull_impact": 0.4, "qbc_disagreement": 0.25, "diversity": 0.2, "independent_audit": 0.15}}
    settings.update(config or {})
    values = {key: min(1.0, max(0.0, float(data[key]))) for key in settings["weights"] if data.get(key) is not None}
    if not values:
        return {"status": "unknown", "score": None, "components": {}, "basis": settings, "missing": list(settings["weights"]), "evidence": data, "warning": "QBC disagreement is not true error"}
    weights = settings["weights"]
    total = sum(weights[key] for key in values)
    missing = [key for key in weights if key not in values]
    return {"status": "partial" if missing else "completed", "score": sum(weights[key] * value for key, value in values.items()) / total, "components": values, "basis": settings, "missing": missing, "evidence": data, "warning": "QBC disagreement is not true error"}

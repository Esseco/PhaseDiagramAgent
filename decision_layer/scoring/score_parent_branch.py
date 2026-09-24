"""根据低能潜力、覆盖和不确定性评价父 branch。"""

from __future__ import annotations

from typing import Any


def score_parent_branch(branch: dict[str, Any], *, weights: dict[str, float] | None = None) -> dict[str, Any]:
    config = {"low_energy": 0.5, "coverage_gap": 0.25, "uncertainty": 0.25}
    config.update(weights or {})
    fields = {"low_energy": branch.get("low_energy_score"), "coverage_gap": branch.get("coverage_gap_score"), "uncertainty": branch.get("uncertainty_score")}
    available = {key: float(value) for key, value in fields.items() if value is not None}
    missing = [key for key, value in fields.items() if value is None]
    if not available:
        return {"status": "unknown", "score": None, "components": {}, "basis": config, "missing": missing, "evidence": {"branch_id": branch.get("branch_id")}}
    used_weight = sum(config[key] for key in available)
    score = sum(config[key] * value for key, value in available.items()) / used_weight
    return {"status": "completed", "score": score, "components": available, "basis": config, "missing": missing, "evidence": {"branch_id": branch.get("branch_id")}}

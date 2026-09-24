"""评估相、组分和超胞区域的覆盖缺口。"""

from __future__ import annotations

from typing import Any


def score_coverage_gap(region: dict[str, Any], *, config: dict[str, float] | None = None) -> dict[str, Any]:
    settings = {"target_branches": 3.0, "target_structures": 6.0, "branch_weight": 0.5, "structure_weight": 0.5}
    settings.update(config or {})
    missing = [key for key in ("branch_count", "structure_count") if region.get(key) is None]
    if missing:
        return _unknown(missing, region)
    branch_gap = max(0.0, 1 - float(region["branch_count"]) / settings["target_branches"])
    structure_gap = max(0.0, 1 - float(region["structure_count"]) / settings["target_structures"])
    components = {"branch_gap": branch_gap, "structure_gap": structure_gap}
    score = settings["branch_weight"] * branch_gap + settings["structure_weight"] * structure_gap
    return {"status": "completed", "score": score, "components": components, "basis": settings, "missing": [], "evidence": {"region": region}}


def _unknown(missing, evidence):
    return {"status": "unknown", "score": None, "components": {}, "basis": {}, "missing": missing, "evidence": evidence}

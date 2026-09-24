"""在相同能量基准下计算凸包改善。"""


def score_hull_improvement(before: dict, after: dict) -> dict:
    basis = before.get("energy_basis_id")
    if not basis or basis != after.get("energy_basis_id"):
        return {"status": "unknown", "score": None, "components": {}, "basis": {"energy_basis_id": basis}, "missing": ["matching_energy_basis_id"], "evidence": {"before": before.get("energy_basis_id"), "after": after.get("energy_basis_id")}}
    old = {item["record_id"]: item for item in before.get("entries", []) if item.get("ehull") is not None}
    improvements = [max(0.0, float(old[item["record_id"]]["ehull"]) - float(item["ehull"])) for item in after.get("entries", []) if item.get("record_id") in old and item.get("ehull") is not None]
    new_stable = sum(bool(item.get("is_stable")) and item.get("record_id") not in old for item in after.get("entries", []))
    score = sum(improvements)
    return {"status": "completed", "score": score, "components": {"ehull_improvement": score, "new_stable_entries": new_stable}, "basis": {"energy_basis_id": basis}, "missing": [], "evidence": {"matched_entries": len(improvements)}}

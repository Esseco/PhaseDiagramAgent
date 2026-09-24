"""用现有五阶段规则并附加可解释评分。"""

from decision_layer.calculation.decide_next_calculation import decide_next_calculation


def decide_next_stage(record: dict, *, scores=None, rules=None) -> dict:
    decision = decide_next_calculation(record, rules=rules)
    return {"status": "completed", "score": None, "components": dict(scores or {}), "basis": dict(rules or {}), "missing": [], "evidence": {"record_id": record.get("structure_id") or record.get("id")}, **decision}

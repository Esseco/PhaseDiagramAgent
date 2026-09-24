"""Deterministic round-level fallback; never chooses individual tasks."""

from copy import deepcopy


def choose_rule_round_strategy(summary: dict, *, config: dict) -> dict:
    strategy = deepcopy(config.get("rule_default") or {})
    gaps = sorted(summary.get("coverage_gaps") or [], key=lambda item: (-float(item.get("score", 0)), str(item.get("region_id"))))
    if gaps:
        strategy["focus_regions"] = [item["region_id"] for item in gaps[:3]]
    return {**strategy, "source": "rule", "decision_level": "round_strategy"}

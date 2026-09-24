"""检查低预算是否能预测高预算排序。"""

import numpy as np


def evaluate_budget_correlation(observations: list[dict], *, low_budget: int, high_budget: int, minimum_pairs: int = 5) -> dict:
    low = {item["branch_id"]: item["loss"] for item in observations if item.get("status") == "completed" and item.get("budget") == low_budget and item.get("loss") is not None}
    high = {item["branch_id"]: item["loss"] for item in observations if item.get("status") == "completed" and item.get("budget") == high_budget and item.get("loss") is not None}
    ids = sorted(low.keys() & high.keys())
    if len(ids) < minimum_pairs:
        return {"status": "insufficient_data", "pair_count": len(ids), "spearman": None, "minimum_pairs": minimum_pairs}
    low_rank = _ranks([low[key] for key in ids])
    high_rank = _ranks([high[key] for key in ids])
    value = float(np.corrcoef(low_rank, high_rank)[0, 1])
    return {"status": "completed", "pair_count": len(ids), "spearman": value, "low_budget": low_budget, "high_budget": high_budget, "interpretation": "validate_fidelity_before_claiming_bohb_benefit"}


def _ranks(values):
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks

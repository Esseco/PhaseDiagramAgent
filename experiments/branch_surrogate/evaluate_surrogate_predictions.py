"""Evaluate error, ranking, important misses and uncertainty behavior."""


def evaluate_surrogate_predictions(prediction_record: dict, rows: list[dict], *, low_energy_fraction=.1) -> dict:
    if prediction_record.get("status") != "completed":
        return {"status": "skipped", "reason": prediction_record.get("reason")}
    truth = {row["branch_id"]: row for row in rows if row.get("target") is not None}
    pairs = [(item, truth[item["branch_id"]]) for item in prediction_record["predictions"] if item["branch_id"] in truth]
    if not pairs:
        return {"status": "insufficient_data", "row_count": 0}
    import numpy as np
    from scipy.stats import spearmanr
    y = np.array([row["target"] for _, row in pairs]); p = np.array([item["prediction"] for item, _ in pairs])
    count = max(1, round(len(pairs) * float(low_energy_fraction)))
    selected = {item["branch_id"] for item, _ in sorted(pairs, key=lambda pair: pair[0]["prediction"])[:count]}
    important = {row["branch_id"] for _, row in pairs if row.get("important")}
    true_low = {row["branch_id"] for _, row in sorted(pairs, key=lambda pair: pair[1]["target"])[:count]}
    uncertainties = [(item["uncertainty"], abs(item["prediction"] - row["target"])) for item, row in pairs if item.get("uncertainty") is not None]
    uncertainty_corr = _spearman([a for a, _ in uncertainties], [b for _, b in uncertainties]) if len(uncertainties) >= 3 else None
    return {"status": "completed", "row_count": len(pairs), "mae": float(np.mean(abs(p-y))), "rmse": float(np.sqrt(np.mean((p-y)**2))), "spearman": _spearman(p, y) if len(pairs) >= 2 else None, "low_energy_recall": len(selected & true_low) / len(true_low), "important_missed": sorted(important - selected), "important_miss_rate": (len(important - selected) / len(important)) if important else None, "uncertainty_error_spearman": uncertainty_corr, "uncertainty_available_count": len(uncertainties)}


def _spearman(left, right):
    import numpy as np
    from scipy.stats import spearmanr
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None

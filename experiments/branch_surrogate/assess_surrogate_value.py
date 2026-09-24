"""Keep surrogate usefulness separate from BOHB effectiveness."""


def assess_surrogate_value(methods: dict, *, primary_metric="important_miss_rate") -> dict:
    baseline_names = {"random", "trial_energy"}
    baselines, surrogates = [], []
    for name, record in methods.items():
        test = record.get("test") or {}
        value = test.get(primary_metric)
        if record.get("status") == "completed" and value is not None:
            (baselines if name in baseline_names else surrogates).append((float(value), name))
    if not baselines or not surrogates:
        return {"status": "insufficient_evidence", "surrogate_has_value": None, "reason": "validated baseline or surrogate metric missing"}
    best_baseline, best_surrogate = min(baselines), min(surrogates)
    return {"status": "completed", "surrogate_has_value": best_surrogate[0] < best_baseline[0], "primary_metric": primary_metric, "best_baseline": {"method": best_baseline[1], "value": best_baseline[0]}, "best_surrogate": {"method": best_surrogate[1], "value": best_surrogate[0]}, "note": "offline evidence only"}

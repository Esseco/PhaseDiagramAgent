"""Predict scores and RF ensemble uncertainty."""

import time

from experiments.branch_surrogate.fit_surrogate_method import _flat


def predict_surrogate_method(fitted: dict, rows: list[dict], *, seed=0) -> dict:
    method = fitted["method"]
    if fitted.get("status") != "completed":
        return {"status": "skipped", "method": method, "reason": fitted.get("reason"), "predictions": []}
    started = time.perf_counter()
    if method == "random":
        import random
        rng = random.Random(seed)
        predictions = [{"branch_id": row["branch_id"], "prediction": rng.random(), "uncertainty": None} for row in rows]
    elif method == "trial_energy":
        usable = [row for row in rows if row.get("trial_energy") is not None]
        if len(usable) != len(rows):
            return {"status": "skipped", "method": method, "reason": "trial energy missing", "predictions": []}
        predictions = [{"branch_id": row["branch_id"], "prediction": row["trial_energy"], "uncertainty": None} for row in rows]
    else:
        names, model = fitted["feature_names"], fitted["model"]
        usable = [row for row in rows if all(_flat(row, method.replace("_rf", "")).get(name) is not None for name in names)]
        X = [[_flat(row, method.replace("_rf", ""))[name] for name in names] for row in usable]
        means = model.predict(X) if X else []
        tree_values = [[tree.predict([line])[0] for tree in model.estimators_] for line in X]
        import statistics
        predictions = [{"branch_id": row["branch_id"], "prediction": float(mean), "uncertainty": float(statistics.pstdev(values))} for row, mean, values in zip(usable, means, tree_values)]
    return {"status": "completed", "method": method, "predictions": predictions, "prediction_cost": time.perf_counter() - started}

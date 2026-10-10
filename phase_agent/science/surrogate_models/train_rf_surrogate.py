"""Train RF using a selector already fitted on the training split."""


def train_rf_surrogate(
    rows: list[dict], *, selected_features: list[str], preprocessor: dict, rf_config: dict
) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    X, y = [], []
    medians = preprocessor.get("medians") or {}
    for row in rows:
        X.append([row["features"].get(name, medians.get(name)) for name in selected_features])
        y.append(row["target"])
    if not X or any(value is None for line in X for value in line):
        return {"status": "incomplete", "model": None, "error": "missing selected features"}
    model = RandomForestRegressor(**rf_config).fit(X, y)
    return {
        "status": "completed",
        "model": model,
        "model_config": dict(rf_config),
        "selected_features": list(selected_features),
    }

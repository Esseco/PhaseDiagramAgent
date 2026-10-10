"""Fit cleaning, redundancy, relevance and model selection on training only."""


def select_features(X_train, y_train, *, feature_names: list[str], config: dict) -> dict:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("X_train 与 feature_names 不一致")
    missing_fraction = np.mean(~np.isfinite(X), axis=0)
    keep = [
        i
        for i in range(X.shape[1])
        if missing_fraction[i] <= float(config.get("missing_fraction_max", 0.0))
    ]
    if not keep:
        return {
            "status": "insufficient_features",
            "selected_features": [],
            "steps": {"invalid_removed": feature_names},
        }
    medians = np.nanmedian(X[:, keep], axis=0)
    clean = np.where(np.isfinite(X[:, keep]), X[:, keep], medians)
    variable = [
        j
        for j in range(clean.shape[1])
        if np.var(clean[:, j]) > float(config.get("variance_min", 1e-12))
    ]
    names = [feature_names[keep[j]] for j in variable]
    clean = clean[:, variable]
    selected = []
    threshold = float(config.get("redundancy_correlation_max", 0.95))
    for j in range(clean.shape[1]):
        if all(abs(np.corrcoef(clean[:, j], clean[:, k])[0, 1]) < threshold for k in selected):
            selected.append(j)
    clean, names = clean[:, selected], [names[j] for j in selected]
    relevance = {
        name: abs(float(np.corrcoef(clean[:, j], y)[0, 1])) if len(y) > 1 else 0.0
        for j, name in enumerate(names)
    }
    top = config.get("relevance_top_k")
    if top:
        names = sorted(names, key=lambda name: (-relevance[name], name))[: int(top)]
    positions = [feature_names.index(name) for name in names]
    train = X[:, positions]
    final_medians = np.nanmedian(train, axis=0)
    train = np.where(np.isfinite(train), train, final_medians)
    importance = {}
    model_top = config.get("model_top_k")
    if names and model_top and len(y) >= 2:
        model = RandomForestRegressor(n_estimators=100, random_state=0).fit(train, y)
        importance = dict(zip(names, map(float, model.feature_importances_)))
        names = sorted(names, key=lambda name: (-importance[name], name))[: int(model_top)]
    return {
        "status": "completed" if names else "insufficient_features",
        "selected_features": names,
        "preprocessor": {
            "input_features": feature_names,
            "medians": dict(zip([feature_names[i] for i in positions], map(float, final_medians))),
        },
        "steps": {
            "missing_fraction": dict(zip(feature_names, map(float, missing_fraction))),
            "relevance": relevance,
            "rf_importance": importance,
        },
    }

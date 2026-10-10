"""Fit one configured method using training rows only."""

import time


def fit_surrogate_method(method: str, train_rows: list[dict], *, surrogate_config: dict) -> dict:
    if method in {"random", "trial_energy"}:
        return {"status": "completed", "method": method, "model": None, "training_cost": 0.0}
    if method == "mattertune":
        adapter = (surrogate_config.get("mattertune") or {}).get("adapter")
        if not (surrogate_config.get("mattertune") or {}).get("enabled") or not callable(adapter):
            return _skip(method, "MatterTune adapter not configured")
        return adapter(train_rows=train_rows, config=surrogate_config.get("mattertune"))
    kind = "manual" if method == "manual_rf" else "embedding" if method == "embedding_rf" else "hybrid" if method == "hybrid_rf" else None
    if kind is None:
        return _skip(method, "unknown method")
    available = [row for row in train_rows if row.get("target") is not None and _vector(row, kind) is not None]
    if len(available) < 2:
        return _skip(method, "insufficient configured training rows")
    names = _names(available, kind)
    if not names:
        return _skip(method, "inconsistent or missing features")
    from sklearn.ensemble import RandomForestRegressor
    from phase_agent.science.surrogate_models.select_features import select_features
    started = time.perf_counter()
    X = [[_flat(row, kind).get(name) for name in names] for row in available]
    if any(value is None for line in X for value in line):
        return _skip(method, "missing feature values")
    selection = select_features(X, [row["target"] for row in available], feature_names=names, config=surrogate_config.get("selection") or {})
    if selection.get("status") != "completed":
        return _skip(method, "no feature remains after training-only selection")
    names = selection["selected_features"]
    X = [[_flat(row, kind).get(name) for name in names] for row in available]
    model = RandomForestRegressor(**(surrogate_config.get("rf") or {})).fit(X, [row["target"] for row in available])
    return {"status": "completed", "method": method, "model": model, "feature_names": names, "feature_selection": selection, "training_rows": len(available), "training_cost": time.perf_counter() - started}


def _flat(row, kind):
    values = {}
    if kind in {"manual", "hybrid"}:
        values.update({f"manual:{key}": value for key, value in (row.get("manual") or {}).items()})
    if kind in {"embedding", "hybrid"}:
        values.update({f"embedding:{i}": value for i, value in enumerate(row.get("embedding") or [])})
    return values


def _vector(row, kind):
    values = _flat(row, kind)
    return values if values else None


def _names(rows, kind):
    sets = [set(_flat(row, kind)) for row in rows]
    return sorted(set.intersection(*sets)) if sets else []


def _skip(method, reason):
    return {"status": "skipped", "method": method, "reason": reason, "model": None, "training_cost": None}

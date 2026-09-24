"""Replay sequential selection without exposing future labels."""


def replay_sequential_selection(rows: list[dict], *, method: str, surrogate_config: dict, comparison_config: dict, hull_metric=None) -> dict:
    complete = [row for row in rows if row.get("target") is not None and (row.get("cost_limit_check") or {}).get("allowed", True) and _row_cost(row, method) is not None]
    initial_n = int(comparison_config.get("initial_revealed", 10))
    if len(complete) <= initial_n:
        return {"status": "insufficient_history", "reason": f"need more than {initial_n} labeled rows with cost", "rounds": []}
    import random
    rng = random.Random(int(comparison_config.get("random_seed", 0)))
    order = sorted(complete, key=lambda row: row["branch_id"]); rng.shuffle(order)
    revealed, pool = order[:initial_n], order[initial_n:]
    cumulative = sum(_row_cost(row, method) for row in revealed)
    rounds = []
    while pool:
        from experiments.branch_surrogate.fit_surrogate_method import fit_surrogate_method
        from experiments.branch_surrogate.predict_surrogate_method import predict_surrogate_method
        fitted = fit_surrogate_method(method, revealed, surrogate_config=surrogate_config)
        predicted = predict_surrogate_method(fitted, pool, seed=len(rounds) + int(comparison_config.get("random_seed", 0)))
        if predicted.get("status") != "completed":
            return {"status": "skipped", "reason": predicted.get("reason"), "rounds": rounds}
        scores = {item["branch_id"]: item["prediction"] for item in predicted["predictions"]}
        selectable = [row for row in pool if row["branch_id"] in scores]
        if not selectable:
            break
        chosen = sorted(selectable, key=lambda row: (scores[row["branch_id"]], row["branch_id"]))[:int(comparison_config.get("selection_batch_size", 5))]
        added = sum(_row_cost(row, method) for row in chosen)
        limit = comparison_config.get("total_cost_budget")
        if limit is not None and cumulative + added > float(limit):
            break
        revealed.extend(chosen); chosen_ids = {row["branch_id"] for row in chosen}; pool = [row for row in pool if row["branch_id"] not in chosen_ids]; cumulative += added
        hull = hull_metric(revealed) if callable(hull_metric) else None
        rounds.append({"round": len(rounds), "selected": sorted(chosen_ids), "revealed_count": len(revealed), "cumulative_cost": cumulative, "hull_quality": hull, "hull_metric_status": "completed" if callable(hull_metric) else "not_configured"})
    return {"status": "completed" if rounds else "insufficient_history", "method": method, "cost_basis": comparison_config.get("cost_basis", "proxy_relative"), "initial_revealed": [row["branch_id"] for row in order[:initial_n]], "rounds": rounds, "label_visibility_rule": "round n uses labels revealed before round n only"}


def _row_cost(row, method):
    label = row.get("label_cost")
    if not isinstance(label, (int, float)):
        return None
    if method in {"manual_rf", "embedding_rf", "hybrid_rf", "mattertune"}:
        feature = row.get("feature_cost")
        return label + feature if isinstance(feature, (int, float)) else None
    return label

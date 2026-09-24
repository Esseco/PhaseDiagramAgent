"""按 Successive Halving 和组分覆盖决定预算晋升。"""


def promote_bohb_candidates(observations: list[dict], *, budget_levels: list[int], eta: int) -> list[dict]:
    promotions = []
    for index, budget in enumerate(budget_levels[:-1]):
        higher = budget_levels[index + 1]
        rows = [item for item in observations if item.get("status") == "completed" and item.get("budget") == budget and item.get("loss") is not None and item.get("promoted_to", 0) < higher]
        already = {item["branch_id"] for item in observations if item.get("budget") == higher}
        groups = {}
        for row in rows:
            groups.setdefault(row.get("group", "unknown"), []).append(row)
        chosen = []
        for group_rows in groups.values():
            group_rows.sort(key=lambda item: (item["loss"], item["branch_id"]))
            chosen.extend(group_rows[: max(1, len(group_rows) // eta)])
        limit = max(1, len(rows) // eta) if rows else 0
        chosen = sorted(chosen, key=lambda item: (item["loss"], item["branch_id"]))[: max(limit, len(groups))]
        for row in chosen:
            if row["branch_id"] not in already:
                promotions.append({"branch_id": row["branch_id"], "from_budget": budget, "to_budget": higher, "checkpoint": row.get("checkpoint"), "group": row.get("group")})
    return promotions

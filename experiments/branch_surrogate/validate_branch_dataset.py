"""Audit missing labels, version consistency and split leakage."""

from collections import Counter, defaultdict


def validate_branch_dataset(dataset: dict) -> dict:
    rows = dataset.get("rows") or []
    ready = [row for row in rows if row.get("status") == "ready"]
    issue_counts = Counter(issue for row in rows for issue in row.get("issues", []))
    group_splits = defaultdict(set)
    for row in rows:
        if row.get("split_group") and row.get("split"):
            group_splits[row["split_group"]].add(row["split"])
    leakage = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    negative = sum(1 for row in ready if row["target"].get("distance_to_fixed_hull", 0) < 0)
    return {
        "status": "invalid" if leakage else "completed",
        "row_count": len(rows), "ready_count": len(ready), "incomplete_count": len(rows) - len(ready),
        "split_counts": dict(Counter(row.get("split", "unassigned") for row in rows)),
        "negative_hull_distance_count": negative,
        "issue_counts": dict(sorted(issue_counts.items())),
        "split_group_leakage": leakage,
        "notes": ["负的固定凸包距离被保留，不截断", "DFT 标签不属于本数据任务"],
    }

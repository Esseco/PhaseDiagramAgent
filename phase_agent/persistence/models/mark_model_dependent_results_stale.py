"""Mark derived products stale after a validated model switch, preserving history."""

from copy import deepcopy


def mark_model_dependent_results_stale(
    state: dict, *, old_model_version: str, new_model_version: str
) -> dict:
    updated = deepcopy(state)
    marked = []
    for collection, kind in (
        ("energy_records", "energy"),
        ("feature_records", "features"),
        ("surrogate_predictions", "surrogate_prediction"),
        ("phase_diagrams", "convex_hull"),
    ):
        values = updated.get(collection, [])
        records = values.values() if isinstance(values, dict) else values
        for item in records:
            if not isinstance(item, dict):
                continue
            if (
                item.get("model_version") == old_model_version
                and item.get("validity", "valid") == "valid"
            ):
                item["validity"] = "stale"
                item["stale_reason"] = "model_version_changed"
                item["refresh_for_model_version"] = new_model_version
                marked.append(
                    {
                        "collection": collection,
                        "record_id": item.get("record_id") or item.get("id"),
                        "kind": kind,
                    }
                )
    updated["active_model_version"] = new_model_version
    updated.setdefault("model_transitions", []).append(
        {
            "old_model_version": old_model_version,
            "new_model_version": new_model_version,
            "refresh_required": marked,
        }
    )
    return {"state": updated, "marked": marked}

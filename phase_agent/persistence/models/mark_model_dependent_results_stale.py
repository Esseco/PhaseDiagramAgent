"""Mark derived products stale after a validated model switch, preserving history."""

from copy import deepcopy


def mark_model_dependent_results_stale(
    state: dict, *, old_model_version: str, new_model_version: str, copy_state: bool = True
) -> dict:
    if copy_state:
        updated = deepcopy(state)
    else:
        # Activation owns the loaded state.  Copy only collections whose
        # records are changed; leave large task/audit histories shared.
        updated = dict(state)
        updated["model_transitions"] = list(state.get("model_transitions") or [])
    marked = []
    for collection, kind in (
        ("energy_records", "energy"),
        ("feature_records", "features"),
        ("surrogate_predictions", "surrogate_prediction"),
        ("phase_diagrams", "convex_hull"),
    ):
        values = updated.get(collection, [])
        if not copy_state and values:
            values = dict(values) if isinstance(values, dict) else list(values)
            updated[collection] = values
        if copy_state:
            records = values.values() if isinstance(values, dict) else values
            items = ((None, item) for item in records)
        elif isinstance(values, dict):
            items = list(values.items())
        else:
            items = list(enumerate(values))
        for key, item in items:
            if not isinstance(item, dict):
                continue
            if (
                item.get("model_version") == old_model_version
                and item.get("validity", "valid") == "valid"
            ):
                if not copy_state:
                    item = deepcopy(item)
                    if isinstance(values, dict):
                        values[key] = item
                    else:
                        values[key] = item
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
    if updated.get("analysis_output_directory"):
        from phase_agent.analysis.feedback.export_dft_products import publish_output_catalog

        publish_output_catalog(updated, updated["analysis_output_directory"])
    return {"state": updated, "marked": marked}

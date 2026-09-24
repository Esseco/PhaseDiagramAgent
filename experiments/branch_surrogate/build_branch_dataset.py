"""Join branch, initial structures, fixed-budget MC labels, cost and provenance."""

from copy import deepcopy

from experiments.branch_surrogate.build_mc_target import build_mc_target
from experiments.branch_surrogate.extract_selection_features import extract_selection_features
from experiments.branch_surrogate.load_branch_history import load_branch_history


def build_branch_dataset(source, *, config: dict, hull_reference: dict) -> dict:
    manager = load_branch_history(source)
    rows = []
    for branch_id in sorted(manager.data["branches"]):
        branch = manager.data["branches"][branch_id]
        structures = [manager.data["structures"][item] for item in branch.get("structure_ids", []) if item in manager.data["structures"]]
        feature_record = extract_selection_features(branch, structures, config)
        target = build_mc_target(branch, structures, config=config, hull_reference=hull_reference)
        rows.append({
            "branch_id": branch_id,
            "branch": deepcopy(branch),
            "initial_structures": [{"structure_id": item.get("structure_id"), "source_path": item.get("source_path")} for item in structures],
            "features": feature_record["features"],
            "feature_provenance": feature_record["provenance"],
            "target": target,
            "status": "ready" if feature_record["status"] == "completed" and target["status"] == "completed" else "incomplete",
            "issues": feature_record["missing"] + feature_record["leakage_fields"] + target["missing"],
        })
    return {"schema_version": 1, "task": config.get("task"), "config": deepcopy(config), "hull_reference": deepcopy(hull_reference), "rows": rows}

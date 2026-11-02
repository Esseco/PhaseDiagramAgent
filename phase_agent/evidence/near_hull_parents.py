"""Current-model structure evidence for parent-driven branch strategies."""

from phase_agent.evidence.branch_search_evidence import existing_branch_evidence

NEAR_HULL_EV_PER_ATOM = 0.005
PARENT_STRATEGIES = {"composition", "competing_phase", "tm_ordering"}


def parent_evidence(state, branches, *, boundary=None, references=None):
    """Return auditable current-model parent records; never fall back to old models."""
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    model = state.get("active_model_version")
    if not model or diagram.get("status") != "completed" or diagram.get("model_version") != model:
        raise ValueError("near_hull_requires_current_model_phase_diagram")
    if not diagram.get("version"):
        raise ValueError("near_hull_requires_versioned_phase_diagram")
    entries = {row.get("record_id"): row for row in diagram.get("entries") or []}
    selected, phase_changed = [], []
    for row in existing_branch_evidence(state, branches):
        gap = row.get("current_model_ehull_ev_per_atom")
        if gap is None or gap > NEAR_HULL_EV_PER_ATOM:
            continue
        matches = [
            entries[rid]
            for rid in row["result_record_ids"]
            if rid in entries and entries[rid]["ehull"] <= NEAR_HULL_EV_PER_ATOM
        ]
        if any(item.get("phase") != row.get("phase") for item in matches):
            phase_changed.append(row["branch_id"])
        matches = [
            item
            for item in matches
            if item.get("phase") == row.get("phase")
            and all(
                item.get(k)
                for k in ("record_id", "structure_id", "structure_path", "structure_sha256")
            )
        ]
        if not matches:
            continue
        best = min(matches, key=lambda item: item["ehull"])
        selected.append(
            {
                "branch_id": row["branch_id"],
                "model_version": model,
                "phase_diagram_version": diagram["version"],
                "record_id": best["record_id"],
                "structure_id": best["structure_id"],
                "structure_path": best["structure_path"],
                "structure_sha256": best["structure_sha256"],
                "phase": best["phase"],
                "branch_x": row["x"],
                "branch_H": branches[row["branch_id"]].get("H"),
                "ehull_ev_per_atom": best["ehull"],
            }
        )
    invalid_structures = []
    if boundary is not None and references is not None:
        from phase_agent.science.structures.inherit_near_hull_occupancy import (
            verify_parent_matches_branch,
        )

        verified = []
        for source in selected:
            try:
                verify_parent_matches_branch(
                    source, branches[source["branch_id"]], boundary, references
                )
            except (OSError, ValueError, KeyError) as error:
                invalid_structures.append({"branch_id": source["branch_id"], "reason": str(error)})
            else:
                verified.append(source)
        selected = verified
    return {
        "invalid_parent_structures_excluded": invalid_structures,
        "model_version": model,
        "phase_diagram_version": diagram["version"],
        "threshold_ev_per_atom": NEAR_HULL_EV_PER_ATOM,
        "parents": selected,
        "phase_changed_branch_ids_excluded": phase_changed,
    }

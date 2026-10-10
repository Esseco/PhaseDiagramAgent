"""Build DFT context from recovered MC structures, never original initial files."""

from pathlib import Path


def post_mc_candidates(state, model_version):
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if diagram.get("model_version") != model_version or diagram.get("status") != "completed":
        return []
    entries = diagram.get("entries") or []
    by_branch = {}
    for task in state.get("tasks") or []:
        if (
            task.get("stage") == "deep_search"
            and task.get("status") == "completed"
            and task.get("model_version") == model_version
        ):
            prior = by_branch.get(task.get("branch_id"))
            if prior is None or int(task.get("segment_index") or 0) >= int(
                prior.get("segment_index") or 0
            ):
                by_branch[task.get("branch_id")] = task
    candidates = []
    for task in by_branch.values():
        output = task.get("outputs") or {}
        path = output.get("structure_path") or output.get("final_structure_path")
        matches = [
            entry
            for entry in entries
            if entry.get("structure_path") == path
            and entry.get("phase_identification_status") == "identified"
            and entry.get("ehull_unit") == "eV/atom"
            and entry.get("ehull") is not None
        ]
        if not path or not Path(path).is_file() or len(matches) != 1:
            continue
        entry = matches[0]
        candidates.append(
            {
                "candidate_id": task.get("structure_id"),
                "structure_id": task.get("structure_id"),
                "branch_id": task.get("branch_id"),
                "structure_path": path,
                "composition": entry.get("composition"),
                "atom_count": entry.get("atom_count")
                or sum((entry.get("composition") or {}).values()),
                "phase": entry.get("phase"),
                "x_Na_per_O2": entry.get("x_Na_per_O2"),
                "hull_impact": 1.0 / (1.0 + max(0.0, float(entry["ehull"])) / 0.01),
                "predicted_Ehull": entry["ehull"],
                "ehull_unit": "eV/atom",
                "phase_diagram_version": diagram.get("version"),
                "qbc": output.get("qbc") or {"status": "not_configured"},
            }
        )
    # Candidate IDs must be unique for the existing selection contract.
    ids = [row["candidate_id"] for row in candidates]
    return [
        row for row in candidates if row["candidate_id"] and ids.count(row["candidate_id"]) == 1
    ]

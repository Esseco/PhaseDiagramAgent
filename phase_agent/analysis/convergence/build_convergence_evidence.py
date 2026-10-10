"""Build auditable convergence evidence from ledger and workflow state."""


def build_convergence_evidence(state, manager, *, budget_remaining=None):
    structures = manager.data.get("structures", {})
    branches = manager.data.get("branches", {})
    completed_dft = 0
    deep_search_branches = set()
    covered_branches = set()
    for structure in structures.values():
        branch_id = structure.get("branch_id")
        if branch_id:
            covered_branches.add(branch_id)
        history = structure.get("stage_history") or {}
        if history.get("dft_single_point") or history.get("dft_relax"):
            completed_dft += 1
        if history.get("deep_search") and branch_id:
            deep_search_branches.add(branch_id)
    total_structures = len(structures)
    total_branches = len(branches)
    hull_changes = list(state.get("hull_changes") or [])
    dft_diagram = (state.get("phase_diagrams") or {}).get("dft") or {}
    hull_version = dft_diagram.get("version") if dft_diagram.get("status") == "completed" else None
    previous_hull_version = state.get("convergence_hull_version")
    if hull_version is not None and previous_hull_version is not None:
        hull_changes.append(0.0 if hull_version == previous_hull_version else None)
    return {
        "tasks": list(state.get("pending_tasks") or []),
        "budget_remaining": budget_remaining,
        "dft_validation_fraction": completed_dft / total_structures if total_structures else None,
        "dft_validation_count": completed_dft,
        "coverage_fraction": len(covered_branches) / total_branches if total_branches else None,
        "open_branches": total_branches - len(deep_search_branches) if total_branches else None,
        "hull_changes": hull_changes,
        "convergence_hull_version": hull_version or previous_hull_version,
    }

"""Quality tier exports and exact-record exclusion of stale candidates."""

from phase_agent.analysis.phase.phase_quality_tiers import annotate_dispositions

__all__ = ["annotate_dispositions", "filter_excluded_candidates"]


def filter_excluded_candidates(state, candidates):
    """Never carry a removed result back to DFT selection via a cached pool."""
    from phase_agent.analysis.phase.phase_boundary import (
        phase_boundary_rule,
        outside_phase_boundary,
    )

    rule = phase_boundary_rule(state)
    cleanup = state.get("phase_boundary_cleanup") or {}
    boundary_ids = (
        set(cleanup.get("structure_ids") or []) if cleanup.get("phase_boundary") == rule else set()
    )
    candidates = [
        r
        for r in candidates
        if (r.get("structure_id") or r.get("candidate_id")) not in boundary_ids
        and not (
            rule and r.get("composition") and r.get("phase") and outside_phase_boundary(r, rule)
        )
    ]
    model = state.get("active_model_version")
    records = [
        r
        for r in (state.get("phase_records") or [])
        + (state.get("phase_quality_archived_records") or [])
        if str(r.get("status", "")).startswith("excluded_")
        and (r.get("model_version") or r.get("source_version")) == model
    ]
    paths = {r.get("structure_path") for r in records} - {None}
    task_ids = {r.get("source_task_id") for r in records} - {None}
    removed_ids = {r.get("record_id") for r in records}
    active_ids = {
        r.get("record_id")
        for r in state.get("phase_records", [])
        if r.get("status", "completed") == "completed"
    }
    return [
        r
        for r in candidates
        if (
            r.get("record_id") in active_ids
            or (r.get("model_version") and r["model_version"] != model)
            or (r.get("source_search_task_id") and r["source_search_task_id"] not in task_ids)
        )
        or not (
            r.get("record_id") in removed_ids
            or r.get("structure_path") in paths
            or r.get("source_search_task_id") in task_ids
        )
    ]

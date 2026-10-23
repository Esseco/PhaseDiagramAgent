"""Explicit backed-up removal of records outside a project's configured phase boundary."""

from copy import deepcopy

from phase_agent.analysis.phase.phase_boundary import phase_boundary_rule, outside_phase_boundary
from phase_agent.analysis.phase.phase_quality_actions import backup_selection
from phase_agent.analysis.phase.phase_quality_review import exclude_records


def cleanup_phase_boundary(state, manager, *, backup_root, human_requested=False, directory=None):
    if human_requested is not True:
        raise ValueError("清理项目边界外数据需要明确人工指令")
    rule = phase_boundary_rule(state)
    if not rule:
        raise ValueError("缺少明确相/组分边界，不能清理")
    record_ids = {
        r["record_id"]
        for r in state.get("phase_records") or []
        if outside_phase_boundary(r, rule) and not str(r.get("status", "")).startswith("excluded_")
    }
    structure_ids, outside_ids = set(), set()
    for sid, row in manager.data["structures"].items():
        branch = manager.data["branches"][row["branch_id"]]
        if outside_phase_boundary(
            {
                "composition": row.get("composition") or branch.get("composition"),
                "phase": branch["P"],
            },
            rule,
        ):
            outside_ids.add(sid)
            if sid in branch.get("structure_ids", []):
                structure_ids.add(sid)
    if not record_ids and not structure_ids:
        return {
            "status": "already_clean",
            "state": deepcopy(state),
            "excluded_record_ids": [],
            "removed_branch_structure_ids": [],
            "submitted": False,
            "backup_path": (state.get("phase_boundary_cleanup") or {}).get("backup_path"),
        }
    reason = "用户指定项目相/Na组分边界：边界外结构退出活动项目，相关相图记录排除"
    backup = backup_selection(
        state,
        manager,
        record_ids,
        structure_ids,
        backup_root,
        source="human_boundary_cleanup",
        reason=reason,
    )
    result = exclude_records(
        state,
        manager,
        record_ids,
        structure_ids,
        directory=directory,
        entry={
            "model_version": state.get("active_model_version"),
            "hull_version": state["phase_diagrams"]["mlip"]["version"],
            "reason": reason,
            "source": "human_boundary_cleanup",
            "human_requested": True,
            "backup_path": str(backup),
            "phase_boundary": rule,
        },
    )
    current = result["state"]
    current["phase_boundary_cleanup"] = {
        "phase_boundary": deepcopy(rule),
        "structure_ids": sorted(outside_ids),
        "backup_path": str(backup),
    }
    # All formal candidate caches must follow the active membership change.
    from phase_agent.analysis.phase.phase_quality_disposition import filter_excluded_candidates

    for field in ("qbc_candidates", "branch_candidates"):
        current[field] = [
            r
            for r in current.get(field) or []
            if (r.get("structure_id") or r.get("candidate_id")) not in structure_ids
        ]
        if field == "qbc_candidates":
            current[field] = filter_excluded_candidates(current, current[field])
    result["backup_path"] = str(backup)
    return result

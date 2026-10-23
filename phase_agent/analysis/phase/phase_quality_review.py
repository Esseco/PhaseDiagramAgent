"""Explicit human numbered review API; no autonomous rejection or file deletion."""

from copy import deepcopy


def apply_numbered_exclusions(
    state, manager, report, *, bundle_id, numbers, reason, human_requested=False
):
    """Return changed state; caller persists state/manager using its existing transaction.

    Remove branch active membership, retaining historical structure and raw task records.
    This API is intentionally explicit and is never called by the diagnostic scorer.
    """
    if human_requested is not True or not isinstance(reason, str) or not reason.strip():
        raise ValueError("需要明确人工选择和审查原因")
    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    model = state.get("active_model_version")
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    from phase_agent.analysis.phase.phase_quality import RULE_VERSION, snapshot_evidence_digest
    from phase_agent.analysis.phase.phase_quality_policy import effective_policy

    if (
        report.get("bundle_id") != bundle_id
        or report.get("rule_version") != RULE_VERSION
        or report.get("screening_policy") != effective_policy(state)
        or report.get("snapshot_evidence_digest") != snapshot_evidence_digest(snapshot)
        or report.get("model_version") != model
        or report.get("hull_version") != snapshot.get("version")
        or report.get("import_checksum") != batch.get("import_checksum")
        or not batch.get("import_checksum")
        or batch.get("status") != "completed"
    ):
        raise ValueError("审查报告与本轮、当前模型或凸包不一致，请重新核对")
    requested = set(numbers)
    if not requested or any(isinstance(n, bool) or not isinstance(n, int) for n in requested):
        raise ValueError("需要有效的审查编号")
    from phase_agent.analysis.phase.phase_quality_markdown import review_groups

    groups = [g for g in review_groups(report) if g["number"] in requested]
    if len(groups) != len(requested):
        raise ValueError("存在未知审查编号")
    selected_ids = {rid for g in groups for rid in g["review_ids"]}
    selected = [r for r in report["structures"] if r["review_id"] in selected_ids]
    tasks = [
        t
        for t in state.get("tasks") or []
        if t.get("relax_mc_bundle_id") == bundle_id and t.get("model_version") == model
    ]
    task_ids = {t["task_id"] for t in tasks}
    records = state.get("phase_records") or []
    selected_records, structure_ids = set(), set()
    for item in selected:
        matching = [
            r
            for r in records
            if r.get("record_id") in item["record_ids"]
            and r.get("source_task_id") in task_ids
            and r.get("energy_method") == "mlip"
            and (r.get("model_version") or r.get("source_version")) == model
        ]
        if not matching:
            raise ValueError("所选项没有绑定到本轮的MLIP记录，不能扩大剔除范围")
        selected_records.update(r["record_id"] for r in matching)
        structure_ids.update(r["structure_id"] for r in matching if r.get("structure_id"))
    if any(sid not in manager.data["structures"] for sid in structure_ids):
        raise ValueError("所选结构未登记，需先核对台账")
    return exclude_records(
        state,
        manager,
        selected_records,
        structure_ids,
        entry={
            "bundle_id": bundle_id,
            "model_version": model,
            "hull_version": report["hull_version"],
            "numbers": sorted(requested),
            "reason": reason.strip(),
            "human_requested": True,
        },
    )


def exclude_records(state, manager, record_ids, structure_ids, *, entry, directory=None):
    """Exact record tombstones and inactive branch membership, with historical registration."""
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    selected_records = set(record_ids)
    model = entry["model_version"]
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    records = state.get("phase_records") or []
    if entry.get("source") != "human_boundary_cleanup":
        from phase_agent.analysis.phase.phase_quality import _points
        from phase_agent.analysis.phase.phase_quality_policy import effective_policy
        from phase_agent.analysis.phase.phase_quality_endpoints import assert_endpoint_removal

        points = _points(snapshot, [])
        removed = {
            p["key"]
            for p in points
            if any(r.get("record_id") in selected_records for r in p["rows"])
        }
        assert_endpoint_removal(points, effective_policy(state), removed)
    if any(sid not in manager.data["structures"] for sid in structure_ids):
        raise ValueError("所选结构未登记，需先核对台账")
    current = deepcopy(state)
    entry = {
        **deepcopy(entry),
        "record_ids": sorted(selected_records),
        "structure_ids": sorted(structure_ids),
    }
    current.setdefault("phase_quality_exclusions", []).append(entry)
    archived = [r for r in current["phase_records"] if r.get("record_id") in selected_records]
    for record in archived:
        record["status"] = "excluded_phase_quality_review"
        record["phase_quality_review"] = {
            "disposition": "excluded",
            "reason": entry["reason"],
            "source": entry.get("source", "human"),
            "backup_path": entry.get("backup_path"),
        }
    if entry.get("source") != "human_boundary_cleanup":
        current.setdefault("phase_quality_archived_records", []).extend(archived)
        current["phase_records"] = [
            r for r in current["phase_records"] if r.get("record_id") not in selected_records
        ]
    updated = update_phase_diagram(
        current["phase_records"],
        active_model_version=model,
        output_directory=directory,
        output_state=current,
        parent_versions={"mlip": snapshot.get("version")},
    )
    current["phase_diagrams"].update(updated["diagrams"])
    current.setdefault("phase_diagrams_by_model", {}).update(updated["mlip_by_version"])
    # Old energies may be cached without record provenance. Rebuild, never blacklist geometry.
    current["qbc_candidates"] = []
    # Any old branch pool may contain an excluded result; the current hull is authoritative.
    current["current_branch_hull_by_model"] = {}
    current["current_branch_hull_version"] = None
    current.pop("current_state_snapshot", None)
    current.pop("agent_state_summary", None)
    removed_members = []
    for sid in structure_ids:
        structure = manager.data["structures"][sid]
        if any(
            r.get("structure_id") == sid and r.get("status", "completed") == "completed"
            for r in current["phase_records"]
        ):
            continue  # Other effective records still own this registration.
        branch = manager.data["branches"][structure["branch_id"]]
        branch["structure_ids"] = [value for value in branch["structure_ids"] if value != sid]
        removed_members.append(sid)
        structure["metadata"] = structure.get("metadata") or {}
        structure["metadata"]["phase_quality_exclusion"] = {
            **{
                k: deepcopy(entry.get(k))
                for k in ("model_version", "source", "reason", "backup_path")
            },
            "record_ids": sorted(
                r["record_id"]
                for r in records
                if r.get("record_id") in selected_records and r.get("structure_id") == sid
            ),
        }
        if entry.get("source") != "human_boundary_cleanup":
            manager.data.setdefault("phase_quality_archived_structures", {}).setdefault(
                sid, []
            ).append(deepcopy(structure))
            del manager.data["structures"][sid]
    current["coverage"] = manager.coverage()
    manager.data.setdefault("phase_quality_review_log", []).append(deepcopy(entry))
    return {
        "status": "completed",
        "state": current,
        "excluded_record_ids": sorted(selected_records),
        "removed_branch_structure_ids": sorted(removed_members),
        "retained_shared_structure_ids": sorted(set(structure_ids) - set(removed_members)),
        "submitted": False,
    }

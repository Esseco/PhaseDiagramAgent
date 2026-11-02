import pytest

from phase_agent.evidence.near_hull_parents import parent_evidence


def evidence_state():
    rows = []
    tasks = []
    records = []
    branches = {}
    for i, (gap, phase, version) in enumerate(
        [
            (0.005, "O3", "m"),
            (0.0051, "O3", "m"),
            (0.001, "P3", "m"),
            (0.001, "O3", "old"),
        ]
    ):
        bid, sid, tid, rid = f"b{i}", f"s{i}", f"t{i}", f"r{i}"
        branches[bid] = {"P": "O3", "x": "1/2"}
        tasks.append({"task_id": tid, "branch_id": bid, "structure_id": sid})
        records.append({"record_id": rid, "source_task_id": tid})
        rows.append(
            {
                "record_id": rid,
                "structure_id": sid,
                "structure_path": f"{sid}.vasp",
                "structure_sha256": "abc",
                "phase": phase,
                "ehull": gap,
                "ehull_unit": "eV/atom",
                "source_version": version,
                "phase_identification_status": "identified",
            }
        )
    return {
        "active_model_version": "m",
        "tasks": tasks,
        "phase_records": records,
        "phase_diagrams": {
            "mlip": {"status": "completed", "model_version": "m", "version": "h1", "entries": rows}
        },
    }, branches


def test_only_current_version_at_or_below_5_mev_and_actual_phase_is_selected():
    state, branches = evidence_state()
    result = parent_evidence(state, branches)
    assert [item["branch_id"] for item in result["parents"]] == ["b0"]
    assert result["parents"][0]["record_id"] == "r0"
    assert result["threshold_ev_per_atom"] == 0.005
    assert result["phase_changed_branch_ids_excluded"] == ["b2"]
    assert result["phase_diagram_version"] == "h1"


def test_stale_diagram_and_missing_version_fail_closed():
    state, branches = evidence_state()
    state["phase_diagrams"]["mlip"]["model_version"] = "old"
    with pytest.raises(ValueError, match="current_model"):
        parent_evidence(state, branches)
    state["phase_diagrams"]["mlip"]["model_version"] = "m"
    del state["phase_diagrams"]["mlip"]["version"]
    with pytest.raises(ValueError, match="versioned"):
        parent_evidence(state, branches)


def test_invalid_actual_structure_is_excluded_with_reason(monkeypatch):
    from phase_agent.science.structures import inherit_near_hull_occupancy as module

    state, branches = evidence_state()

    def reject(source, branch, boundary, references):
        raise ValueError("actual_H_does_not_match")

    monkeypatch.setattr(module, "verify_parent_matches_branch", reject)
    result = parent_evidence(state, branches, boundary={}, references={})
    assert result["parents"] == []
    assert result["invalid_parent_structures_excluded"] == [
        {"branch_id": "b0", "reason": "actual_H_does_not_match"}
    ]

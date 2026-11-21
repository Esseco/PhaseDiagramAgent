from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram


def record(rid, sid, sha, version, na, energy):
    return {
        "record_id": rid,
        "structure_id": sid,
        "structure_sha256": sha,
        "model_version": version,
        "source_version": version,
        "energy_method": "mlip",
        "energy_unit": "eV",
        "energy": energy,
        "status": "completed",
        "composition": {"Na": na, "Cr": 1, "O": 2},
        "phase": "O3",
    }


def test_latest_hull_keeps_old_unreplaced_and_provenance():
    records = [
        record("a0", "a", "aaa", "old", 0, -5),
        record("a1", "a", "aaa", "new", 0, -6),
        record("b0", "b", "bbb", "old", 1, -7),
    ]
    state = {"tasks": [], "upload_layout": {"model_rounds": {"old": 1, "new": 2}}}
    result = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )
    mixed = result["diagrams"]["mlip"]
    assert mixed["status"] == "completed" and mixed["mixed_model_versions"] is True
    assert mixed["energy_policy"] == "latest_valid_per_structure_v1"
    assert {r["record_id"] for r in mixed["entries"]} == {"a1", "b0"}
    assert {r["source_version"] for r in mixed["entries"]} == {"old", "new"}
    assert {r["record_id"] for r in result["mlip_by_version"]["old"]["entries"]} == {"a0", "b0"}


def test_recovery_lineage_excludes_invalidated_refresh_but_retains_old_data():
    records = [record("a0", "a", "aaa", "old", 0, -5),
               {**record("a1", "a", "aaa", "bad", 0, -6), "source_task_id": "invalid-task"},
               record("b0", "b", "bbb", "old", 1, -7)]
    records.append({**record("b1", "b", "bbb", "bad", 1, -8), "source_task_id": "other-bad-task"})
    state = {"tasks": [], "upload_layout": {"model_rounds": {"old": 1, "bad": 2, "new": 3}},
             "extrapolation_recovery_cycle": {"invalidated_task_ids": ["invalid-task"], "invalidated_model_versions": ["bad"]}}
    result = update_phase_diagram(records, active_model_version="new", output_state=state, latest_per_structure=True)
    assert {row["record_id"] for row in result["diagrams"]["mlip"]["entries"]} == {"a0", "b0"}


def test_mixed_hull_does_not_guess_unknown_epoch():
    records = [record("a", "a", "aaa", "unknown", 0, -5), record("b", "b", "bbb", "new", 1, -7)]
    state = {"tasks": [], "upload_layout": {"model_rounds": {"new": 2}}}
    result = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )
    assert result["diagrams"]["mlip"]["status"] == "unknown"
    assert result["diagrams"]["mlip"]["reason"] == "latest_structure_lineage_unresolved"


def test_old_frozen_plan_does_not_enable_latest_policy():
    records = [record("a", "a", "aaa", "old", 0, -5), record("b", "b", "bbb", "new", 1, -7)]
    state = {
        "tasks": [],
        "model_refresh": {"plan": {}},
        "upload_layout": {"model_rounds": {"old": 1, "new": 2}},
    }
    result = update_phase_diagram(records, active_model_version="new", output_state=state)
    assert "energy_policy" not in result["diagrams"]["mlip"]


def test_new_frozen_policy_enables_latest_hull_without_mixing_up_sources():
    records = [
        record("a0", "a", "aaa", "old", 0, -5),
        record("a1", "a", "aaa", "new", 0, -6),
        record("b0", "b", "bbb", "old", 1, -7),
    ]
    state = {
        "tasks": [],
        "model_refresh": {"plan": {"energy_policy": "latest_valid_per_structure_v1"}},
        "upload_layout": {"model_rounds": {"old": 1, "new": 2}},
    }
    result = update_phase_diagram(records, active_model_version="new", output_state=state)
    assert result["diagrams"]["mlip"]["energy_policy"] == "latest_valid_per_structure_v1"
    assert {r["record_id"] for r in result["diagrams"]["mlip"]["entries"]} == {"a1", "b0"}
    assert {r["source_version"] for r in result["diagrams"]["mlip"]["entries"]} == {"old", "new"}


def test_mixed_hull_snapshots_are_immutable(tmp_path):
    records = [
        record("a0", "a", "aaa", "old", 0, -5),
        record("a1", "a", "aaa", "new", 0, -6),
        record("b0", "b", "bbb", "old", 1, -7),
    ]
    state = {"tasks": [], "upload_layout": {"model_rounds": {"old": 1, "new": 2}}}
    first = update_phase_diagram(
        records,
        active_model_version="new",
        output_state=state,
        output_directory=tmp_path,
        latest_per_structure=True,
    )["diagrams"]["mlip"]
    second = update_phase_diagram(
        records,
        active_model_version="new",
        output_state=state,
        output_directory=tmp_path,
        latest_per_structure=True,
    )["diagrams"]["mlip"]
    assert first["path"] == second["path"] and first["created_at"] == second["created_at"]


def test_duplicate_task_ids_block_latest_hull():
    records = [record("a", "a", "aaa", "new", 0, -5), record("b", "b", "bbb", "new", 1, -7)]
    state = {
        "tasks": [{"task_id": "t"}, {"task_id": "t"}],
        "upload_layout": {"model_rounds": {"new": 2}},
    }
    hull = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )["diagrams"]["mlip"]
    assert hull["status"] == "unknown"
    assert hull["issues"][0]["reason"] == "duplicate_task_id"


def test_unknown_model_does_not_register_itself_during_publication(tmp_path):
    records = [record("a", "a", "aaa", "unknown", 0, -5), record("b", "b", "bbb", "new", 1, -7)]
    state = {"tasks": [], "upload_layout": {"model_rounds": {"new": 2}}}
    result = update_phase_diagram(
        records,
        active_model_version="new",
        output_state=state,
        output_directory=tmp_path,
        latest_per_structure=True,
    )
    assert result["diagrams"]["mlip"]["status"] == "unknown"
    assert "unknown" not in state["upload_layout"]["model_rounds"]


def test_latest_only_preview_matches_full_mixed_hull():
    records = [record("a0", "a", "aaa", "old", 0, -5), record("a1", "a", "aaa", "new", 0, -6), record("b0", "b", "bbb", "old", 1, -7)]
    state = {"tasks": [], "upload_layout": {"model_rounds": {"old": 1, "new": 2}}}
    full = update_phase_diagram(records, active_model_version="new", output_state=state, latest_per_structure=True)["diagrams"]["mlip"]
    fast = update_phase_diagram(records, active_model_version="new", output_state=state, latest_per_structure=True, latest_only=True)["diagrams"]["mlip"]
    assert fast["version"] == full["version"]
    assert fast["entries"] == full["entries"]


def test_invalid_model_cannot_republish_after_direct(tmp_path):
    import pytest
    baseline = {"status": "completed", "version": "old-hull", "model_version": "old", "entries": []}
    cycle = {"status": "awaiting_reviewed_dft_labels", "invalid_model_version": "bad", "baseline_model_version": "old", "baseline_hull_version": "old-hull", "baseline_diagram": baseline}
    state = {"active_model_version": "bad", "extrapolation_recovery_cycle": cycle}
    result = update_phase_diagram([record("bad", "a", "x", "bad", 0, -2)], output_state=state, active_model_version="bad", output_directory=tmp_path)
    assert result["diagrams"]["mlip"] == baseline
    assert list(tmp_path.iterdir()) == []
    cycle["baseline_hull_version"] = "wrong"
    with pytest.raises(ValueError, match="recovery_previous_hull_invalid"):
        update_phase_diagram([], output_state=state, active_model_version="bad")


def test_explicit_current_model_policy_never_mixes_old_energy():
    records = [
        record("a1", "a", "aaa", "new", 0, -5),
        record("b1", "b", "bbb", "new", 1, -7),
        record("old-low", "c", "ccc", "old", 0.5, -20),
    ]
    state = {
        "model_refresh": {"new_model_version": "new", "hull_source_policy": "current_model_only_v1",
                          "plan": {"energy_policy": "latest_valid_per_structure_v1", "checksum": "plan", "total_unique": 2}},
        "upload_layout": {"model_rounds": {"old": 1, "new": 2}}, "tasks": [],
    }
    mixed = update_phase_diagram(records, active_model_version="new", output_state=state,
                                 latest_per_structure=True)["diagrams"]["mlip"]
    assert mixed["status"] == "completed"
    assert mixed["energy_policy"] == "current_model_only_v1"
    assert mixed["mixed_model_versions"] is False
    assert {r["record_id"] for r in mixed["entries"]} == {"a1", "b1"}
    state["model_refresh"]["new_model_version"] = "other"
    mixed = update_phase_diagram(records, active_model_version="new", output_state=state,
                                 latest_per_structure=True)["diagrams"]["mlip"]
    assert "old-low" in {r["record_id"] for r in mixed["entries"]}

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
    state = {"tasks": [{"task_id": "t"}, {"task_id": "t"}], "upload_layout": {"model_rounds": {"new": 2}}}
    hull = update_phase_diagram(records, active_model_version="new", output_state=state, latest_per_structure=True)["diagrams"]["mlip"]
    assert hull["status"] == "unknown"
    assert hull["issues"][0]["reason"] == "duplicate_task_id"

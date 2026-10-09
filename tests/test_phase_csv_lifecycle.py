from pathlib import Path
from analysis_layer.phase.update_phase_diagram import update_phase_diagram
from analysis_layer.phase.export_current_phase_diagram import export_current_phase_diagram
from tests.test_na_eform_phase_csv import record


def test_unchanged_and_reordered_data_do_not_rewrite(tmp_path):
    rows = [record("left", 0, -3), record("right", 1, -4)]
    first = update_phase_diagram(rows, active_model_version="m1", output_directory=tmp_path)["diagrams"]["mlip"]
    csv = Path(first["csv_path"])
    old_time = csv.stat().st_mtime_ns
    json_time = Path(first["path"]).stat().st_mtime_ns
    repeated = update_phase_diagram(rows[::-1], active_model_version="m1", output_directory=tmp_path)["diagrams"]["mlip"]
    assert repeated["version"] == first["version"]
    assert csv.stat().st_mtime_ns == old_time
    assert Path(repeated["path"]).stat().st_mtime_ns == json_time
    path, _, _ = export_current_phase_diagram({"active_model_version": "m1", "phase_diagrams": {"mlip": repeated}})
    assert path == csv and path.stat().st_mtime_ns == old_time


def test_new_data_updates_current_preserving_history(tmp_path):
    rows = [record("left", 0, -3), record("right", 1, -4)]
    first = update_phase_diagram(rows, active_model_version="m1", output_directory=tmp_path)["diagrams"]["mlip"]
    archive = Path(first["archive_csv_path"])
    original = archive.read_bytes()
    newer = update_phase_diagram(rows + [record("middle", .5, -3.7)], active_model_version="m1", output_directory=tmp_path)["diagrams"]["mlip"]
    assert newer["version"] != first["version"]
    assert newer["csv_path"] == first["csv_path"]
    assert archive.read_bytes() == original
    assert len(list((tmp_path / "m1" / "phase_diagrams" / "history").glob("*.csv"))) == 2
    assert not list((tmp_path / "dft").rglob("*.csv"))


def test_feedback_without_new_data_skips_hull_rebuild(monkeypatch):
    from types import SimpleNamespace
    import execution_layer.workflows.apply_scientific_feedback as module
    state = {"phase_records": [record("left", 0, -3)], "phase_diagrams": {
        "mlip": {"model_version": "m1", "version": "saved", "status": "completed"}}}
    monkeypatch.setattr(module, "ensure_phase_identification", lambda current, manager, **kwargs: (current, {}))
    monkeypatch.setattr(module, "refresh_identified_phases", lambda current: False)
    monkeypatch.setattr(module, "coverage", lambda *args: {})
    monkeypatch.setattr(module, "summarize_agent_state", lambda current: {})
    def unexpected(*args, **kwargs):
        raise AssertionError("unchanged data must not rebuild")
    monkeypatch.setattr(module, "update_phase_diagram", unexpected)
    manager = SimpleNamespace(data={}, stages=[], STAGES=[], stage_labels={})
    first = module.apply_scientific_feedback(state, [], manager=manager, active_model_version="m1")
    repeated = module.apply_scientific_feedback(first["state"], [], manager=manager, active_model_version="m1")
    assert repeated["state"]["phase_diagrams"]["mlip"]["version"] == "saved"

"""Current Na-content phase diagram and read-only CSV request."""

import csv
import json
from pathlib import Path

import pytest

from analysis_layer.phase.update_phase_diagram import update_phase_diagram
from run.agent_api import RunWorkflowChatHandler


def record(name, x, energy, *, fe=1, method="mlip", version="m1"):
    return {"record_id": name, "structure_id": name, "structure_path": None,
            "phase": "O3", "phase_identification_status": "identified",
            "structure_sha256": name, "energy_method": method,
            "model_version": version, "composition": {"Na": x, "Fe": fe, "O": 2},
            "energy": energy, "energy_unit": "eV"}


def test_eform_hull_ground_state_and_csv(tmp_path):
    rows = [record("left", 0, -3), record("right", 1, -4),
            record("mid-ground", .5, -3.7), record("mid-high", .5, -3.5),
            record("left-high", 0, -2.8)]
    snapshot = update_phase_diagram(rows, active_model_version="m1",
                                    output_directory=tmp_path)["diagrams"]["mlip"]
    assert snapshot["csv_path"] == str(tmp_path / "m1" / "phase_diagrams" / "phase_diagram.csv")
    assert Path(snapshot["archive_csv_path"]).parent == tmp_path / "m1" / "phase_diagrams" / "history"
    by_id = {row["record_id"]: row for row in snapshot["entries"]}
    assert by_id["mid-ground"]["eform_per_O2"] == pytest.approx(-.2)
    assert by_id["mid-ground"]["ehull"] == pytest.approx(0)
    assert by_id["mid-high"]["ehull_per_O2"] == pytest.approx(.2)
    assert by_id["mid-high"]["ehull"] == pytest.approx(.2 / 3.5)
    assert by_id["mid-high"]["is_composition_ground_state"] is False
    assert by_id["mid-ground"]["is_composition_ground_state"] is True
    with open(snapshot["csv_path"], encoding="utf-8-sig", newline="") as stream:
        exported = {row["record_id"]: row for row in csv.DictReader(stream)}
    assert exported["mid-ground"]["phase"] == "O3"
    assert exported["mid-ground"]["phase_identification_status"] == "identified"
    assert float(exported["mid-ground"]["eform_eV_per_O2"]) == pytest.approx(-.2)
    assert exported["mid-ground"]["x_Na_per_O2"] == "0.5000000000"
    assert exported["mid-ground"]["is_stable"] == "True"


def test_mixed_tm_is_reported_not_merged():
    rows = [record("left", 0, -3), record("right", 1, -4, fe=.5)]
    snapshot = update_phase_diagram(rows, active_model_version="m1")["diagrams"]["mlip"]
    assert snapshot["status"] == "failed"
    assert "TM/O2" in snapshot["error"]
    assert snapshot["entries"] == []


def test_csv_request_uses_current_snapshot_without_running_workflow(tmp_path):
    snapshot = update_phase_diagram([record("left", 0, -3), record("right", 1, -4)],
        active_model_version="m1", output_directory=tmp_path)["diagrams"]["mlip"]
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active_model_version": "m1",
        "phase_diagrams": {"mlip": snapshot},
        "pending_mc_regeneration": {"directory": "untouched"}}, ensure_ascii=False), encoding="utf-8")
    calls = []
    handler = RunWorkflowChatHandler({"state_path": str(state_path),
        "phase_diagram_directory": str(tmp_path)}, workflow=lambda **kw: calls.append(kw))
    for request in ("导出当前版本csv", "导出当前版本相图", "输出最新相图"):
        reply = handler([{"role": "user", "content": request}])
        assert snapshot["version"] in reply
        assert "MC 输入重生成" not in reply
    assert calls == []
    csv_path = tmp_path / "m1" / "phase_diagrams" / "phase_diagram.csv"
    assert csv_path.is_file()
    assert json.loads(state_path.read_text(encoding="utf-8"))["active_model_version"] == "m1"
    assert json.loads(state_path.read_text(encoding="utf-8"))["pending_mc_regeneration"] == {"directory": "untouched"}


def test_missing_current_csv_is_restored_from_saved_snapshot_only(tmp_path):
    snapshot = update_phase_diagram([record("left", 0, -3), record("right", 1, -4)],
        active_model_version="m1", output_directory=tmp_path)["diagrams"]["mlip"]
    current_csv = Path(snapshot["csv_path"])
    current_csv.unlink()
    snapshot["csv_path"] = str(tmp_path / current_csv.name)  # stale legacy path
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active_model_version": "m1",
        "phase_diagrams": {"mlip": snapshot}, "pending_mc_regeneration": {"keep": True}},
        ensure_ascii=False), encoding="utf-8")
    calls = []
    handler = RunWorkflowChatHandler({"state_path": str(state_path),
        "phase_diagram_directory": str(tmp_path)}, workflow=lambda **kw: calls.append(kw))
    reply = handler([{"role": "user", "content": "导出当前版本相图"}])
    assert str(current_csv) in reply
    assert current_csv.is_file()
    with current_csv.open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["phase_diagrams"]["mlip"]["csv_path"] == str(current_csv)
    assert saved["pending_mc_regeneration"] == {"keep": True}
    assert calls == []


def test_version_folders_and_legacy_snapshot_are_not_rewritten(tmp_path):
    rows = [record("left", 0, -3), record("right", 1, -4)]
    first = update_phase_diagram(rows, active_model_version="m1")["diagrams"]["mlip"]
    legacy_json = tmp_path / f"phase_diagram_mlip_{first['version']}.json"
    legacy_csv = tmp_path / f"phase_diagram_mlip_{first['version']}.csv"
    first["path"], first["csv_path"] = str(legacy_json), str(legacy_csv)
    legacy_json.write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")
    legacy_csv.write_text("old csv stays untouched\n", encoding="utf-8")
    repeated = update_phase_diagram(rows, active_model_version="m1",
                                    output_directory=tmp_path)["diagrams"]["mlip"]
    assert repeated["path"] == str(legacy_json)
    assert legacy_csv.read_text(encoding="utf-8") == "old csv stays untouched\n"
    assert not (tmp_path / "m1" / legacy_csv.name).exists()

    other = update_phase_diagram([record("a", 0, -3, version="m2"),
        record("b", 1, -4, version="m2")], active_model_version="m2",
        output_directory=tmp_path)["diagrams"]["mlip"]
    assert Path(other["csv_path"]).parent == tmp_path / "m2" / "phase_diagrams"

import json
from pathlib import Path
from copy import deepcopy
import pytest

from execution_layer.local.migrate_workspace_layout import migrate_workspace_layout as _migrate
from config_layer.session.editable_config_location import editable_config_location
from data_layer.memory.publish_memory_views import publish_memory_views


def migrate_workspace_layout(*args, **kwargs):
    # Unit tests use isolated synthetic folders, not a running desktop service.
    return _migrate(*args, agent_port=None, **kwargs)


def fixture(root):
    frozen = {"storage": {"workspace_root": str(root), "paths": {"state": "current/state.json"}},
              "science": {"patience": 20}}
    state = {"active_model_version": "m1", "upload_layout": {"model_rounds": {"m1": 1}},
             "confirmed_config": frozen, "confirmed_config_version": "v1", "confirmed_config_hash": "same",
             "phase_diagrams": {"mlip": {"model_version": "m1", "csv_path": str(root / "outputs/m1/phase_diagrams/phase_diagram.csv")}},
             "decision_memory": {"version": 1, "records": [{"record_id": "R1"}]}}
    files = {"current/state.json": state, "current/phase_data.json": {"energy": -12.3, "remote": "/remote/current/state.json"},
             "current/branch_energy_pools.json": {}, "agent_runtime.json": {"state_path": "current/state.json", "ledger_path": "current/phase_data.json"},
             "config_session.json": {"status": "confirmed", "confirmed_snapshot": {"config": frozen}, "config": frozen},
             "search_config.project.json": {"config": frozen}, "config_snapshots/v1.json": frozen}
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    phase = root / "outputs/m1/phase_diagrams/phase_diagram.csv"
    phase.parent.mkdir(parents=True)
    phase.write_bytes(b"energy\n-12.3\n")
    return deepcopy(state)


def test_migration_keeps_science_and_frozen_configuration(tmp_path):
    old = fixture(tmp_path)
    snapshot = (tmp_path / "config_snapshots/v1.json").read_bytes()
    preview = migrate_workspace_layout(tmp_path, dry_run=True)
    assert preview["status"] == "planned" and (tmp_path / "current/state.json").is_file()
    result = migrate_workspace_layout(tmp_path)
    state = json.loads((tmp_path / "runtime/state.json").read_text(encoding="utf-8"))
    assert state["confirmed_config"] == old["confirmed_config"]
    assert state["confirmed_config_version"] == "v1" and state["confirmed_config_hash"] == "same"
    assert (tmp_path / "config/snapshots/v1.json").read_bytes() == snapshot
    phase = tmp_path / "outputs/epoch0_m1/phase_diagrams/mlip/phase_diagram.csv"
    assert phase.read_bytes() == b"energy\n-12.3\n"
    assert state["phase_diagrams"]["mlip"]["csv_path"] == str(phase)
    ledger = json.loads((tmp_path / "runtime/ledgers/phase_data.json").read_text(encoding="utf-8"))
    assert ledger == {"energy": -12.3, "remote": "/remote/current/state.json"}
    session = json.loads((tmp_path / "config/config_session.json").read_text(encoding="utf-8"))
    assert session["config"] == old["confirmed_config"]
    assert (tmp_path / "outputs/output_index.md").is_file()
    assert (tmp_path / "memory/decision_memory.json").is_file()
    assert Path(result["backup"]).is_dir()
    assert migrate_workspace_layout(tmp_path)["moves"] == []


def test_conflict_stops_before_moving(tmp_path):
    fixture(tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime/state.json").write_text("{}")
    with pytest.raises(FileExistsError):
        migrate_workspace_layout(tmp_path)
    assert (tmp_path / "current/state.json").is_file()


def test_failure_rolls_back(tmp_path, monkeypatch):
    fixture(tmp_path)
    old = (tmp_path / "current/state.json").read_bytes()
    def fail(*args):
        raise OSError("test failure")
    monkeypatch.setattr("execution_layer.local.migrate_workspace_layout._atomic_write", fail)
    with pytest.raises(OSError):
        migrate_workspace_layout(tmp_path)
    assert (tmp_path / "current/state.json").read_bytes() == old
    assert (tmp_path / "config_snapshots/v1.json").is_file()


def test_memory_view_is_idempotent_and_not_a_second_truth(tmp_path):
    state = {"decision_memory": {"version": 2}, "memory_candidates": [{"id": "a"}]}
    before = deepcopy(state)
    path = tmp_path / "runtime/state.json"
    publish_memory_views(state, path)
    view = tmp_path / "memory/decision_memory.json"
    stamp = view.stat().st_mtime_ns
    publish_memory_views(state, path)
    assert state == before and view.stat().st_mtime_ns == stamp
    assert json.loads(view.read_text(encoding="utf-8"))["generated_view"] is True


def test_editable_subdirectory_is_validated(tmp_path):
    assert editable_config_location(tmp_path, "config/search_config.project.json") == tmp_path / "config/search_config.project.json"
    with pytest.raises(ValueError):
        editable_config_location(tmp_path, "../outside.json")


def test_csv_metadata_preserves_numbers(tmp_path):
    import csv
    from execution_layer.local.workspace_csv_metadata import rewrite_csv_metadata
    path = tmp_path / "phase.csv"
    path.write_text("model_version,energy,structure_path\nm1,-1.23000,old/structure.vasp\n", encoding="utf-8")
    output = rewrite_csv_metadata(path, [("old", "new")], "epoch0")
    rows = list(csv.DictReader(output.decode("utf-8-sig").splitlines()))
    assert rows[0] == {"epoch": "epoch0", "model_version": "m1", "energy": "-1.23000", "structure_path": "new/structure.vasp"}


def test_live_agent_blocks_migration_without_changes(tmp_path, monkeypatch):
    fixture(tmp_path)
    def fail(port):
        raise RuntimeError("Agent服务仍在运行")
    monkeypatch.setattr("execution_layer.local.migrate_workspace_layout.require_agent_offline", fail)
    with pytest.raises(RuntimeError):
        _migrate(tmp_path)
    assert (tmp_path / "current/state.json").is_file()
    assert not (tmp_path / "runtime").exists()


def test_concurrent_edit_is_not_rolled_back(tmp_path, monkeypatch):
    fixture(tmp_path)
    source = tmp_path / "current/state.json"
    checks = []
    def check(port):
        checks.append(port)
        if len(checks) == 2:
            current = json.loads(source.read_text(encoding="utf-8"))
            current["user_concurrent_edit"] = True
            source.write_text(json.dumps(current), encoding="utf-8")
    monkeypatch.setattr("execution_layer.local.migrate_workspace_layout.require_agent_offline", check)
    with pytest.raises(RuntimeError):
        migrate_workspace_layout(tmp_path)
    assert json.loads(source.read_text(encoding="utf-8"))["user_concurrent_edit"] is True
    assert not (tmp_path / "runtime/state.json").exists()

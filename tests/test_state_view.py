import json
import os

import pytest

from phase_agent.persistence.state.state_view import (
    publish_state_view,
    read_state_view,
    state_view_path,
)
from phase_agent.tools.step_runner.file_protocol import write_json


def _state(model="m1"):
    return {
        "confirmed_config_version": "c1",
        "active_model_version": model,
        "tasks": [
            {
                "task_id": "t1",
                "task_key": "k1",
                "stage": "relax_and_feature",
                "status": "completed",
                "model_version": model,
                "outputs": {"large": ["x"] * 1000},
            }
        ],
        "pending_execution_policies": {},
        "generation_history": [
            {
                "task_key": "g1",
                "model_version": model,
                "new_branch_ids": ["b1"],
                "registered_ids": ["s1"],
            }
        ],
    }


def test_state_view_is_small_and_cache_bound(tmp_path):
    path = tmp_path / "state.json"
    original = _state()
    write_json(path, original)

    view = read_state_view(path)
    assert view["cache_status"] == "hit"
    assert view["task_rows"] == [
        {
            "task_id": "t1",
            "task_key": "k1",
            "batch_id": None,
            "stage": "relax_and_feature",
            "status": "completed",
            "model_version": "m1",
            "config_version": None,
        }
    ]
    assert "outputs" not in view["task_rows"][0]
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_missing_corrupt_and_stale_views_rebuild(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, _state())
    sidecar = state_view_path(path)
    sidecar.write_text("{broken", encoding="utf-8")
    assert read_state_view(path)["cache_status"] == "rebuilt"

    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(_state("m2")), encoding="utf-8")
    os.replace(replacement, path)
    view = read_state_view(path)
    assert view["cache_status"] == "rebuilt"
    assert view["active_model_version"] == "m2"


def test_old_project_without_view_is_readable(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(_state()), encoding="utf-8")
    assert not state_view_path(path).exists()
    view = read_state_view(path)
    assert view["cache_status"] == "rebuilt"
    assert state_view_path(path).is_file()


def test_control_revision_change_invalidates_view(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, _state())
    control = path.with_name("state.json.control.json")
    control.write_text(
        json.dumps(
            {
                "state_path": str(path.resolve()),
                "revision": "r1",
                "binding": {"confirmed_config_version": "c1", "active_model_version": "m1"},
                "patch": {},
            }
        ),
        encoding="utf-8",
    )
    first = read_state_view(path)
    control.write_text(
        json.dumps(
            {
                "state_path": str(path.resolve()),
                "revision": "r2",
                "binding": {"confirmed_config_version": "c1", "active_model_version": "m1"},
                "patch": {},
            }
        ),
        encoding="utf-8",
    )
    second = read_state_view(path)
    assert first["cache_status"] == "rebuilt"
    assert second["cache_status"] == "rebuilt"


def test_publish_rejects_source_change(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    write_json(path, _state())
    original = path.read_bytes()
    replacement = tmp_path / "replacement.json"

    from phase_agent.persistence.state import state_view as module

    real = module.build_state_view

    def replace_during_build(state, state_path):
        replacement.write_bytes(original)
        os.replace(replacement, path)
        return real(state, state_path)

    monkeypatch.setattr(module, "build_state_view", replace_during_build)
    with pytest.raises(ValueError, match="state_changed_during_view_publish"):
        publish_state_view(path, _state())


def test_view_rebuilds_after_same_metadata_inplace_change(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, _state())
    stat = path.stat()
    data = path.read_bytes()
    assert b'"active_model_version":"m1"' in data
    path.write_bytes(
        data.replace(b'"active_model_version":"m1"', b'"active_model_version":"m2"', 1)
    )
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    view = read_state_view(path)
    assert view["cache_status"] == "rebuilt"
    assert view["active_model_version"] == "m2"


def test_view_does_not_embed_scientific_snapshots_or_pending_actions(tmp_path):
    path = tmp_path / "state.json"
    source = _state()
    source["current_state_snapshot"] = {"nested_payload": "x" * 1000000}
    source["pending_execution_policies"] = {
        "p": {
            "record_id": "p",
            "status": "awaiting_approval",
            "agent_proposal": {"raw_action": {"large": "y" * 1000000}},
        }
    }
    write_json(path, source)
    view = read_state_view(path)
    assert "current_state_snapshot" not in view
    assert view["snapshot_locator"]["field"] == "current_state_snapshot"
    assert "agent_proposal" not in view["pending_execution_policies"]["p"]
    assert state_view_path(path).stat().st_size < 10000
    assert path.stat().st_size > 2000000


def test_missing_view_can_use_valid_fields_without_parsing_unrelated_history(tmp_path):
    from phase_agent.runtime.turn_process import capture_read_metrics

    path = tmp_path / "state.json"
    source = _state()
    source["event_history"] = [{"large": "x" * 1000000}]
    write_json(path, source)
    state_view_path(path).unlink()
    with capture_read_metrics() as metrics:
        view = read_state_view(path)
    assert view["status"]["task_counts"] == {"completed": 1}
    assert metrics["counts"].get("full_state_parse", 0) == 0
    assert not any(row.get("field") == "event_history" for row in metrics["details"])
    assert view["history_counts"]["event_history"] is None

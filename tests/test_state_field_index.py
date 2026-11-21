import json
import subprocess
import sys
from unittest.mock import patch

import pytest
from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state
from phase_agent.runtime.codex.context import build_codex_context
from phase_agent.tools.step_runner.file_protocol import write_json
from tests.test_codex_refresh_snapshot import make_control


def test_indexed_context_matches_full_without_loading_history(tmp_path):
    control, _ = make_control(tmp_path)
    control.chat_handler.workflow_kwargs = {"config_session": {"config": {"agent": {"allowed_tools": []}}}}
    state = {"active_model_version": "new", "tasks": [{"task_id": "1", "status": "completed", "stage": "relax_and_feature"}],
             "phase_diagrams": {"mlip": {"version": "v1", "model_version": "new", "status": "completed"}},
             "qbc_candidates": [{"candidate_id": "current"}],
             "dft_audit_candidate_ids": ["stale", "current"],
             "event_history": [{"large": "x" * 1000}]}
    write_json(control.state_path, state)
    indexed = open_indexed_state(control.state_path)
    assert indexed is not None
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=None):
        baseline = build_codex_context(control.chat_handler)
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=indexed), patch("phase_agent.runtime.codex.context.read_json", side_effect=AssertionError("full read")):
        actual = build_codex_context(control.chat_handler)
    assert actual == baseline
    assert actual["decision_context"]["independent_audit_candidate_ids"] == ["current"]
    assert "event_history" not in indexed.cache


def test_stale_and_corrupt_index_fall_back(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"a": 1, "b": 2})
    indexed = open_indexed_state(path)
    assert indexed["a"] == 1
    subprocess.run([sys.executable, "-c", "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('{}')", str(path)], check=True)
    with pytest.raises(IndexUnavailable):
        indexed["a"]
    assert open_indexed_state(path) is None
    from phase_agent.tools.step_runner.file_protocol import read_json

    read_json(path)
    write_json(path, {"a": 1})
    index = path.with_name("state.json.index.json")
    saved = json.loads(index.read_text(encoding="utf-8"))
    saved["fields"]["a"][0] += 1
    index.write_text(json.dumps(saved), encoding="utf-8")
    assert open_indexed_state(path) is None


def test_control_change_disables_index(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"workflow_control_revision": "old"})
    control = {"state_path": str(path.resolve()), "revision": "new", "binding": {}}
    path.with_name("state.json.control.json").write_text(json.dumps(control), encoding="utf-8")
    assert open_indexed_state(path) is None


def test_index_publication_failure_does_not_invalidate_state(tmp_path):
    path = tmp_path / "state.json"
    with patch("phase_agent.persistence.state.field_index.publish_index", side_effect=OSError("injected")):
        write_json(path, {"tasks": [{"task_id": "t"}]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"tasks": [{"task_id": "t"}]}
    assert open_indexed_state(path) is None


def test_incompatible_type_check_reloads_full_state(tmp_path):
    control, _ = make_control(tmp_path)
    control.chat_handler.workflow_kwargs = {"config_session": {"config": {"agent": {"allowed_tools": []}}}}
    write_json(control.state_path, {"tasks": [], "event_history": [{"id": "old"}]})
    from phase_agent.runtime.codex import context as module

    original = module._build_codex_context
    calls = []
    def uses_dict_check(handler, state):
        calls.append(1)
        try:
            if isinstance(state, dict):
                return original(handler, state)
        except Exception:
            raise AssertionError("fallback must not be swallowed")
        raise AssertionError("type check must request a full reload")
    with patch.object(module, "_build_codex_context", side_effect=uses_dict_check):
        result = build_codex_context(control.chat_handler)
    assert result["context_version"]
    assert len(calls) == 2


def test_field_corruption_cannot_be_used(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"a": {"energy": 1}})
    indexed = open_indexed_state(path)
    indexed.fields["a"][2] = "0" * 64
    with pytest.raises(IndexUnavailable, match="checksum"):
        indexed["a"]


def test_official_cross_process_write_invalidates_cached_index(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"a": 1, "b": 2})
    indexed = open_indexed_state(path)
    assert indexed["a"] == 1
    code = "from phase_agent.tools.step_runner.file_protocol import write_json;import sys;write_json(sys.argv[1], {'a':3,'b':4})"
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    with pytest.raises(IndexUnavailable):
        indexed["a"]
    fresh = open_indexed_state(path)
    assert fresh is not None and fresh["b"] == 4


def test_pending_approval_context_matches_full_read(tmp_path):
    control, _ = make_control(tmp_path)
    control.chat_handler.workflow_kwargs = {"config_session": {"config": {"agent": {"allowed_tools": []}}}}
    state = {"active_model_version": "m1", "tasks": [],
             "pending_execution_policies": {"p": {"record_id": "p", "revision": 0, "status": "awaiting_approval"}},
             "decision_memory": {"long_term": {"search_rules": ["reviewed"]}},
             "event_history": [{"id": "unrelated", "data": "x" * 2000}]}
    write_json(control.state_path, state)
    indexed = open_indexed_state(control.state_path)
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=None):
        baseline = build_codex_context(control.chat_handler)
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=indexed), patch("phase_agent.runtime.codex.context.read_json", side_effect=AssertionError("full read")):
        actual = build_codex_context(control.chat_handler)
    assert actual == baseline
    assert "event_history" not in indexed.cache


def test_pending_queue_matches_full_without_loading_history(tmp_path):
    from threading import RLock
    from types import SimpleNamespace
    from phase_agent.runtime.local_agent_control import LocalAgentControl

    path = tmp_path / "state.json"
    state = {"active_model_version": "m1", "tasks": [],
             "pending_execution_policies": {"p1": {"agent_proposal": {"raw_action": {"tool": "update_mlip"}, "recommended_action": "update_mlip"}}},
             "event_history": [{"data": "x" * 1000}]}
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=RLock(), workflow_kwargs={}))
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=None):
        baseline = control.pending()
    indexed = open_indexed_state(path)
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=indexed), patch("phase_agent.runtime.local_agent_control.read_json", side_effect=AssertionError("full read")):
        actual = control.pending()
    assert actual == baseline and actual["count"] == 1
    assert "event_history" not in indexed.cache


def test_index_publication_rejects_a_different_source_version(tmp_path):
    from phase_agent.persistence.state.field_index import publish_index
    from phase_agent.persistence.state.read_versions import read_identity

    path = tmp_path / "state.json"
    write_json(path, {"a": 1})
    old_index = json.loads(path.with_name("state.json.index.json").read_text(encoding="utf-8"))
    old_identity = read_identity(path)
    path.write_text('{"a":2}', encoding="utf-8")
    with pytest.raises(IndexUnavailable, match="state changed"):
        publish_index(path, old_index["fields"], old_identity)
    assert open_indexed_state(path) is None


def test_view_publication_rejects_a_different_source_version(tmp_path):
    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.persistence.state.state_view import publish_state_view

    path = tmp_path / "state.json"
    state = {"tasks": []}
    write_json(path, state)
    old_identity = read_identity(path)
    path.write_text('{"tasks":[{"task_id":"new"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="state_changed_before_view_publish"):
        publish_state_view(path, state, expected_identity=old_identity)


def test_view_rebuild_rejects_state_changed_during_load(tmp_path):
    from phase_agent.persistence.state.state_view import read_state_view

    path = tmp_path / "state.json"
    path.write_text('{"tasks":[]}', encoding="utf-8")
    def loader(source, default):
        old = json.loads(source.read_text(encoding="utf-8"))
        source.write_text('{"tasks":[{"task_id":"new"}]}', encoding="utf-8")
        return old
    with pytest.raises(ValueError, match="state_changed_during_view_build"):
        read_state_view(path, state_loader=loader)


def test_index_field_io_error_falls_back_to_full_context(tmp_path, monkeypatch):
    from pathlib import Path

    control, _ = make_control(tmp_path)
    control.chat_handler.workflow_kwargs = {"config_session": {"config": {"agent": {"allowed_tools": []}}}}
    write_json(control.state_path, {"tasks": [], "active_model_version": "m1"})
    with patch("phase_agent.persistence.state.field_index.open_indexed_state", return_value=None):
        baseline = build_codex_context(control.chat_handler)
    original = Path.open
    errors = []
    def flaky(path, mode="r", *args, **kwargs):
        if path == control.state_path and mode == "rb" and not errors:
            errors.append(1)
            raise OSError("injected")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", flaky)
    assert build_codex_context(control.chat_handler) == baseline
    assert errors == [1]


def test_context_reloads_after_cross_process_change_during_read(tmp_path):
    control, _ = make_control(tmp_path)
    control.chat_handler.workflow_kwargs = {"config_session": {"config": {"agent": {"allowed_tools": []}}}}
    write_json(control.state_path, {"active_model_version": "m1", "tasks": []})
    from phase_agent.runtime.codex import context as module

    original = module._build_codex_context
    calls = []
    def update_once(handler, state):
        calls.append(1)
        if len(calls) == 1:
            code = "from phase_agent.tools.step_runner.file_protocol import write_json;import sys;write_json(sys.argv[1], {'active_model_version':'m2','tasks':[]})"
            subprocess.run([sys.executable, "-c", code, str(control.state_path)], check=True)
        return original(handler, state)
    with patch.object(module, "_build_codex_context", side_effect=update_once):
        actual = build_codex_context(control.chat_handler)
    assert len(calls) == 2
    assert actual["model_version"] == "m2"


def test_control_same_metadata_rewrite_invalidates_index(tmp_path):
    from tests.test_strong_state_versions import _other_process_same_metadata

    path = tmp_path / "state.json"
    control = path.with_name("state.json.control.json")
    control.write_text(json.dumps({"state_path": str(path.resolve()), "revision": "r", "audit": "A"}), encoding="utf8")
    write_json(path, {"workflow_control_revision": "r", "action_records": []})
    assert open_indexed_state(path) is not None
    _other_process_same_metadata(control)
    assert open_indexed_state(path) is None


def test_old_index_remains_readable_without_control_and_falls_back_with_control(tmp_path):
    from phase_agent.persistence.state.field_index import _digest

    path = tmp_path / "state.json"
    index = path.with_name("state.json.index.json")
    write_json(path, {"action_records": []})
    saved = json.loads(index.read_text(encoding="utf8"))
    saved.update(version=1, digest=_digest(saved["fields"]))
    saved.pop("control_digest", None)
    index.write_text(json.dumps(saved), encoding="utf8")
    assert open_indexed_state(path) is not None

    control = path.with_name("state.json.control.json")
    control.write_text(json.dumps({"state_path": str(path.resolve()), "revision": "r"}), encoding="utf8")
    write_json(path, {"workflow_control_revision": "r", "action_records": []})
    saved = json.loads(index.read_text(encoding="utf8"))
    saved.update(version=1, digest=_digest(saved["fields"]))
    saved.pop("control_digest", None)
    index.write_text(json.dumps(saved), encoding="utf8")
    assert open_indexed_state(path) is None

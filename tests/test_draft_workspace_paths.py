from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.runtime.draft_workspace_paths import prepare_draft_workspace_paths


def test_existing_user_draft_is_not_overwritten(tmp_path):
    path = tmp_path / "search_config.project.json"
    write_json(path, {"user_setting": 17})
    original = path.read_bytes()
    session = {"status": "draft", "setup_stage": "json_ready",
               "config": {"storage": {"workspace_root": str(tmp_path)}}}
    result = prepare_draft_workspace_paths(session, {}, base=tmp_path,
        resolved_session=tmp_path / "session.json", workspace_root=tmp_path,
        workspace_defaults={})
    assert result.editable_config_path == path
    assert result.editable_config_filename == path.name
    assert path.read_bytes() == original


def test_missing_pending_workspace_returns_to_path_selection(tmp_path):
    session = {"status": "draft", "setup_stage": "awaiting_storage_confirmation",
               "editable_config_json_path": "old.json", "config": {"preserved": 17}}
    result = prepare_draft_workspace_paths(session, {}, base=tmp_path,
        resolved_session=tmp_path / "session.json", workspace_root=tmp_path,
        workspace_defaults={})
    assert result.editable_config_path is None
    assert session["setup_stage"] == "awaiting_storage_path"
    assert "editable_config_json_path" not in session
    assert session["config"]["preserved"] == 17
    assert (tmp_path / "session.json").is_file()
    assert not (tmp_path / "search_config.project.json").exists()

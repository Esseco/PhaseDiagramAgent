import json
import pytest
from phase_agent.runtime.project_remote_workspace import remote_workspace, set_remote_workspace
from phase_agent.configuration.runtime.project_template import (
    DEFAULT_PROJECT_SETTINGS,
    BOOTSTRAP_HINTS,
)


def test_new_project_has_editable_remote_location():
    assert "remote_workspace" in DEFAULT_PROJECT_SETTINGS
    assert DEFAULT_PROJECT_SETTINGS["remote_workspace"] is None
    assert any("remote_workspace" in note for note in BOOTSTRAP_HINTS["notes"])


def test_location_isolated_changeable_and_does_not_move_data(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p in [a, b]:
        p.write_text(json.dumps({"other": "keep"}))
    set_remote_workspace(a, "/remote/project-a")
    assert remote_workspace(b) is None
    result = set_remote_workspace(a, "/remote/project-a-new")
    assert result["previous"] == "/remote/project-a"
    assert not result["submitted"] and not result["uploaded"]
    assert json.loads(a.read_text())["other"] == "keep"
    assert remote_workspace(a) == "/remote/project-a-new"


@pytest.mark.parametrize("value", ["", "/", "relative", "C:/remote", "/remote/../other"])
def test_invalid_remote_does_not_modify_runtime(tmp_path, value):
    p = tmp_path / "runtime.json"
    p.write_text("{}")
    with pytest.raises(ValueError):
        set_remote_workspace(p, value)
    assert p.read_text() == "{}"


def test_mcp_remote_setter_uses_selected_project_entry():
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    class Project:
        def set_remote_workspace(self, value):
            return {"remote_workspace": value, "submitted": False}

    server = CodexMcpServer(Project())
    assert server._call_tool("phase_set_remote_workspace", {"remote_root": "/remote/a"}) == {
        "remote_workspace": "/remote/a",
        "submitted": False,
    }

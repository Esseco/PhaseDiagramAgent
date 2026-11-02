"""Saved-only MCP must not initialize, migrate or execute the project."""

import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.runtime.codex.mcp import CodexMcpServer


def call(server, name, arguments=None):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    assert "error" not in response, response
    return json.loads(response["result"]["content"][0]["text"])


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_readonly_views_preserve_files_and_skip_runtime(tmp_path, monkeypatch):
    import phase_agent.runtime.runtime_composition as composition

    monkeypatch.setattr(
        composition, "compose_runtime", lambda *a, **k: pytest.fail("must not compose runtime")
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "active_model_version": "m0",
                "tasks": [],
                "decision_memory": {"long_term": {"search_rules": ["reviewed local rule"]}},
                "pending_execution_policies": {
                    "p1": {
                        "agent_proposal": {
                            "raw_action": {"tool": "update_mlip"},
                            "recommended_action": "update_mlip",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(
        json.dumps(
            {"state_path": "state.json", "local_path_relocations": [{"from": "old", "to": "new"}]}
        ),
        encoding="utf-8",
    )
    before = snapshot(tmp_path)
    server = CodexMcpServer(runtime_config=runtime, read_only=True)
    assert call(server, "phase_status")["model_version"] == "m0"
    assert call(server, "phase_memory")["active"]["search_rules"] == ["reviewed local rule"]
    assert call(server, "phase_pending")["count"] == 1
    assert snapshot(tmp_path) == before
    names = {
        t["name"]
        for t in server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"][
            "tools"
        ]
    }
    assert names == {
        "phase_status",
        "phase_memory",
        "phase_pending",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
    }


@pytest.mark.parametrize(
    "tool",
    [
        "phase_propose",
        "phase_decide",
        "phase_plot",
        "phase_create_project",
        "phase_team_start",
        "phase_team_propose",
        "phase_memory_propose",
        "phase_memory_review",
        "phase_review_relax_mc_return",
        "phase_set_remote_workspace",
        "phase_context",
    ],
)
def test_readonly_rejects_direct_mutations_even_if_not_discovered(tool):
    server = CodexMcpServer(read_only=True)
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": {}},
        }
    )
    assert "read_only" in response["error"]["message"]


def test_readonly_requires_existing_state_without_creating_it(tmp_path):
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text(json.dumps({"state_path": "missing.json"}), encoding="utf-8")
    before = snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        CodexMcpServer(runtime_config=runtime, read_only=True)
    assert snapshot(tmp_path) == before

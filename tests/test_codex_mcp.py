import json
import tempfile
from pathlib import Path

from phase_agent.runtime.codex.mcp import CodexMcpServer


class FakeCodex:
    def recover(self):
        return {"status": "observed", "submitted": False}

    def context(self):
        return {"context_version": "v1", "allowed_tools": ["pause_search"]}

    def pending(self):
        return {"pending": []}

    def propose(self, action, *, conversation_id="codex-offline"):
        return {"status": "awaiting_approval", "action": action, "conversation_id": conversation_id}

    def decide(self, decision, *, plan_id, state_version, proposal_hash, comment=""):
        return {"decision": decision, "plan_id": plan_id, "state_version": state_version}

    def config(self):
        return {"config": {"project": "Cr"}}


def test_mcp_lists_only_existing_codex_control_tools():
    server = CodexMcpServer(FakeCodex())
    result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    names = {tool["name"] for tool in result["result"]["tools"]}
    assert names == {
        "phase_status",
        "phase_recover",
        "phase_team_start",
        "phase_team_reconcile",
        "phase_refresh_dft_candidates",
        "phase_team_read",
        "phase_team_advise",
        "phase_team_draft",
        "phase_team_simulate",
        "phase_team_feedback",
        "phase_team_propose",
        "phase_memory_propose",
        "phase_memory_review",
        "phase_review_relax_mc_return",
        "phase_hpc_recovery",
        "phase_context",
        "phase_set_remote_workspace",
        "phase_propose_recovery",
        "phase_pending",
        "phase_memory",
        "phase_propose",
        "phase_decide",
        "phase_report",
        "phase_plot",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
        "phase_create_project",
        "phase_takeover_repo",
    }


def test_mcp_rejects_control_calls_until_a_project_is_selected():
    with tempfile.TemporaryDirectory(dir=r"E:\Py-Code") as root:
        server = CodexMcpServer(project_roots=[root], codex_factory=lambda _: FakeCodex())
        result = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "phase_context", "arguments": {}},
            }
        )
        assert result["error"]["code"] == -32000
        assert "选择项目" in result["error"]["message"]


def test_mcp_lists_and_selects_a_project_runtime():
    with tempfile.TemporaryDirectory(dir=r"E:\Py-Code") as root:
        project = Path(root) / "NaCoO2"
        project.mkdir()
        runtime = project / "agent_runtime.json"
        runtime.write_text("{}", encoding="utf-8")
        server = CodexMcpServer(project_roots=[root], codex_factory=lambda _: FakeCodex())

        listed = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"name": "phase_list_projects", "arguments": {}},
            }
        )
        projects = json.loads(listed["result"]["content"][0]["text"])["projects"]
        assert projects == [
            {"name": "NaCoO2", "runtime_config": str(runtime.resolve()), "selected": False}
        ]

        selected = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "phase_select_project",
                    "arguments": {"runtime_config": str(runtime)},
                },
            }
        )
        payload = json.loads(selected["result"]["content"][0]["text"])
        assert payload["name"] == "NaCoO2"
        assert payload["runtime_config"] == str(runtime.resolve())

        current = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {"name": "phase_current_project", "arguments": {}},
            }
        )
        assert json.loads(current["result"]["content"][0]["text"])["runtime_config"] == str(
            runtime.resolve()
        )


def test_mcp_legacy_runtime_config_still_selects_on_startup():
    with tempfile.TemporaryDirectory(dir=r"E:\Py-Code") as root:
        runtime = Path(root) / "agent_runtime.json"
        runtime.write_text("{}", encoding="utf-8")
        server = CodexMcpServer(
            runtime_config=runtime,
            codex_factory=lambda _: FakeCodex(),
        )
        result = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 14,
                "method": "tools/call",
                "params": {"name": "phase_current_project", "arguments": {}},
            }
        )
        assert json.loads(result["result"]["content"][0]["text"])["selected"] is True


def test_mcp_creates_and_selects_a_project_without_running_science():
    with tempfile.TemporaryDirectory(dir=r"E:\Py-Code") as root:
        runtime = Path(root) / "Created" / "agent_runtime.json"

        def creator(workspace_root, execution_mode):
            assert workspace_root == str(Path(root) / "Created")
            assert execution_mode == "debug"
            runtime.parent.mkdir()
            runtime.write_text("{}", encoding="utf-8")
            return runtime

        server = CodexMcpServer(
            project_creator=creator,
            codex_factory=lambda _: FakeCodex(),
        )
        result = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 15,
                "method": "tools/call",
                "params": {
                    "name": "phase_create_project",
                    "arguments": {"workspace_root": str(runtime.parent)},
                },
            }
        )
        payload = json.loads(result["result"]["content"][0]["text"])
        assert payload["created"] is True
        assert payload["selected"] is True
        assert payload["runtime_config"] == str(runtime.resolve())


def test_mcp_confirms_repo_takeover_without_selecting_a_project():
    repo = Path(r"E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram")
    server = CodexMcpServer(repo_root=repo)
    result = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/call",
            "params": {"name": "phase_takeover_repo", "arguments": {}},
        }
    )
    payload = json.loads(result["result"]["content"][0]["text"])
    assert payload["status"] == "ready"
    from phase_agent.runtime.codex.context import HPC_APPROVAL_POLICY

    assert payload["hpc_approval_policy"] == HPC_APPROVAL_POLICY
    payload["hpc_approval_policy"]["data_transfer"]["required_data_only"] = False
    assert HPC_APPROVAL_POLICY["data_transfer"]["required_data_only"] is True
    assert payload["repo_root"] == str(repo.resolve())


def test_mcp_propose_delegates_to_existing_codex_gate():
    server = CodexMcpServer(FakeCodex())
    result = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "phase_propose",
                "arguments": {"action": {"tool": "pause_search"}, "conversation_id": "mcp"},
            },
        }
    )
    payload = json.loads(result["result"]["content"][0]["text"])
    assert payload["status"] == "awaiting_approval"
    assert payload["conversation_id"] == "mcp"


def test_mcp_rejects_unknown_tool_without_touching_control():
    server = CodexMcpServer(FakeCodex())
    result = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "execute_science", "arguments": {}},
        }
    )
    assert result["error"]["code"] == -32601


def test_mcp_recovery_uses_existing_runtime_and_respects_access_modes():
    server = CodexMcpServer(FakeCodex())
    assert server._call_tool("phase_recover", {}) == {"status": "observed", "submitted": False}
    for mode in ("read_only", "analysis_only"):
        limited = CodexMcpServer(FakeCodex(), **{mode: True})
        import pytest

        with pytest.raises(ValueError, match="mode"):
            limited._call_tool("phase_recover", {})


def test_mcp_stdio_reads_and_writes_utf8_even_with_gbk_console(monkeypatch):
    import io
    from phase_agent.runtime.codex import mcp

    request = {"id": 1, "method": "echo", "params": {"text": "钠镍相图：专家意见"}}
    incoming = io.TextIOWrapper(
        io.BytesIO((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")), encoding="gbk"
    )
    buffer = io.BytesIO()
    outgoing = io.TextIOWrapper(buffer, encoding="gbk")

    class Echo:
        def __init__(self, **kwargs):
            pass

        def handle(self, message):
            assert message == request
            return {"id": message["id"], "result": message["params"]}

    monkeypatch.setattr(mcp, "CodexMcpServer", Echo)
    monkeypatch.setattr(mcp.sys, "stdin", incoming)
    monkeypatch.setattr(mcp.sys, "stdout", outgoing)
    assert mcp.main([]) == 0
    outgoing.flush()
    assert json.loads(buffer.getvalue().decode("utf-8"))["result"] == request["params"]


def test_stdio_keeps_library_logs_out_of_json_rpc(monkeypatch, capsys):
    import io
    from phase_agent.runtime.codex import mcp

    class NoisyServer:
        def __init__(self, **kwargs):
            print("constructor progress")

        def handle(self, message):
            print("5 electrostatic structures found")
            return {"jsonrpc": "2.0", "id": message["id"], "result": {"status": "ok"}}

    monkeypatch.setattr(mcp, "CodexMcpServer", NoisyServer)
    monkeypatch.setattr(mcp.sys, "stdin", io.StringIO('{"jsonrpc":"2.0","id":1}\n'))
    assert mcp.main([]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["result"]["status"] == "ok"
    assert "constructor progress" in captured.err
    assert "5 electrostatic structures found" in captured.err


def test_stdio_redirects_actual_electrostatic_library_message(monkeypatch, capsys):
    import io
    from phase_agent.runtime.codex import mcp
    from Process_Vasp.structures import structure as vasp_structure

    class FakeTransformation:
        def __init__(self, **kwargs):
            pass

        def apply_transformation(self, structure, count):
            return [{"structure": "candidate"}]

    class Server:
        def __init__(self, **kwargs):
            pass

        def handle(self, message):
            value = vasp_structure.gen_ESGS_structure(None, 1)
            return {"jsonrpc": "2.0", "id": message["id"], "result": value}

    monkeypatch.setattr(
        vasp_structure, "OrderDisorderedStructureTransformation", FakeTransformation
    )
    monkeypatch.setattr(mcp, "CodexMcpServer", Server)
    monkeypatch.setattr(mcp.sys, "stdin", io.StringIO('{"jsonrpc":"2.0","id":1}\n'))
    assert mcp.main([]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["result"] == ["candidate"]
    assert "electrostatic ground statete structures found" in captured.err

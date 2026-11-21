import json
import tempfile

import pytest
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
        "phase_recover_phase_evidence",
        "phase_publish_current_hull",
        "phase_overview",
        "phase_direct_review_report",
        "phase_perturbation_recovery_candidates",
        "phase_register_dft_failure_analysis",
        "phase_dft_recalculation_review",
        "phase_remove_quality_records",
        "phase_propose_reopened_refresh",
        "phase_register_same_frame_comparisons",
        "phase_reconcile_recovery_refresh_epoch",
        "phase_rebuild_outside_refresh_hull",
        "phase_adopt_epoch_recovery_policy",
        "phase_confirm_project_config",
        "phase_migrate_project_dft_nelm",
        "phase_set_project_dft_nelm",
        "phase_register_diagnostic_retry",
        "phase_regenerate_training_preflight",
        "phase_team_restore_dft_audit",
        "phase_settle_cancelled_dft_reservations",
        "phase_register_dft_wait_waiver",
        "phase_register_scheduler_terminal_evidence",
        "phase_register_prepared_unsubmitted_evidence",
        "phase_name_direct_dft_handoff",
        "phase_bind_direct_candidates",
        "phase_archive_failed_epoch",
        "phase_preview_completed_refresh",
        "phase_reopen_completed_refresh",
        "phase_quarantine_extrapolation_refresh",
        "phase_recover",
        "phase_recovery_diagnostic",
        "phase_team_start",
        "phase_team_reconcile",
        "phase_refresh_dft_candidates",
        "phase_authorize_exact_candidate_relax",
        "phase_authorize_unknown_gs_dft",
        "phase_register_endpoint_supplement",
        "phase_restore_relocation_metadata",
        "phase_repair_refresh_handoff",
        "phase_prepare_cpu_recovery",
        "phase_repair_dft_handoff",
        "phase_team_read",
        "phase_team_evidence",
        "phase_team_advise",
        "phase_team_clarify",
        "phase_team_draft",
        "phase_team_simulate",
        "phase_team_plot_dft",
        "phase_team_feedback",
        "phase_team_propose",
        "phase_memory_propose",
        "phase_memory_review",
        "phase_review_relax_mc_return",
        "phase_register_relax_mc_threshold_transition",
        "phase_minimize_unsent_relax_mc_delivery",
        "phase_register_relax_mc_input_attestation",
        "phase_preview_relax_mc_threshold_continuation",
        "phase_hpc_recovery",
        "phase_context",
        "phase_set_remote_workspace",
        "phase_propose_recovery",
        "phase_pending",
        "phase_memory",
        "phase_propose",
        "phase_decide",
        "phase_operation_status",
        "phase_report",
        "phase_plot",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
        "phase_create_project",
        "phase_takeover_repo",
    }


def test_mcp_rejects_control_calls_until_a_project_is_selected():
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
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
    with tempfile.TemporaryDirectory() as root:
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


def test_mcp_overview_uses_control_entry(tmp_path):
    from threading import RLock
    from types import SimpleNamespace
    from phase_agent.runtime.local_agent_control import LocalAgentControl

    state = tmp_path / "workflow_state" / "state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"active_model_version": "m1", "tasks": [{"status": "completed"}]}), encoding="utf-8")
    handler = SimpleNamespace(state_path=state, lock=RLock(), workflow_kwargs={})
    codex = FakeCodex()
    codex.control = LocalAgentControl(handler)
    server = CodexMcpServer(codex)
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "phase_overview", "arguments": {}}}
    first = server.handle(request)
    payload = json.loads(first["result"]["content"][0]["text"])
    assert payload["cache_status"] == "rebuilt"
    assert payload["model_version"] == "m1"
    second = server.handle(request)
    assert json.loads(second["result"]["content"][0]["text"])["cache_status"] == "hit"


def test_read_only_mcp_overview_does_not_create_index(tmp_path):
    state = tmp_path / "workflow_state" / "state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"active_model_version": "m1"}), encoding="utf-8")
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": "workflow_state/state.json", "ledger_path": "workflow_state/ledgers/phase_data.json"}), encoding="utf-8")
    server = CodexMcpServer(runtime_config=config, read_only=True)
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "phase_overview", "arguments": {}}}
    result = json.loads(server.handle(request)["result"]["content"][0]["text"])
    assert result["cache_status"] == "computed_read_only"
    assert result["detail_sources"]["ledger"].endswith("phase_data.json")
    assert not (state.parent / "context_overview.json").exists()
    assert not (state.parent / "langgraph_lifecycle.sqlite.lock").exists()


def test_mcp_forwards_perturbation_candidate_query():
    from types import SimpleNamespace
    server = CodexMcpServer(FakeCodex())
    server.codex.control = SimpleNamespace(perturbation_recovery_candidates=lambda max_det_H: {"max_det_H": max_det_H, "candidates": []})
    result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "phase_perturbation_recovery_candidates", "arguments": {"max_det_H": 3}}})
    payload = json.loads(result["result"]["content"][0]["text"])
    assert payload == {"max_det_H": 3, "candidates": []}


def test_mcp_forwards_bound_dft_failure_analysis():
    from types import SimpleNamespace
    server = CodexMcpServer(FakeCodex())
    server.codex.control = SimpleNamespace(register_dft_failure_analysis=lambda **kwargs: {'status':'recorded', **kwargs})
    diagnosis = {'assessment':'failure analysis','evidence':['execution.operation_outcome'],
                 'recommendation':'return_to_strategic','limitations':'logs needed'}
    result = server.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
        'name':'phase_register_dft_failure_analysis','arguments':{'failure_evidence_digest':'digest','diagnosis':diagnosis}}})
    payload = json.loads(result['result']['content'][0]['text'])
    assert payload == {'status':'recorded','failure_evidence_digest':'digest','diagnosis':diagnosis}


def test_mcp_forwards_prepared_unsubmitted_evidence():
    from types import SimpleNamespace

    server = CodexMcpServer(FakeCodex())
    server.codex.control = SimpleNamespace(
        register_prepared_unsubmitted_evidence=lambda observation: {"status": "recorded", "observation": observation}
    )
    observation = {"task_ids": ["t1"], "source": "electerm_ssh"}
    result = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "phase_register_prepared_unsubmitted_evidence", "arguments": {"observation": observation}}})
    payload = json.loads(result["result"]["content"][0]["text"])
    assert payload == {"status": "recorded", "observation": observation}


def test_cpu_recovery_mcp_dispatch_uses_control_without_scientific_runtime():
    from types import SimpleNamespace
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        return {"status": "prepared", "submitted": False}
    fake = SimpleNamespace(control=SimpleNamespace(prepare_phase_cpu_recovery=prepare))
    server = CodexMcpServer(fake)
    args = {"source_sha256": "a" * 64, "spec_sha256": "b" * 64,
            "task_count": 201, "remote_root": "/scratch/epoch4/Batch03_input"}
    assert server._call_tool("phase_prepare_cpu_recovery", args)["status"] == "prepared"
    assert calls == [args]


def test_phase_evidence_recovery_and_publication_mcp_dispatch():
    codex = FakeCodex()
    codex.recover_phase_evidence = lambda **kwargs: {"received": kwargs}
    codex.publish_phase = lambda: {"status": "completed"}
    server = CodexMcpServer(codex)
    args = {"expected_model": "m", "task_ids": ["t"]}
    assert server._call_tool("phase_recover_phase_evidence", args) == {"received": args}
    assert server._call_tool("phase_publish_current_hull", {}) == {"status": "completed"}
    with pytest.raises(ValueError):
        server._call_tool("phase_publish_current_hull", {"unexpected": True})


def test_recovery_failures_default_to_four_fields(tmp_path):
    from types import SimpleNamespace

    batch = "submissions/epoch1/delivery/Batch_output"
    archived = tmp_path / batch / "diagnostics.zip"
    detailed = {"status": "observed", "result_collection": {
        "recovered_count": 0, "transfer_errors": [{
            "archive": str(archived), "reason": "remote_dispatch_failed:queue limit\nextra error"
        }],
    }}
    control = SimpleNamespace(state_path=tmp_path / "workflow_state/state.json")
    server = CodexMcpServer(SimpleNamespace(recover=lambda: detailed, control=control))
    assert server._call_tool("phase_recover", {}) == {
        "failure_step": "dispatch", "batch_id": batch,
        "first_error": "queue limit", "resume_from": "verify_scheduler_receipt",
    }
    assert server._call_tool("phase_recover", {"include_details": True}) == detailed


def test_mcp_minimal_delivery_dispatch_forwards_user_evidence():
    class RevisionCodex(FakeCodex):
        def minimize_unsent_relax_mc_delivery(self, **arguments):
            return {"status": "revised", "arguments": arguments, "submitted": False}

    server = CodexMcpServer(RevisionCodex())
    args = {"bundle_id": "relax-mc-example", "source_sha256": "a" * 64, "confirmed_unsubmitted": True}
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "phase_minimize_unsent_relax_mc_delivery", "arguments": args}})
    result = json.loads(reply["result"]["content"][0]["text"])
    assert result == {"status": "revised", "arguments": args, "submitted": False}

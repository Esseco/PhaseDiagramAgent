from types import SimpleNamespace
from threading import RLock

import pytest

from phase_agent.runtime import chat_review
from phase_agent.runtime.codex.mcp import compact_decision_response
from phase_agent.policies.file_approval import proposal_hash
from phase_agent.tools.remote.integrity import payload_checksum
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.tools.workflows.saved_dft_input_frame import prepare_saved_dft_input_frame


@pytest.mark.parametrize("indexed", [True, False])
def test_approved_plan_read_skips_irrelevant_fields(tmp_path, monkeypatch, indexed):
    state = {
        "pending_execution_policies": {
            "p": {"agent_proposal": {"raw_action": {"tool": "pause_search"}}}
        },
        "phase_records": [{"large": "a" * 10000}],
    }
    path = tmp_path / "state.json"
    write_json(path, state)
    if not indexed:
        path.with_name("state.json.index.json").unlink()
    from phase_agent.persistence.state.field_index import IndexedState

    original_read = IndexedState.__getitem__
    reads = []

    def track(self, key):
        reads.append(key)
        return original_read(self, key)

    monkeypatch.setattr(IndexedState, "__getitem__", track)
    calls = []
    handler = SimpleNamespace(
        state_path=path,
        lock=RLock(),
        workflow_kwargs={},
        _run=lambda *args: calls.append(args) or {"status": "completed"},
    )
    result = chat_review.review_pending(
        handler,
        "p",
        "approve",
        expected_state_version=build_status_summary(state)["summary_id"],
        expected_proposal_hash=proposal_hash(
            state["pending_execution_policies"]["p"]["agent_proposal"]
        ),
        is_sensitive=lambda _: False,
    )
    assert result["status"] == "completed" and len(calls) == 1
    assert bool(reads) == indexed
    assert not {"phase_records", "invocations"} & set(reads)


@pytest.mark.parametrize(
    "block", [None, "active", "return", "backlog", "checksum", "config", "receipt", "collector"]
)
def test_saved_relax_mc_only_skips_observation_when_idle(tmp_path, monkeypatch, block):
    preview = {"config_version": "c", "model_version": "m", "branch_ids": ["B"]}
    preview["checksum"] = payload_checksum(preview)
    preview["bundle_id"] = "relax-mc-" + preview["checksum"][:16]
    action = {
        "tool": "prepare_local_batch_files",
        "target_ids": ["B"],
        "parameters": {"mode": "relax_mc_inputs", "relax_mc_preview": preview},
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "confirmed_config": {},
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m"}},
        "pending_execution_policies": {"p": {"agent_proposal": {"raw_action": action}}},
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "human_feedback": "approve",
        "snapshot": {"config_version": "c", "config": {}},
        "invocation_id": "p",
        "state_path": tmp_path / "state.json",
    }
    if block == "active":
        state["tasks"] = [{"task_id": "t", "status": "pending"}]
    if block == "backlog":
        state["processed_task_ids"] = ["t"]
    if block == "checksum":
        preview["extra"] = True
    if block == "config":
        frame["snapshot"]["config"] = {"changed": True}
    if block == "collector":
        frame["result_collector"] = object()
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return",
        lambda _: block == "return",
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report",
        lambda *args: {"status": "clear", "unsettled": ["t"] if block == "receipt" else []},
    )
    assert bool(prepare_saved_dft_input_frame(frame).get("saved_input_only")) == (block is None)


def test_codex_receipt_keeps_paths_and_omits_full_state():
    response = {
        "status": "prepared",
        "result": {
            "state": {"history": "x" * 200000},
            "module_preparation": {"upload_archive": "inputs.zip", "status": "prepared"},
            "events": [{"agent_proposal": {"massive": True}, "execution": {"submitted": False}}],
        },
    }
    compact = compact_decision_response(response, "p")
    assert compact["plan_id"] == "p" and compact["status"] == "prepared"
    assert compact["result"]["module_preparation"]["upload_archive"] == "inputs.zip"
    assert compact["result"]["events"] == [{"execution": {"submitted": False}}]
    assert compact["details_omitted"] == ["agent_proposal", "state"]


def test_pending_refresh_and_approval_reuse_verified_status_version(tmp_path, monkeypatch):
    from phase_agent.persistence.state.field_index import IndexedState
    from phase_agent.runtime.local_agent_control import LocalAgentControl

    proposal = {"raw_action": {"tool": "pause_search"}}
    state = {
        "tasks": [{"task_id": "task", "status": "pending"}],
        "pending_execution_policies": {"p": {"agent_proposal": proposal}},
    }
    path = tmp_path / "state.json"
    write_json(path, state)
    handler = SimpleNamespace(
        state_path=path, lock=RLock(), workflow_kwargs={},
        _run=lambda *args: {"status": "completed"},
    )
    control = LocalAgentControl(handler)
    read_fields = []
    original_read = IndexedState.__getitem__

    def track(self, key):
        read_fields.append(key)
        return original_read(self, key)

    monkeypatch.setattr(IndexedState, "__getitem__", track)
    first = control.pending()
    assert "tasks" in read_fields
    read_fields.clear()
    assert control.pending() == first
    chat_review.review_pending(
        handler, "p", "approve", expected_state_version=first["state_version"],
        expected_proposal_hash=proposal_hash(proposal), is_sensitive=lambda _: False,
    )
    assert "tasks" not in read_fields
    state["tasks"][0]["status"] = "completed"
    write_json(path, state)
    assert control.pending()["state_version"] != first["state_version"]
    assert "tasks" in read_fields


def test_mcp_pending_refresh_compares_full_visible_queue():
    import json
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    pending = {"pending": [{"review_card": {}}], "state_version": "same", "count": 1}
    server = CodexMcpServer(SimpleNamespace(pending=lambda: pending))

    def call(arguments):
        result = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "phase_pending", "arguments": arguments},
        })
        return json.loads(result["result"]["content"][0]["text"])

    first = call({})
    assert call({"if_queue_revision": first["queue_revision"]})["status"] == "unchanged"
    pending["pending"][0]["review_card"]["blocked"] = "live recovery guard"
    changed = call({"if_queue_revision": first["queue_revision"]})
    assert changed["queue_revision"] != first["queue_revision"]
    assert changed["pending"] == pending["pending"]


def test_default_generation_receipt_keeps_counts_and_artifacts():
    execution = {
        "tool": "generate_branches", "status": "completed",
        "result": {
            "registered": [{"branch_id": "b1"}], "coverage": [{"x": 0.5}],
            "generation_artifacts": {"json": "branch-results.json"}, "summary": {"total": 1},
        },
    }
    response = {
        "status": "completed",
        "result": {"action": {"tool": "generate_branches"}, "execution": execution,
                   "execution_result": execution},
    }
    result = compact_decision_response(response, "p")
    assert result["result"]["execution"]["result"] == {
        "registered_count": 1, "coverage_row_count": 1,
        "generation_artifacts": {"json": "branch-results.json"}, "summary": {"total": 1},
    }
    assert result["details_omitted"] == ["action", "coverage", "execution_result", "registered"]
    assert "registered" in response["result"]["execution_result"]["result"]


def test_pending_generation_keeps_frozen_quotas_and_reports_preview_omissions():
    from phase_agent.runtime.codex.mcp import compact_pending_response

    params = {
        "regions": [{"region_id": "r", "sample_count": 24, "phase": "O3"}],
        "region_preview": {
            "regions": [{"region_id": "r", "cells": [{"x": 0.5}], "legal_cells": 4}],
            "near_hull_parent_evidence": {"parents": ["p"], "phase_diagram_version": "v"},
        },
    }
    pending = {"pending": [{"recommended_action": "generate_branches", "parameters": params}]}
    compact = compact_pending_response(pending)["pending"][0]
    assert compact["parameters"]["regions"] == params["regions"]
    assert compact["parameters"]["region_preview"]["regions"] == [
        {"region_id": "r", "legal_cells": 4, "cell_count": 1}
    ]
    assert compact["parameters"]["region_preview"]["near_hull_parent_evidence"] == {
        "parent_count": 1, "phase_diagram_version": "v"
    }
    assert params["region_preview"]["regions"][0]["cells"] == [{"x": 0.5}]


def test_index_reuses_only_verified_control_header(tmp_path, monkeypatch):
    import json
    from phase_agent.persistence.state.field_index import open_indexed_state
    from phase_agent.tools.state import workflow_control

    path = tmp_path / "state.json"
    owner = str(path.resolve())
    control_path = workflow_control.control_path(path)
    control_path.write_text(
        json.dumps({"state_path": owner, "revision": "c1"}),
        encoding="utf-8",
    )
    write_json(path, {workflow_control.FIELD: "c1"})
    original = workflow_control.read_control
    calls = []

    def track(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow_control, "read_control", track)
    first = open_indexed_state(path)
    assert first is not None
    first.verify(strong=True)
    second = open_indexed_state(path)
    assert second is not None
    second.verify(strong=True)
    assert calls == [path]



def test_lifecycle_checks_pending_plan_without_parsing_unrelated_fields(tmp_path, monkeypatch):
    from phase_agent.graphs.project import runner
    from phase_agent.persistence.state.field_index import IndexedState

    path = tmp_path / "state.json"
    write_json(path, {"pending_execution_policies": {"p": {}}, "history": ["large"]})
    feedback = {"decision": "approve"}
    expected_full_read = runner._read_business_state
    keys = []
    real_get = IndexedState.__getitem__

    def track(self, key):
        keys.append(key)
        return real_get(self, key)

    def no_full_read(_):
        raise AssertionError("unneeded full state parse")

    monkeypatch.setattr(IndexedState, "__getitem__", track)
    monkeypatch.setattr(runner, "_read_business_state", no_full_read)
    assert runner._pending_approval_for_request(path, "p", feedback) == "p"
    assert runner._pending_approval_for_request(path, "other", feedback) is None
    assert "history" not in keys

    monkeypatch.setattr(runner, "_read_business_state", expected_full_read)
    path.with_name("state.json.index.json").write_text("{}", encoding="utf-8")
    assert runner._pending_approval_for_request(path, "p", feedback) == "p"

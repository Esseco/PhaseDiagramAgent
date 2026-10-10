from copy import deepcopy

from phase_agent.graphs.execution_recovery_graph import execution_recovery_report
from phase_agent.tools.state.execution_receipts import begin_execution, recovery_report
from phase_agent.tools.workflows.model_refresh_scope import (
    approved_refresh_scope,
    refresh_scope_errors,
)


def test_recovery_graph_checks_registered_artifacts_without_mutation(tmp_path):
    path = tmp_path / "state.json"
    artifact = tmp_path / "result.json"
    artifact.write_text('{"energy":1}')
    identity = {
        "invocation_id": "inv1",
        "config_version": "c1",
        "action_hash": "hash1",
        "tool": "prepare_local_batch_files",
    }
    begin_execution(path, identity)
    state = {
        "tasks": [
            {"task_id": "t1", "task_key": "k1", "status": "completed", "result_path": str(artifact)}
        ],
        "action_records": [{"record_id": "inv1", "final_action": {"task_key": "k1"}}],
        "budget_reservations": {"k1": {"status": "reserved"}},
        "phase_records": [{"record_id": "r1", "source_task_id": "t1"}],
    }
    before = deepcopy(state)
    files = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    report = execution_recovery_report(path, state)
    check = report["checks"][0]
    assert check["completed_task_ids"] == ["t1"]
    assert check["phase_record_ids"] == ["r1"]
    assert check["artifact_checks"][0]["sha256"]
    assert "复用" in check["next_action"]
    assert not check["automatic_replay_allowed"]
    assert state == before
    assert files == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    artifact.unlink()
    check = execution_recovery_report(path, state)["checks"][0]
    assert check["artifact_checks"][0]["status"] == "missing"
    assert "缺失" in check["next_action"]


def test_recovery_with_no_receipts_creates_no_files(tmp_path):
    assert execution_recovery_report(tmp_path / "state.json", {})["status"] == "clear"
    assert list(tmp_path.iterdir()) == []


def test_refresh_scope_blocks_model_config_targets_cost_and_extra_wave():
    plan = {
        "checksum": "p1",
        "new_model_version": "new",
        "maximum_supplemental_count": 1,
        "maximum_supplemental_cost": 2,
        "candidates": [{"structure_id": "s1", "operation": "predict"}],
    }
    refresh = {
        "plan": plan,
        "new_model_version": "new",
        "wave": 1,
        "supplemental_candidates": [{"structure_id": "s1", "relative_cost": 1}],
        "approved_scope": approved_refresh_scope(plan, "c1"),
    }
    assert refresh_scope_errors(refresh, "c1") == []
    assert refresh_scope_errors(refresh, "c2")
    for patch in [
        {"new_model_version": "different"},
        {"wave": 2},
        {"supplemental_candidates": [{"structure_id": "other", "relative_cost": 1}]},
        {"supplemental_candidates": [{"structure_id": "s1", "relative_cost": 3}]},
        {"supplemental_candidates": [{"structure_id": "s1", "relative_cost": float("nan")}]},
        {"approved_scope": None},
    ]:
        assert refresh_scope_errors({**refresh, **patch}, "c1")


def test_refresh_native_approval_and_business_receipt_are_consistent(tmp_path):
    from tests.test_model_refresh_workflow import setup
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    state, session, context, registry = setup(tmp_path)
    context["state_path"] = str(tmp_path / "state.json")
    pending = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert pending["status"] == "awaiting_approval"
    assert (tmp_path / "langgraph_approvals.sqlite").is_file()
    prepared = run_model_refresh_gate(
        pending["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert prepared["status"] == "prepared"
    assert recovery_report(context["state_path"], prepared["state"])["status"] == "clear"
    scope = prepared["state"]["model_refresh"]["approved_scope"]
    assert scope["remaining_waves"] == 1
    assert not scope["submit_authorized"]
    # Replaying an old approval cannot prepare the same files again.
    replay = run_model_refresh_gate(
        pending["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert replay["status"] == "approval_reconciliation_required"


def test_unsettled_execution_blocks_new_training_and_direction(tmp_path):
    from types import SimpleNamespace
    from phase_agent.graphs.project.nodes import create_lifecycle_nodes
    from phase_agent.graphs.runtime_context import WorkflowRuntime

    frame = {
        "pre_reconciled": {
            "state": {"execution_recovery_report": {"unsettled": [{"phase": "started"}]}}
        },
        "effective_config": {},
    }
    context = WorkflowRuntime(stages={})
    context.frame = frame
    runtime = SimpleNamespace(context=context)
    nodes = create_lifecycle_nodes()
    assert nodes["training_lifecycle"]({}, runtime)["phase"] == "training_waits_for_reconciliation"
    assert (
        nodes["edge_direction_review"]({}, runtime)["phase"] == "direction_waits_for_reconciliation"
    )
    assert "training_handoffs" not in context.frame


def test_recovery_subgraph_is_visible_without_changing_main_topology():
    from phase_agent.graphs.project.graph import build_react_lifecycle_graph

    graph = build_react_lifecycle_graph()
    children = [child.name for _, child in graph.get_subgraphs(recurse=True)]
    assert "execution_recovery" in children
    assert {"initialize", "observe", "wait", "analyze", "react", "finalize"}.issubset(graph.nodes)


def test_recovery_precedes_a_restored_training_wait(tmp_path):
    from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate

    path = tmp_path / "state.json"
    begin_execution(
        path,
        {
            "invocation_id": "unfinished",
            "config_version": "c",
            "action_hash": "h",
            "tool": "generate_branches",
        },
    )
    frame = {
        "recovery_question": None,
        "execution_mode": "interactive",
        "feedback": {"state": {}},
        "state_path": str(path),
        "effective_config": {},
        "recovered_count": 0,
        "collection_report": None,
        "snapshot": {},
        "pre_reconciled": {},
        "manual_wait": None,
        "rebuilding": False,
        "training_handoffs": [
            {"stage": "awaiting_activation_approval", "reason": "old waiting message"}
        ],
    }
    result = _workflow_wait_gate(frame)
    assert result["status"] == "execution_reconciliation_required"
    assert result["steps_executed"] == 0
    assert result["recovery_report"]["checks"]

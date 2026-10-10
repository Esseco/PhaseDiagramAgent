from phase_agent.graphs.approval_graph import durable_approval, build_approval_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
import pytest


def test_durable_restart_and_once_only_handoff(tmp_path):
    path = tmp_path / "state.json"
    envelope = {"invocation_id": "a", "proposal_hash": "h", "config_version": "c", "revision": 0}
    assert durable_approval(path, envelope)["status"] == "awaiting_approval"
    assert durable_approval(path, envelope)["status"] == "awaiting_approval"
    assert durable_approval(path, envelope, "approve")["status"] == "decision_delivered"
    assert (
        durable_approval(path, envelope, "approve")["status"] == "approval_reconciliation_required"
    )


def test_revised_proposal_and_other_workspace_require_new_approval(tmp_path):
    envelope = {"invocation_id": "a", "config_version": "c", "proposal_hash": "old", "revision": 0}
    assert (
        durable_approval(tmp_path / "a" / "state.json", envelope, "reject")["status"]
        == "decision_delivered"
    )
    assert (
        durable_approval(tmp_path / "a" / "other-state.json", envelope)["status"]
        == "awaiting_approval"
    )
    assert (
        durable_approval(tmp_path / "b" / "state.json", envelope)["status"] == "awaiting_approval"
    )
    assert (
        durable_approval(
            tmp_path / "a" / "state.json", {**envelope, "proposal_hash": "new", "revision": 1}
        )["status"]
        == "awaiting_approval"
    )


def test_native_interrupt_rejects_wrong_identity(tmp_path):
    config = {"configurable": {"thread_id": "x"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "approval.sqlite")) as saver:
        graph = build_approval_graph(saver)
        envelope = {
            "invocation_id": "a",
            "config_version": "c",
            "proposal_hash": "expected",
            "revision": 0,
        }
        result = graph.invoke({"envelope": envelope}, config)
        assert result["__interrupt__"]
        with pytest.raises(ValueError, match="identity mismatch"):
            graph.invoke(
                Command(
                    resume={
                        "decision": "approve",
                        "envelope": {**envelope, "proposal_hash": "wrong"},
                    }
                ),
                config,
            )


def test_workflow_approval_is_persistent_and_does_not_execute_twice(tmp_path):
    from tests.test_execution_policy import ExecutionPolicyTest
    from phase_agent.tools.workflows.run_tool_step import run_tool_step

    setup = ExecutionPolicyTest()
    setup.setUp()
    context = {"state_path": str(tmp_path / "state.json")}
    proposed = run_tool_step(
        None,
        setup.session,
        registry=setup.registry,
        agent_client=setup._agent,
        execution_mode="interactive",
        invocation_id="native-1",
        context=context,
    )
    assert proposed["status"] == "awaiting_approval"
    assert setup.calls == []
    assert (tmp_path / "langgraph_approvals.sqlite").is_file()
    approved = run_tool_step(
        proposed["state"],
        setup.session,
        registry=setup.registry,
        execution_mode="interactive",
        invocation_id="native-1",
        context=context,
        human_feedback="approve",
    )
    assert approved["status"] == "completed"
    assert len(setup.calls) == 1
    replay = run_tool_step(
        approved["state"],
        setup.session,
        registry=setup.registry,
        execution_mode="interactive",
        invocation_id="native-1",
        context=context,
        human_feedback="approve",
    )
    assert replay["idempotent_replay"]
    assert len(setup.calls) == 1
    uncertain = run_tool_step(
        proposed["state"],
        setup.session,
        registry=setup.registry,
        execution_mode="interactive",
        invocation_id="native-1",
        context=context,
        human_feedback="approve",
    )
    assert uncertain["status"] == "approval_reconciliation_required"
    assert len(setup.calls) == 1


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_continue_after_delivery_reports_reconciliation_not_new_approval(tmp_path, decision):
    path = tmp_path / "state.json"
    envelope = {
        "invocation_id": "continue",
        "proposal_hash": "h",
        "config_version": "c",
        "revision": 0,
    }
    assert durable_approval(path, envelope, decision)["status"] == "decision_delivered"
    for _ in range(2):
        result = durable_approval(path, envelope)
        assert result["status"] == "approval_reconciliation_required"
        assert result["decision"] == decision


def test_workflow_continue_after_delivery_never_repeats_tool(tmp_path):
    from tests.test_execution_policy import ExecutionPolicyTest
    from phase_agent.tools.workflows.run_tool_step import run_tool_step

    setup = ExecutionPolicyTest()
    setup.setUp()
    context = {"state_path": str(tmp_path / "state.json")}
    proposed = run_tool_step(
        None,
        setup.session,
        registry=setup.registry,
        agent_client=setup._agent,
        execution_mode="interactive",
        invocation_id="continue-delivered",
        context=context,
    )
    completed = run_tool_step(
        proposed["state"],
        setup.session,
        registry=setup.registry,
        execution_mode="interactive",
        invocation_id="continue-delivered",
        context=context,
        human_feedback="approve",
    )
    assert completed["status"] == "completed"
    uncertain = run_tool_step(
        proposed["state"],
        setup.session,
        registry=setup.registry,
        execution_mode="interactive",
        invocation_id="continue-delivered",
        context=context,
    )
    assert uncertain["status"] == "approval_reconciliation_required"
    assert len(setup.calls) == 1

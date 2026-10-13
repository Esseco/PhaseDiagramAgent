from copy import deepcopy
import json

import pytest

from phase_agent.configuration.schema.action_state_schema import normalize_action
from phase_agent.decisions.agent.action_contracts import action_contract_errors
from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.dispatch.scientific_operations import OPERATIONS, operation_catalog


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_each_operation_routes_one_existing_tool_without_mutating_input(name):
    action = {
        "operation": name,
        "tool": OPERATIONS[name]["tool"],
        "task_key": "k",
        "budget": 0,
        "parameters": {},
        "target_ids": [],
    }
    original = deepcopy(action)
    normalized = normalize_action(action)
    assert normalized["parameters"] == OPERATIONS[name]["parameters"]
    assert normalized["tool"] == OPERATIONS[name]["tool"]
    assert action == original
    assert not action_contract_errors(normalized)


@pytest.mark.parametrize(
    "action",
    [
        {"operation": "prepare_mlip_relax", "tool": "update_mlip"},
        {
            "operation": "prepare_mlip_relax",
            "tool": "prepare_local_batch_files",
            "parameters": {"mode": "mc_inputs"},
        },
        {"operation": "unknown", "tool": "pause_search"},
    ],
)
def test_operation_conflicts_are_rejected_before_call(action):
    assert action_contract_errors(action)
    with pytest.raises(ValueError):
        normalize_action(action)
    called = []
    registry = create_tool_registry({"prepare_local_batch_files": lambda **kw: called.append(kw)})
    result = execute_tool_action(action, registry=registry, context={})
    assert result["status"] == "rejected"
    assert result["operation_outcome"]["dispatch_status"] == "rejected"
    assert called == []


def test_catalog_contains_only_allowed_and_configured_handlers():
    registry = create_tool_registry({"update_mlip": lambda **_: None})
    specs = operation_catalog(["update_mlip", "prepare_local_batch_files"], registry=registry)
    assert {row["operation"] for row in specs} == {
        "prepare_mlip_training",
        "activate_mlip_candidate",
    }
    assert registry["prepare_local_batch_files"]["operations"] == []
    json.dumps(specs)


def test_request_advertises_individual_operations_without_expanding_allowed_tools():
    payload = {"mode": "autonomous_search", "state": {}, "allowed_tools": ["update_mlip"]}
    result = prepare_llm_request(payload)
    assert result["allowed_tools"] == ["update_mlip"]
    assert len(result["scientific_operations"]) == 2
    assert "scientific_operations" not in payload
    assert "scientific_operations" not in prepare_llm_request({"state": {}, "allowed_tools": []})


@pytest.mark.parametrize(
    "context_binding", [{"model_version": "old"}, {"config_version": "old"}, {"bad": "x"}]
)
def test_stale_versions_cannot_reach_handler(context_binding):
    called = []
    registry = create_tool_registry({"update_mlip": lambda **kw: called.append(kw)})
    action = {
        "operation": "prepare_mlip_training",
        "tool": "update_mlip",
        "operation_context": context_binding,
    }
    result = execute_tool_action(
        action,
        registry=registry,
        context={"config_version": "c1", "event_state": {"active_model_version": "m1"}},
    )
    assert result["status"] == "rejected"
    assert result["operation_outcome"]["dispatch_status"] == "rejected"
    assert called == []


def test_prepared_result_is_not_reported_as_scientific_completion():
    seen = []

    def handler(**kwargs):
        seen.append(kwargs["action"])
        return {
            "status": "prepared",
            "submitted": False,
            "tasks": [{"task_id": "t1", "status": "pending"}],
        }

    registry = create_tool_registry({"prepare_local_batch_files": handler})
    action = {
        "operation": "prepare_mlip_relax",
        "tool": "prepare_local_batch_files",
        "task_key": "k",
    }
    original = deepcopy(action)
    result = execute_tool_action(
        action,
        registry=registry,
        context={"config_version": "c1", "event_state": {"active_model_version": "m1"}},
    )
    assert seen[0]["parameters"]["mode"] == "relax_inputs"
    facts = result["operation_outcome"]
    assert facts["dispatch_status"] == "completed" and facts["result_status"] == "prepared"
    assert facts["task_ids"] == ["t1"]
    assert facts["model_version"] == "m1" and facts["config_version"] == "c1"
    assert facts["submitted"] is False
    assert action == original
    json.dumps(facts)


def test_graph_exposes_only_configured_operation_nodes():
    from phase_agent.graphs.actions.graph import build_tool_action_graph

    registry = create_tool_registry({"update_mlip": lambda **_: None})
    graph = build_tool_action_graph(registry=registry)
    assert "approval_gate" in graph.nodes
    assert "validate_action" in graph.nodes
    assert "tool__update_mlip" in graph.nodes
    assert {r["operation"] for r in graph.scientific_operations} == {
        "prepare_mlip_training",
        "activate_mlip_candidate",
    }


@pytest.mark.parametrize("grouped", [False, True])
def test_operation_has_a_real_node_and_keeps_approval_before_dispatch(grouped):
    from phase_agent.graphs.actions.graph import build_tool_action_graph, run_tool_action_graph

    registry = create_tool_registry({"update_mlip": lambda **_: None})
    graph = build_tool_action_graph(registry=registry, group_actions=grouped)
    nodes = dict(graph.get_subgraphs())["mlip_finetune"].nodes if grouped else graph.nodes
    assert "operation__prepare_mlip_training" in nodes
    called = []

    def initialize():
        return {
            "_graph_prepared": True,
            "action": {"tool": "update_mlip", "operation": "prepare_mlip_training"},
            "dispatch_ready": True,
        }

    def approved(frame):
        called.append("approved")
        return frame

    def dispatch(frame):
        called.append("dispatch")
        return frame

    response = run_tool_action_graph(
        tool_graph=graph,
        registry=registry,
        initialize=initialize,
        refresh=lambda f: f,
        prepare=lambda f: f,
        approve=approved,
        validate=lambda f: f,
        dispatch=dispatch,
        finish=lambda f: {"status": "prepared"},
    )
    assert response["status"] == "prepared"
    assert called == ["approved", "dispatch"]
    called.clear()
    response = run_tool_action_graph(
        tool_graph=graph,
        registry=registry,
        initialize=initialize,
        refresh=lambda f: f,
        prepare=lambda f: f,
        approve=lambda f: {"status": "awaiting_approval"},
        validate=lambda f: called.append("validation"),
        dispatch=dispatch,
        finish=lambda f: f,
    )
    assert response["status"] == "awaiting_approval"
    assert called == []


def test_cached_production_graph_has_operation_nodes_without_global_handlers():
    from phase_agent.graphs.scientific_graph import scientific_graph

    graph = scientific_graph()
    assert "scientific_feedback" in graph.observation_contracts
    children = dict(graph.get_subgraphs(recurse=True))
    assert "operation__prepare_mlip_relax" in children["react|structure_and_batch_inputs"].nodes
    assert "operation__prepare_mlip_training" in children["react|mlip_finetune"].nodes


def test_decision_backend_advertises_operations_to_injected_agent():
    from phase_agent.decisions.agent.decision_backend import request_decision_action

    received = []

    def client(payload):
        received.append(payload)
        return {"tool": "check_convergence", "budget": 0, "reason": "检查"}

    action = request_decision_action(client, {"state": {}, "allowed_tools": ["check_convergence"]})
    assert action["tool"] == "check_convergence"
    assert received[0]["scientific_operations"][0]["operation"] == "check_convergence"


def test_mc_input_operation_does_not_silently_ignore_target_scope():
    with pytest.raises(ValueError, match="operation_scope_conflict"):
        normalize_action({"operation": "prepare_mlip_mc", "target_ids": ["one-task"]})


def test_operation_cannot_charge_relax_to_dft_stage():
    with pytest.raises(ValueError, match="operation_stage_conflict"):
        normalize_action({"operation": "prepare_mlip_relax", "stage": "dft_single_point"})


def test_result_contract_reads_structured_batch_ids_and_artifacts_only():
    from phase_agent.tools.dispatch.operation_outcome import operation_outcome

    execution = {
        "status": "completed",
        "result": {
            "status": "prepared",
            "submitted": False,
            "batches": [{"task_ids": ["t1", "t1"], "directory": "/upload/input"}],
            "handoffs": [{"folder": "/upload", "return_directory": "/return"}],
            "state": {"tasks": [{"task_id": "unrelated-historical-task"}]},
        },
    }
    facts = operation_outcome({"tool": "prepare_local_batch_files"}, execution, {})
    assert facts["task_ids"] == ["t1"]
    assert {item["path"] for item in facts["artifacts"]} == {"/upload/input", "/upload", "/return"}
    assert "state" not in facts


def test_named_operation_reuses_persisted_receipt_and_cannot_execute_twice(tmp_path):
    calls = []
    registry = create_tool_registry(
        {"prepare_local_batch_files": lambda **kw: calls.append(kw) or {"status": "prepared"}}
    )
    action = normalize_action(
        {
            "operation": "prepare_mlip_relax",
            "tool": "prepare_local_batch_files",
            "task_key": "one-plan",
        }
    )
    original = deepcopy(action)
    context = {
        "state_path": str(tmp_path / "state.json"),
        "config_version": "c1",
        "invocation_id": "approval-1",
        "event_state": {"active_model_version": "m1"},
    }
    first = execute_tool_action(action, registry=registry, context=context)
    assert first["operation_outcome"]["result_status"] == "prepared"
    second = execute_tool_action(action, registry=registry, context=context)
    assert second["status"] == "execution_reconciliation_required"
    assert len(calls) == 1
    assert action == original

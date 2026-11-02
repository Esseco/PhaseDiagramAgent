from copy import deepcopy

import pytest

from tests.test_event_loop import _session
from phase_agent.configuration.schema.action_state_schema import normalize_action
from phase_agent.policies.validate_tool_action import validate_tool_action
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.workflows.run_tool_step import run_tool_step


@pytest.mark.parametrize(
    "patch",
    [
        {"budget": "invalid"},
        {"budget": float("nan")},
        {"budget": float("inf")},
        {"action_type": "update_mlip"},
        {"parameters": ["bad"]},
        {"parameters": []},
    ],
)
def test_invalid_final_action_is_structured_rejection(patch):
    action = {"tool": "check_convergence", "parameters": {}, "budget": 0, **patch}
    result = validate_tool_action(
        action, {}, _session(), create_tool_registry({"check_convergence": lambda **_: {}})
    )
    assert result["valid"] is False
    assert result["errors"]


def test_structured_human_modify_never_dispatches_invalid_action():
    calls = []
    registry = create_tool_registry(
        {
            "check_convergence": lambda action, context: (
                calls.append(action) or {"status": "completed"}
            )
        }
    )
    options = dict(
        session=_session(), registry=registry, execution_mode="interactive", invocation_id="test"
    )
    first = run_tool_step(
        {},
        agent_client=lambda _: {"tool": "check_convergence", "parameters": {}, "budget": 0},
        **options,
    )
    result = run_tool_step(
        first["state"],
        human_feedback={
            "decision": "modify",
            "comment": "approve",
            "modifications": {"budget": "invalid", "action_type": "update_mlip"},
        },
        **options,
    )
    assert result["status"] == "rejected"
    assert not calls


def test_legacy_cost_is_normalized_and_dispatched_identically():
    assert (
        normalize_action({"action_type": "pause_search", "expected_cost": {"value": 2}})["budget"]
        == 2
    )
    calls = []
    registry = create_tool_registry(
        {
            "check_convergence": lambda action, context: (
                calls.append(deepcopy(action)) or {"status": "completed"}
            )
        }
    )
    result = run_tool_step(
        {},
        _session(),
        registry=registry,
        agent_client=lambda _: {"tool": "check_convergence", "budget": 0},
        execution_mode="autonomous",
    )
    assert result["validation"]["valid"]
    assert calls[0] == result["final_action"]

from copy import deepcopy
import json

import pytest

from phase_agent.tools.workflows.compact_action_history import (
    compact_action_history,
    compact_state_history,
)
from tests.test_approval_history_performance import UncopyableHistory


@pytest.mark.parametrize("owned", [False, True])
def test_direct_and_nested_execution_states_are_removed_before_any_copy(owned):
    state = UncopyableHistory()
    record = {
        "state": state,
        "execution_result": {
            "state": state,
            "status": "completed",
            "result": {"state": state, "structure_ids": ["s"]},
        },
        "execution": {"state": state, "result": {"state": state, "seconds": 12.5}},
    }
    result = compact_action_history(record, _owned=owned)
    assert result == {
        "execution_result": {"status": "completed", "result": {"structure_ids": ["s"]}},
        "execution": {"result": {"seconds": 12.5}},
    }
    if not owned:
        assert record["execution_result"]["state"] is state
        result["execution_result"]["result"]["structure_ids"].append("new")
        assert record["execution_result"]["result"]["structure_ids"] == ["s"]


def test_repeated_refreshes_do_not_embed_previous_histories_and_keep_audit_values():
    state = {
        "tasks": [{"task_id": "t", "status": "completed", "energy": -8.2}],
        "event_history": [],
    }
    for n in range(10):
        returned = {
            "status": "completed",
            "state": deepcopy(state),
            "gpu_hours": n + 0.5,
            "approval_basis": "explicit_user_approval",
            "refresh_summary": {"completed": 1},
        }
        event = compact_action_history({"state": state, "execution_result": returned})
        state["event_history"].append(event)
    assert len(json.dumps(state)) < 3000
    assert [e["execution_result"]["gpu_hours"] for e in state["event_history"]] == [
        i + 0.5 for i in range(10)
    ]
    assert all(
        e["execution_result"]["approval_basis"] == "explicit_user_approval"
        for e in state["event_history"]
    )


def test_old_history_migration_keeps_scientific_state_and_all_audit_records():
    canonical = {
        "tasks": [{"task_id": "t", "status": "completed", "energy": -8.2}],
        "cost_history": [{"gpu_hours": 1.25}],
        "model_refresh": {"status": "completed"},
        "pending_execution_policies": {"p": {"approved": False}},
    }
    state = {
        **deepcopy(canonical),
        "event_history": [
            {
                "execution_result": {
                    "state": deepcopy(canonical),
                    "status": "completed",
                    "refresh_summary": {"structures": ["s"]},
                }
            }
        ],
    }
    compact_state_history(state, _owned=True)
    assert {k: state[k] for k in canonical} == canonical
    assert state["event_history"] == [
        {"execution_result": {"status": "completed", "refresh_summary": {"structures": ["s"]}}}
    ]


def test_restoring_legacy_checkpoint_cannot_reintroduce_direct_state_copies():
    from phase_agent.graphs.business_checkpoint import restore_frame

    scientific = {"tasks": [{"task_id": "t", "energy": -1.5}], "cost_history": [{"gpu_hours": 2}]}
    legacy = {
        **deepcopy(scientific),
        "event_history": [
            {
                "execution_result": {
                    "state": deepcopy(scientific),
                    "status": "completed",
                    "approved": True,
                }
            }
        ],
    }
    saved = {"loaded_state": legacy, "result": {"state": deepcopy(legacy), "status": "completed"}}
    before = deepcopy(saved)
    restored = restore_frame({"loaded_state": legacy}, saved)
    assert saved == before
    for state in (restored["loaded_state"], restored["result"]["state"]):
        assert state["tasks"] == scientific["tasks"]
        assert state["cost_history"] == scientific["cost_history"]
        assert state["event_history"] == [
            {"execution_result": {"status": "completed", "approved": True}}
        ]

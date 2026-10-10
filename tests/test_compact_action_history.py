"""Regression checks for bounded persisted workflow history."""

from phase_agent.tools.workflows.compact_action_history import (
    compact_action_history,
    compact_state_history,
)


def test_compact_action_history_preserves_results_without_nested_state():
    record = {
        "status": "completed",
        "execution": {"status": "completed", "result": {"state": {"tasks": [1]}, "energy": -1.2}},
        "execution_result": {"status": "completed", "result": {"state": {"tasks": [1]}, "energy": -1.2}},
    }
    compact = compact_action_history(record)
    assert compact["execution"]["result"] == {"energy": -1.2}
    assert compact["execution_result"]["result"] == {"energy": -1.2}
    assert record["execution"]["result"]["state"] == {"tasks": [1]}


def test_compact_state_history_keeps_ids_and_scientific_state():
    row = {"record_id": "r1", "execution_result": {"result": {"state": {"large": True}, "value": 7}}}
    state = {"tasks": [{"task_id": "t1"}], "action_records": [row],
             "decisions": [row], "event_history": [row], "invocations": {"r1": row}}
    compact_state_history(state)
    assert state["tasks"] == [{"task_id": "t1"}]
    assert state["invocations"]["r1"]["record_id"] == "r1"
    for saved in (state["action_records"][0], state["decisions"][0],
                  state["event_history"][0], state["invocations"]["r1"]):
        assert saved["execution_result"]["result"] == {"value": 7}

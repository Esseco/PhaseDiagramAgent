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


def test_generated_task_never_embeds_canonical_project_state_or_drops_scientific_outputs():
    from phase_agent.tools.workflows.tool_outcomes import _apply_execution_result
    payload = {"status": "completed", "state": {"tasks": [{"task_id": "real", "stage": "relax"}],
                "confirmed_config_version": "c1"}, "summary": {"registered_structures": 3},
               "outputs": {"energy": -1.2}, "actual_cost": 0}
    result, status = _apply_execution_result({}, {"tool": "generate_branches", "task_key": "g1"},
        {"status": "completed", "result": payload}, record_id="r1", formal=True)
    saved = next(row for row in result["tasks"] if row["task_id"] == "r1:task")
    assert status == "completed" and "state" not in saved
    assert saved["outputs"] == payload["outputs"] and saved["summary"] == payload["summary"]
    assert result["confirmed_config_version"] == "c1" and payload["state"]["tasks"][0]["task_id"] == "real"


def test_legacy_generated_task_compacts_only_redundant_project_snapshot():
    real = {"task_id": "r1", "stage": "deep_search", "state": {"tasks": ["scientific field"]}, "outputs": {"energy": -1}}
    audit = {"task_id": "webui:task", "status": "completed", "task_key": "g1",
             "state": {"tasks": [real], "confirmed_config_version": "c1"},
             "registered": ["s1"], "actual_cost": 2}
    state = {"tasks": [real, audit], "pending_tasks": [audit], "processed_task_ids": ["webui:task"]}
    compact_state_history(state)
    assert state["tasks"][0] is real and real["state"] == {"tasks": ["scientific field"]}
    assert state["tasks"][1] == {k: v for k, v in audit.items() if k != "state"}
    assert "state" not in state["pending_tasks"][0] and state["processed_task_ids"] == ["webui:task"]
    assert "state" in audit

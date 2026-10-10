import json
from phase_agent.graphs.studio_status_graph import graph


def test_status_graph_reads_fresh_state_without_writing(tmp_path, monkeypatch):
    config = tmp_path / "agent_runtime.json"
    state_path = tmp_path / "state.json"
    config.write_text(json.dumps({"state_path": "state.json"}))
    state = {"active_model_version": "old", "remote_finetune_jobs": {"j": {"directory": str(tmp_path), "status": "results_received",
        "training_handoff": {"direction_status": "awaiting_approval",
            "direction_proposal": {"direction": "other", "reason": "check interface"}}}}}
    state_path.write_text(json.dumps(state))
    before = state_path.read_bytes()
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    result = graph.invoke({})
    assert result["awaiting_approval"]["directions"][0]["reason"] == "check interface"
    assert result["scientific_progress"]["waiting_for"] == "direction_approval"
    assert result["project_name"] == tmp_path.name
    assert state_path.read_bytes() == before
    state["remote_finetune_jobs"]["j"]["training_handoff"]["direction_status"] = "approved"
    state_path.write_text(json.dumps(state))
    assert graph.invoke({})["awaiting_approval"]["directions"] == []


def test_status_projection_does_not_require_a_message_or_runtime(monkeypatch):
    monkeypatch.delenv("PHASE_AGENT_RUNTIME_CONFIG", raising=False)
    result = graph.invoke({})
    assert result["scientific_progress"] == {}
    assert result["awaiting_approval"] == {}

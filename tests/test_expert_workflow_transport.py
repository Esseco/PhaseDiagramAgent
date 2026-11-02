"""Real workflow wrapping must retain invocation-local expert opinions."""

from unittest.mock import Mock

from phase_agent.tools.step_runner.file_protocol import read_json
from tests.test_decision_team import FakeTeam
from tests.test_unified_react_dialogue import make_handler


def test_advance_workflow_delivers_both_opinions_to_raw_coordinator(tmp_path):
    class WorkflowTeam(FakeTeam):
        def __call__(self, request):
            response = super().__call__(request)
            if request.get("semantic_dialogue"):
                return {
                    "kind": "advance",
                    "answer": "Observe current evidence and propose one next step.",
                    "_llm_usage": response["_llm_usage"],
                }
            if request.get("mode") == "autonomous_search":
                response.update(
                    tool="check_convergence",
                    parameters={},
                    budget=0,
                    reason="Assess recorded evidence",
                    expected_purpose="Check remaining gaps",
                )
            return response

    client = WorkflowTeam()
    client.expert_client = client
    execute = Mock(side_effect=AssertionError("Planning does not execute scientific tools"))
    handler = make_handler(tmp_path, client, {"check_convergence": execute})
    reply = handler(
        [{"role": "user", "content": "Observe the current project and propose the next step."}],
        conversation_id="expert-workflow-transport",
    )
    consultation = [request for request in client.calls if request["mode"] == "team_consultation"]
    experts = [request for request in client.calls if request["mode"] == "expert_assessment"]
    coordinated = [
        request
        for request in client.calls
        if request.get("mode") == "autonomous_search" and request.get("agent_role") == "strategic"
    ]
    assert len(consultation) == 1
    assert [request["agent_role"] for request in experts] == ["exploration", "reliability"]
    assert coordinated, reply
    assert coordinated[0]["evidence_reader"]["phase"] == "select_topics"
    snapshot_id = consultation[0]["evidence_snapshot"]["snapshot_id"]
    for request in coordinated:
        context = request["decision_context"]
        assert context["evidence_snapshot_id"] == snapshot_id
        advice = context["expert_advice"]
        assert [opinion["role"] for opinion in advice] == ["exploration", "reliability"]
        assert all(opinion["snapshot_id"] == snapshot_id for opinion in advice)
        assert all(opinion["assessment"] == "Evidence is insufficient." for opinion in advice)
        assert all(opinion["limitations"] == "Synthetic evidence only." for opinion in advice)
    saved = read_json(handler.state_path, {})
    pending = saved.get("pending_execution_policies") or {}
    assert len(pending) == 1, reply
    raw = next(iter(pending.values()))["agent_proposal"]["raw_action"]
    assert raw["tool"] == "check_convergence"
    assert raw["_decision_team"]["snapshot_id"] == snapshot_id
    assert len(raw["_decision_team"]["advice"]) == 2
    assert all(
        request["decision_context"]["expert_advice"] == raw["_decision_team"]["advice"]
        for request in coordinated
    )
    execute.assert_not_called()
    assert not saved.get("tasks")

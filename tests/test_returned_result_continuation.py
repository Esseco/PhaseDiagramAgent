from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from phase_agent.graphs.dialogue.nodes import decide


@pytest.mark.parametrize(
    "present,status,expected",
    [(True, "pending", True), (False, "pending", False), (True, "completed", False)],
)
def test_continue_enters_collector_before_dialogue_model(
    tmp_path, monkeypatch, present, status, expected
):
    path = tmp_path / "result.json"
    if present:
        path.write_text("{}")
    handler = SimpleNamespace(
        state_path=tmp_path / "state.json",
        workflow_kwargs={},
        _run=Mock(return_value={"status": "awaiting_approval"}),
    )
    monkeypatch.setattr("phase_agent.graphs.dialogue.support.fresh_turn_context", lambda *args: {})
    model = Mock(return_value={"status": "answered"})
    monkeypatch.setattr("phase_agent.graphs.dialogue.decision.decide_from_saved_facts", model)
    context = SimpleNamespace(
        handler=handler,
        messages=[{"role": "user", "content": "继续"}],
        conversation_id="A",
        scientific_call=None,
    )
    result = decide(
        {
            "facts": {
                "tasks": [
                    {
                        "task_id": "old",
                        "stage": "relax_and_feature",
                        "status": status,
                        "result_path": str(path),
                    }
                ]
            },
            "message": "继续",
        },
        SimpleNamespace(context=context),
    )
    if expected:
        assert result["operation"] == "scientific_workflow"
        model.assert_not_called()
        context.scientific_call()
        handler._run.assert_called_once()
    else:
        model.assert_called_once()


def test_question_with_returned_file_remains_dialogue(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text("{}")
    handler = SimpleNamespace(state_path=tmp_path / "state.json", workflow_kwargs={}, _run=Mock())
    monkeypatch.setattr("phase_agent.graphs.dialogue.support.fresh_turn_context", lambda *args: {})
    model = Mock(return_value={"status": "answered"})
    monkeypatch.setattr("phase_agent.graphs.dialogue.decision.decide_from_saved_facts", model)
    ctx = SimpleNamespace(handler=handler, messages=[], conversation_id="A")
    decide(
        {
            "facts": {"tasks": [{"status": "pending", "result_path": str(path)}]},
            "message": "解释一下这个模型",
        },
        SimpleNamespace(context=ctx),
    )
    model.assert_called_once()
    handler._run.assert_not_called()

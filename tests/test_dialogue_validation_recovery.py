from unittest.mock import patch

from tests.test_unified_react_dialogue import make_handler


def test_invalid_dialogue_action_is_repaired_once_without_error(tmp_path):
    calls = []

    def request(_client, payload):
        calls.append(payload)
        if len(calls) == 1:
            error = ValueError(
                "Invalid LLM action: LangGraph proposal validation failed: "
                "action.tool: not allowed in this request"
            )
            error.validation_errors = ["action.tool: not allowed in this request"]
            raise error
        return {"kind": "answer", "answer": "已按当前允许范围重新判断。"}

    handler = make_handler(tmp_path, lambda payload: {"kind": "answer", "answer": "unused"})
    from phase_agent.graphs.dialogue.decision import decide_from_saved_facts

    with patch(
        "phase_agent.decisions.agent.decision_backend.request_decision_action",
        side_effect=request,
    ):
        result = decide_from_saved_facts(handler, {}, "继续", lambda: None)

    assert result["status"] == "answered"
    assert result["answer"] == "已按当前允许范围重新判断。"
    assert len(calls) == 2
    feedback = calls[1]["decision_context"]["proposal_validation_feedback"]
    assert feedback["errors"] == ["action.tool: not allowed in this request"]


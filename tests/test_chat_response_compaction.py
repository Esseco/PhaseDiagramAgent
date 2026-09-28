"""Large internal records must not be returned to the conversation."""

from run.open_webui_api import _chat_content, handle_chat_request


def test_large_candidate_json_is_not_echoed():
    detail = '{"composition": {"Fe": 12, "Mn": 12}, "rows": "' + "x" * 5000 + '"}'
    reply = _chat_content(detail)
    assert "composition" not in reply
    assert len(reply) < 200


def test_structured_result_is_summarized():
    reply = _chat_content({"status": "completed", "result": {"candidates": [1] * 10000}})
    assert "completed" in reply
    assert "candidates" not in reply


def test_normal_short_reply_stays_intact():
    response = handle_chat_request(
        {"messages": [{"role": "user", "content": "状态"}]},
        lambda messages, *, conversation_id: "已完成 3 个 branch。",
    )
    assert response["choices"][0]["message"]["content"] == "已完成 3 个 branch。"

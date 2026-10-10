import pytest
from phase_agent.runtime.agent_api import handle_chat_request, OpenWebUIRequestError
from phase_agent.runtime.ui_contracts import ChatRequest, ChatResponse


@pytest.mark.parametrize("change", [{"messages": "继续"}, {"messages": [1]},
    {"messages": [{"role": "user", "content": 12}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": 12}]}]},
    {"messages": [{"role": "user", "content": [{"type": "text"}]}]},
    {"messages": [{"role": "user", "content": [{"type": "input_text", "text": None}]}]},
    {"metadata": []}, {"metadata": {"chat_id": []}}, {"stream": "false"},
    {"user": {}}, {"messages": [{"role": "admin", "content": "secret"}]}])
def test_bad_ui_request_never_calls_handler(change):
    calls = []
    with pytest.raises(OpenWebUIRequestError) as caught:
        handle_chat_request({"messages": [{"role": "user", "content": "private-message"}], **change},
                            lambda *args, **kwargs: calls.append(args))
    assert calls == []
    assert "private-message" not in str(caught.value)


def test_multimodal_metadata_extensions_and_response_remain_compatible():
    messages = [{"role": "system", "content": "context"},
                {"role": "user", "content": [{"type": "input_text", "text": "继续"},
                    {"type": "image_url", "image_url": {"url": "test"}}]}]
    calls = []
    def handler(actual, *, conversation_id):
        calls.append((actual, conversation_id))
        return "等待批准"
    response = handle_chat_request({"messages": messages, "metadata": {"chat_id": "chat-1", "extra": 1},
        "stream": True, "temperature": .1}, handler)
    assert calls == [(messages, "chat-1")]
    assert response["choices"][0]["message"]["content"] == "等待批准"
    ChatResponse.model_validate(response)
    assert ChatRequest.model_json_schema()["properties"]["stream"]["type"] == "boolean"

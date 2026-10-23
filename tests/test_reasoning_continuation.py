import json
from unittest.mock import patch
import pytest
from phase_agent.decisions.agent.create_deepseek_client import (
    create_deepseek_client,
    DeepSeekResponseError,
)
from phase_agent.runtime.turn_process import _current
from tests.test_deepseek_client import FakeResponse, completion


@pytest.mark.parametrize("reply", ['"check_convergence"}', '{"tool":"check_convergence"}'])
def test_partial_json_continues_original_reasoning_without_restarting(reply):
    first = completion('{"tool":', finish_reason="length")
    first["choices"][0]["message"]["reasoning_content"] = "private reasoning already performed"
    record = {"id": "continuation-test", "events": [], "timings": []}
    token = _current.set(record)
    try:
        with patch(
            "urllib.request.urlopen",
            side_effect=[FakeResponse(first), FakeResponse(completion(reply))],
        ) as request:
            action = create_deepseek_client(api_key="fake")(
                {"decision_kind": "strategy", "state": {"human_constraints": ["keep constraint"]}}
            )
        bodies = [json.loads(c.args[0].data) for c in request.call_args_list]
        assert action["tool"] == "check_convergence"
        assert (
            request.call_args_list[1].args[0].full_url
            == "https://api.deepseek.com/beta/chat/completions"
        )
        assert bodies[1]["messages"][:-1] == bodies[0]["messages"]
        assert bodies[1]["messages"][-1] == {
            "role": "assistant",
            "prefix": True,
            "content": '{"tool":',
            "reasoning_content": "private reasoning already performed",
        }
        assert bodies[1]["thinking"] == {"type": "enabled"}
        assert bodies[0]["response_format"] == {"type": "json_object"}
        assert "response_format" not in bodies[1]
        assert "private reasoning" not in json.dumps(record)
        assert record["model_requests"][1]["recovery_kind"] == "prefix_continuation"
    finally:
        _current.reset(token)


def test_custom_endpoint_uses_compatible_restart_and_is_not_sent_to_official_service():
    with patch(
        "urllib.request.urlopen",
        side_effect=[
            FakeResponse(completion('{"tool":', finish_reason="length")),
            FakeResponse(completion('{"tool":"check_convergence"}')),
        ],
    ) as request:
        action = create_deepseek_client(api_key="fake", base_url="https://private.example/v1")(
            {"decision_kind": "strategy"}
        )
    assert action["tool"] == "check_convergence"
    assert all(
        c.args[0].full_url == "https://private.example/v1/chat/completions"
        for c in request.call_args_list
    )
    assert len(json.loads(request.call_args.args[0].data)["messages"]) == 2


def test_second_truncation_or_invalid_suffix_is_never_accepted_or_retried_again():
    for second in [
        completion('"check_convergence"}', finish_reason="length"),
        completion("not-json"),
    ]:
        with patch(
            "urllib.request.urlopen",
            side_effect=[
                FakeResponse(completion('{"tool":', finish_reason="length")),
                FakeResponse(second),
            ],
        ) as request:
            with pytest.raises(DeepSeekResponseError):
                create_deepseek_client(api_key="fake")({"decision_kind": "strategy"})
        assert request.call_count == 2

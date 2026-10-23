import json
from unittest.mock import patch

import pytest

from phase_agent.decisions.agent.create_deepseek_client import (
    DeepSeekResponseError,
    create_deepseek_client,
)
from tests.test_deepseek_client import FakeResponse, completion


@pytest.mark.parametrize("content", ["", '{"tool":', "{}"])
def test_reasoning_length_recovers_once_with_more_room(content):
    first = completion(content, finish_reason="length", completion_tokens=8192)
    first["choices"][0]["message"]["reasoning_content"] = "x" * 29283
    payload = {"decision_kind": "strategy"}
    with patch(
        "urllib.request.urlopen",
        side_effect=[
            FakeResponse(first),
            FakeResponse(
                completion(
                    '"check_convergence"}'
                    if content == '{"tool":'
                    else '{"tool":"check_convergence"}'
                )
            ),
        ],
    ) as request:
        result = create_deepseek_client(api_key="fake")(payload)
    bodies = [json.loads(call.args[0].data) for call in request.call_args_list]
    assert len(bodies) == 2
    assert [b["max_tokens"] for b in bodies] == [8192, 16384]
    assert all(b["thinking"] == {"type": "enabled"} for b in bodies)
    assert result["_llm_usage"]["calls"] == 2
    assert result["_llm_usage"]["output_tokens"] == 8196
    assert payload == {"decision_kind": "strategy"}


def test_repeated_reasoning_length_stops_and_reports_budget():
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(completion("", finish_reason="length", completion_tokens=8192)),
    ) as request:
        with pytest.raises(DeepSeekResponseError) as caught:
            create_deepseek_client(api_key="fake")({"decision_kind": "strategy"})
    assert request.call_count == 2
    assert caught.value.code == "output_truncated"
    assert caught.value.llm_usage["calls"] == 2
    assert caught.value.llm_usage["output_tokens"] == 16384
    assert "16384" in str(caught.value)


@pytest.mark.parametrize("content", ["", "bad"])
def test_non_length_reasoning_failure_is_not_retried(content):
    with patch("urllib.request.urlopen", return_value=FakeResponse(completion(content))) as request:
        with pytest.raises(DeepSeekResponseError):
            create_deepseek_client(api_key="fake")({"decision_kind": "strategy"})
    assert request.call_count == 1


def test_reasoning_recovery_keeps_original_deadline():
    from phase_agent.runtime.api_deadline import APIDeadlineExceeded

    first = completion("", finish_reason="length")
    with patch(
        "phase_agent.runtime.api_deadline.read_response",
        side_effect=[
            json.dumps(first).encode(),
            APIDeadlineExceeded(),
        ],
    ) as transport:
        with pytest.raises(DeepSeekResponseError) as caught:
            create_deepseek_client(api_key="fake")({"decision_kind": "strategy"})
    assert transport.call_count == 2
    assert transport.call_args_list[0].args[2] == transport.call_args_list[1].args[2]
    assert caught.value.code == "total_timeout"
    assert caught.value.llm_usage["calls"] == 2


def test_reasoning_recovery_honors_cancellation():
    from phase_agent.graphs.cancellation import ScientificRunCancelled

    with (
        patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(completion("", finish_reason="length")),
        ) as request,
        patch(
            "phase_agent.graphs.cancellation.check_cancelled",
            side_effect=[
                None,
                None,
                None,
                ScientificRunCancelled("cancelled"),
            ],
        ),
    ):
        with pytest.raises(ScientificRunCancelled):
            create_deepseek_client(api_key="fake")({"decision_kind": "strategy"})
    assert request.call_count == 1


@pytest.mark.parametrize(
    "budget, expected", [(20000, [20000, 32768]), (32768, [32768]), (40000, [40000])]
)
def test_recovery_is_capped_without_reducing_configured_budget(budget, expected):
    with patch(
        "urllib.request.urlopen", return_value=FakeResponse(completion("", finish_reason="length"))
    ) as request:
        with pytest.raises(DeepSeekResponseError):
            create_deepseek_client(api_key="fake", reasoning_max_tokens=budget)(
                {"decision_kind": "strategy"}
            )
    assert [json.loads(c.args[0].data)["max_tokens"] for c in request.call_args_list] == expected

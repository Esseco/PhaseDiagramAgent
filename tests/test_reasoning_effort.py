import json
from unittest.mock import patch

import pytest

from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
from phase_agent.runtime.runtime_client_settings import search_client_settings
from tests.test_deepseek_client import FakeResponse, completion


def test_model_can_choose_low_effort_without_disabling_scientific_thinking():
    payload = {
        "decision_kind": "strategy",
        "analysis_plan": {
            "stage": "registered",
            "question": "compare",
            "reasoning": "scientific",
            "analysis_kind": "scientific_comparison",
            "reasoning_effort": "low",
        },
    }
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(completion('{"tool":"check_convergence"}')),
    ) as request:
        create_deepseek_client(api_key="fake")(payload)
    body = json.loads(request.call_args.args[0].data)
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "first_effort,second_effort", [("max", "high"), ("high", "low"), ("low", "low")]
)
def test_empty_reasoning_truncation_gets_one_effort_step_down(first_effort, second_effort):
    first = completion("", finish_reason="length", completion_tokens=8192)
    first["choices"][0]["message"]["reasoning_content"] = "thought"
    with patch(
        "urllib.request.urlopen",
        side_effect=[FakeResponse(first), FakeResponse(completion('{"tool":"check_convergence"}'))],
    ) as request:
        result = create_deepseek_client(api_key="fake", reasoning_effort=first_effort)(
            {"decision_kind": "strategy"}
        )
    bodies = [json.loads(c.args[0].data) for c in request.call_args_list]
    assert [b["reasoning_effort"] for b in bodies] == [first_effort, second_effort]
    assert all(b["thinking"] == {"type": "enabled"} for b in bodies)
    assert bodies[0]["messages"] == bodies[1]["messages"]
    assert result["_llm_usage"]["reasoning_effort"] == second_effort


def test_partial_answer_truncation_keeps_effort_while_expanding_room():
    with patch(
        "urllib.request.urlopen",
        side_effect=[
            FakeResponse(completion('{"tool":', finish_reason="length")),
            FakeResponse(completion('"check_convergence"}')),
        ],
    ) as request:
        create_deepseek_client(api_key="fake", reasoning_effort="high")(
            {"decision_kind": "strategy"}
        )
    assert [json.loads(c.args[0].data)["reasoning_effort"] for c in request.call_args_list] == [
        "high",
        "high",
    ]


def test_effort_setting_is_validated_and_configuration_does_not_turn_thinking_on():
    assert search_client_settings({"reasoning_effort": "max"})["reasoning_effort"] == "max"
    with pytest.raises(ValueError):
        create_deepseek_client(api_key="fake", reasoning_effort="invalid")
    with patch("urllib.request.urlopen", return_value=FakeResponse(completion("{}"))) as request:
        create_deepseek_client(api_key="fake", reasoning_effort="max")(
            {"mode": "configuration_dialogue"}
        )
    body = json.loads(request.call_args.args[0].data)
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body


@pytest.mark.parametrize("effort", [None, "high", "max"])
def test_explicit_analysis_tier_defaults_to_low_but_agent_can_escalate(effort):
    scope = {
        "stage": "registered",
        "question": "compare existing evidence",
        "reasoning": "scientific",
        "analysis_kind": "scientific_comparison",
    }
    if effort is not None:
        scope.update(reasoning_effort=effort, reasoning_basis="resolve competing model assumptions")
    with patch("urllib.request.urlopen", return_value=FakeResponse(completion("{}"))) as request:
        create_deepseek_client(api_key="fake", reasoning_effort="high")(
            {"decision_kind": "strategy", "analysis_plan": scope}
        )
    body = json.loads(request.call_args.args[0].data)
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == (effort or "low")

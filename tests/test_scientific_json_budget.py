import json
from unittest.mock import patch
from tests.test_deepseek_client import FakeResponse, completion
from phase_agent.decisions.agent.create_deepseek_client import (
    create_deepseek_client,
    DeepSeekResponseError,
)
import pytest


def run(payload, **kwargs):
    client = create_deepseek_client(api_key="fake-no-network", thinking="disabled", **kwargs)
    with patch(
        "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
        return_value=FakeResponse(completion('{"tool":"generate_branches"}')),
    ) as request:
        result = client(payload)
    limits = [json.loads(call.args[0].data)["max_tokens"] for call in request.call_args_list]
    return result, limits


def test_configured_limit_not_silently_capped_at_1600():
    _, limits = run({}, max_tokens=2400)
    assert limits[0] >= 2400
    assert len(limits) == 1


def test_scientific_contract_has_room_on_first_request():
    _, limits = run(
        {"output_contracts": {"scientific_decision": {"type": "object"}}}, max_tokens=2400
    )
    assert limits[0] >= 4096
    assert len(limits) == 1


def test_dft_decisions_and_purpose_pairs_share_output_budget():
    _, limits = run(
        {
            "allowed_tools": ["select_dft_candidates"],
            "state": {"qbc_candidates": [{}] * 100},
            "output_contracts": {"scientific_decision": {"type": "object"}},
        }
    )
    assert limits[0] >= 16000
    assert len(limits) == 1


def test_small_dialogue_keeps_small_initial_output():
    _, limits = run({"semantic_dialogue": True}, max_tokens=800)
    assert limits[0] == 800


@pytest.mark.parametrize("content", ["{}", '{"tool":'])
def test_nonreasoning_truncation_never_returns_partial_action_or_retries(content):
    client = create_deepseek_client(api_key="fake-no-network", thinking="disabled")
    with patch(
        "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
        return_value=FakeResponse(completion(content, finish_reason="length")),
    ) as request:
        with pytest.raises(DeepSeekResponseError) as caught:
            client({"output_contracts": {"scientific_decision": {}}})
    assert request.call_count == 1
    assert caught.value.code == "invalid_json"
    assert caught.value.llm_usage["calls"] == 1


def test_scientific_request_does_not_ask_for_duplicate_explanation():
    from phase_agent.decisions.agent.scientific_decisions import attach_decision_contract

    payload = {
        "instruction": "choose",
        "decision_context": {
            "scientific_decision_brief": {"policy": {"schema_version": 1}},
            "decision_review_format": {"instruction": "repeat observations and reasons"},
        },
    }
    out = attach_decision_contract(payload)
    assert "decision_review_format" not in out["decision_context"]
    assert "decision_review_format" in payload["decision_context"]
    assert "Be concise" in out["instruction"]
    assert "scientific_decision" in out["output_contracts"]

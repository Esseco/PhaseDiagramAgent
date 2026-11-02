"""Expert JSON uses the raw provider transport and retains reviewed evidence."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from phase_agent.models.capabilities import copy_model_capabilities
from phase_agent.models.create_deepseek_client import create_deepseek_client
from phase_agent.models.deepagents_proposal import create_proposal_client
from phase_agent.models.transport_tables import expand_evidence_tables
from phase_agent.policies.relax_mc_preference import RELAX_MC_PREFERENCE
from phase_agent.runtime.token_usage import usage_summary
from phase_agent.runtime.turn_process import _current
from tests.test_deepagents_proposal import OfflineModel
from tests.test_deepseek_client import FakeResponse, completion


def expert_request(mode="expert_assessment"):
    return {
        "mode": mode,
        "agent_role": "exploration" if mode == "expert_assessment" else "coordinator",
        "instruction": "Review only the supplied snapshot; return the role contract.",
        "output_contracts": {"expert_advice": {"type": "object"}},
        "decision_context": {"evidence_snapshot_id": "snapshot-1"},
    }


@pytest.mark.parametrize("mode", ["expert_assessment", "team_consultation"])
def test_raw_expert_json_has_role_prompt_headroom_and_one_cached_usage(mode):
    client = create_deepseek_client(
        api_key="fake", thinking="disabled", system_prompt="Generic action prompt"
    )
    assert client.supports_expert_advice is True
    assert client.expert_client is client
    record = {"id": "expert-transport-test", "events": []}
    token = _current.set(record)
    try:
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(completion('{"assessment":"reviewed"}')),
        ) as request:
            result = client.expert_client(expert_request(mode))
            reused = client.expert_client(expert_request(mode))
        body = json.loads(request.call_args.args[0].data)
        prompt = body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        assert "read-only" in prompt and "output_contracts" in prompt
        assert RELAX_MC_PREFERENCE not in prompt
        assert "Generic action prompt" not in prompt
        assert body["max_tokens"] >= 4096
        assert body["response_format"] == {"type": "json_object"}
        assert payload["agent_role"] == expert_request(mode)["agent_role"]
        assert payload["instruction"] == expert_request(mode)["instruction"]
        assert payload["output_contracts"] == expert_request(mode)["output_contracts"]
        assert result["assessment"] == reused["assessment"] == "reviewed"
        assert result["_llm_usage"]["calls"] == 1
        assert "_llm_usage" not in reused
        assert request.call_count == 1
        summary = usage_summary(record)
        assert summary["api_calls"] == 1 and summary["reused_requests"] == 1
        assert summary["input_tokens"] == 10 and summary["output_tokens"] == 4
    finally:
        _current.reset(token)


@pytest.mark.parametrize("harness", ["deepagents", "hybrid"])
def test_actual_harness_experts_reuse_raw_provider_configuration(harness):
    action_harness = SimpleNamespace(invoke=Mock(side_effect=AssertionError("no expert actions")))
    with patch("deepagents.create_deep_agent", return_value=action_harness):
        client = create_deepseek_client(
            api_key="fake",
            proposal_harness=harness,
            model="configured-model",
            base_url="https://configured.invalid/v1",
            timeout=37,
            total_timeout=41,
            max_tokens=1200,
            routine_max_tokens=2600,
            reasoning_max_tokens=9000,
            thinking="disabled",
        )
    assert client.supports_expert_advice is True
    assert client.expert_client is not client
    assert client.expert_client.expert_client is client.expert_client
    with patch(
        "urllib.request.urlopen",
        return_value=FakeResponse(completion('{"assessment":"reviewed"}')),
    ) as request:
        result = client.expert_client(expert_request())
    body = json.loads(request.call_args.args[0].data)
    assert request.call_args.args[0].full_url == "https://configured.invalid/v1/chat/completions"
    assert request.call_args.kwargs["timeout"] == 37
    assert body["model"] == "configured-model"
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] >= 4096
    assert result["assessment"] == "reviewed"
    action_harness.invoke.assert_not_called()


def test_deep_coordinator_receives_complete_reviewed_advice_and_role():
    captured = []

    class InspectingModel(OfflineModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.extend(
                json.loads(message.content)
                for message in messages
                if isinstance(message, HumanMessage)
                and isinstance(message.content, str)
                and message.content.startswith("{")
            )
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    payload = {
        "mode": "autonomous_search",
        "agent_role": "strategic",
        "instruction": "Compare both reviewed opinions before proposing one action.",
        "decision_context": {
            "evidence_snapshot_id": "snapshot-1",
            "expert_advice": [
                {
                    "role": "exploration",
                    "snapshot_id": "snapshot-1",
                    "opportunities": [{"hypothesis": "near hull", "evidence_ids": ["ev-1"]}],
                    "limits": {"unmeasured_gain": True},
                },
                {
                    "role": "reliability",
                    "snapshot_id": "snapshot-1",
                    "concerns": [{"validation_gap": "coverage", "evidence_ids": ["ev-2"]}],
                    "action_authority": False,
                },
            ],
        },
    }
    original = deepcopy(payload)
    client = create_proposal_client(
        InspectingModel(responses=[AIMessage(content='{"tool":"check_convergence"}')])
    )
    client(payload)
    assert len(captured) == 1
    material = expand_evidence_tables(captured[0])
    assert material["agent_role"] == payload["agent_role"]
    assert material["instruction"] == payload["instruction"]
    assert (
        material["decision_context"]["expert_advice"]
        == payload["decision_context"]["expert_advice"]
    )
    assert material["decision_context"]["evidence_snapshot_id"] == "snapshot-1"
    assert payload == original


def test_model_adapter_preserves_semantic_dialogue_capability():
    source = SimpleNamespace(_semantic_dialogue=False, supports_expert_advice=True)
    target = SimpleNamespace()
    assert copy_model_capabilities(source, target) is target
    assert target._semantic_dialogue is False
    assert target.supports_expert_advice is True

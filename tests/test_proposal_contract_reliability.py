"""Offline contract repair retains human context and cannot execute tools."""

from copy import deepcopy
import importlib
import json
from unittest.mock import Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from phase_agent.decisions.agent.deepagents_proposal import (
    create_proposal_client,
    proposal_material,
)
from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
from phase_agent.decisions.agent.revise_tool_proposal import revise_tool_proposal
from phase_agent.decisions.agent.transport_tables import expand_evidence_tables
from phase_agent.graphs.proposal_graph import request_with_langgraph


@pytest.fixture(autouse=True)
def no_dispatch(monkeypatch):
    module = importlib.import_module("phase_agent.tools.dispatch.execute_tool_action")
    handler = Mock(side_effect=AssertionError("proposal contract correction must never dispatch"))
    monkeypatch.setattr(module, "execute_tool_action", handler)
    yield
    handler.assert_not_called()


def action(**changes):
    return {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {
                    "candidate_id": "real-candidate-01",
                    "action": "DFT_SINGLE_POINT",
                    "reason": "near hull",
                }
            ]
        },
        "budget": 30.0,
        "reason": "independent evidence",
        **changes,
    }


def payload(**changes):
    return {
        "instruction": "Choose one action",
        "allowed_tools": ["select_dft_candidates"],
        "decision_context": {},
        **changes,
    }


def usage(cost):
    return {"calls": 1, "input_tokens": 11, "output_tokens": 7, "cost": cost}


def revision_payload():
    original = action(
        target_ids=["real-candidate-01", "real-candidate-02"],
        current_state_analysis="Scientific review preserved",
        expected_purpose="assess the hull",
    )
    original["parameters"].update(
        scientific_extra={"T": [300, 900], "custom_parameter": {"allowed_name": 4}},
        dft_input_preview={"large": "omitted"},
        generation_cost_preview={"large": "omitted"},
        module_execution_scope={"large": "omitted"},
        relax_mc_preview={"large": "omitted"},
    )
    original["_llm_usage"] = usage(0.3)
    original["decision_context"] = {"raw_state": "omitted"}
    return payload(
        mode="revise_action_from_human_feedback",
        human_comment="Keep the independent audit; revise from the current evidence.",
        current_proposal={"raw_action": original, "wrapper_preview": "omitted"},
    )


def assert_revision_material(material, original):
    material = expand_evidence_tables(material)
    assert material["human_comment"] == original["human_comment"]
    retained = material["current_proposal"]
    expected = deepcopy(original["current_proposal"]["raw_action"])
    expected.pop("_llm_usage")
    expected.pop("decision_context")
    for key in (
        "dft_input_preview",
        "generation_cost_preview",
        "module_execution_scope",
        "relax_mc_preview",
    ):
        expected["parameters"].pop(key)
    assert retained == expected


def test_deep_material_preserves_complete_revision_after_existing_compaction():
    request = revision_payload()
    before = deepcopy(request)
    assert_revision_material(proposal_material(prepare_llm_request(request)), before)
    assert_revision_material(proposal_material(request), before)
    assert request == before


def test_real_deep_harness_offline_model_receives_revision():
    captured = []

    class OfflineModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.extend(
                json.loads(m.content)
                for m in messages
                if isinstance(m, HumanMessage)
                and isinstance(m.content, str)
                and m.content.startswith("{")
            )
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    client = create_proposal_client(
        OfflineModel(responses=[AIMessage(content=json.dumps(action()))])
    )
    request = revision_payload()
    result = client(request)
    assert len(captured) == 1
    assert_revision_material(captured[0], request)
    assert result["_llm_usage"]["calls"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        action(budget="30"),
        action(parameters=[]),
        action(parameters=[{"invalid": True}]),
        action(target_ids="real-candidate-01"),
        action(
            parameters={
                "decisions": [{"candidate_id": "real-candidate-01", "action": "SINGLE_POINT"}]
            }
        ),
        {"tool": "allocate_mc_bohb", "budget": 0, "parameters": {"mc_budget": "2"}},
    ],
)
def test_common_schema_failure_gets_one_model_correction_and_same_validation(bad):
    corrected = (
        action()
        if bad["tool"] != "allocate_mc_bohb"
        else {"tool": "allocate_mc_bohb", "budget": 0, "parameters": {"mc_budget": 2}}
    )
    first = {**deepcopy(bad), "_llm_usage": usage(0.1)}
    second = {**deepcopy(corrected), "_llm_usage": usage(0.2)}
    client = Mock(side_effect=[first, second])
    request = payload(allowed_tools=[bad["tool"]], human_comment="Correct the interface only")
    before = deepcopy(request)
    result = request_with_langgraph(client, request)
    assert client.call_count == 2
    correction = client.call_args_list[1].args[0]
    assert correction["validation_errors"]
    assert correction["current_proposal"] == first
    assert correction["human_comment"] == request["human_comment"]
    assert correction["allowed_tools"] == request["allowed_tools"]
    assert correction["output_contracts"] == client.call_args_list[0].args[0]["output_contracts"]
    assert {k: v for k, v in result.items() if k != "_llm_usage"} == corrected
    assert result["_llm_usage"] == {
        "calls": 2,
        "input_tokens": 22,
        "output_tokens": 14,
        "cost": pytest.approx(0.3),
    }
    assert request == before


def test_repeated_schema_failure_stops_at_two_with_usage_and_original_pending():
    bad = action(budget="30", _llm_usage=usage(None))
    client = Mock(side_effect=[deepcopy(bad), deepcopy(bad)])
    client.decision_backend = "langgraph"
    original = {"raw_action": action(), "reason": "pending original"}
    before = deepcopy(original)
    result = revise_tool_proposal(
        original,
        "Revise with the same evidence",
        state={},
        allowed_tools=["select_dft_candidates"],
        agent_client=client,
    )
    assert client.call_count == 2
    assert result["revision_status"] == "revision_failed"
    assert result["action"] == before["raw_action"]
    assert "Invalid LLM action" in result["analysis"]
    assert result["llm_usage"] == {
        "calls": 2,
        "input_tokens": 22,
        "output_tokens": 14,
        "cost": None,
    }
    assert original == before


def test_valid_schema_does_not_make_an_extra_model_call():
    client = Mock(return_value=action(_llm_usage=usage(0.1)))
    assert request_with_langgraph(client, payload())["parameters"] == action()["parameters"]
    client.assert_called_once()


@pytest.mark.parametrize("first_invalid", [False, True])
def test_network_error_is_not_retried_or_wrapped(first_invalid):
    failure = ConnectionError("offline synthetic transport failure")
    failure.llm_usage = usage(0.2)
    responses = (
        [action(budget="30", _llm_usage=usage(0.1)), failure] if first_invalid else [failure]
    )
    client = Mock(side_effect=responses)
    with pytest.raises(ConnectionError) as caught:
        request_with_langgraph(client, payload())
    assert caught.value is failure
    assert client.call_count == len(responses)
    assert failure.llm_usage["calls"] == len(responses)
    assert failure.llm_usage["cost"] == pytest.approx(0.3 if first_invalid else 0.2)


def test_business_prerequisite_failure_is_not_a_schema_retry():
    client = Mock(return_value={"tool": "allocate_mc_bohb", "parameters": {"mc_budget": 2}})
    with pytest.raises(ValueError, match="prerequisite"):
        request_with_langgraph(
            client,
            payload(
                allowed_tools=["allocate_mc_bohb"],
                decision_context={"mc_entry_conditions": {"required": True, "ready": False}},
            ),
        )
    client.assert_called_once()


def test_correction_is_still_subject_to_allowed_tool_policy():
    client = Mock(side_effect=[action(budget="30"), {"tool": "pause_search"}])
    with pytest.raises(ValueError, match="not allowed"):
        request_with_langgraph(client, payload())
    assert client.call_count == 2


def test_non_object_response_is_presented_unchanged_for_one_correction():
    client = Mock(side_effect=[["malformed action"], action()])
    result = request_with_langgraph(client, payload())
    assert client.call_count == 2
    assert client.call_args.args[0]["current_proposal"] == ["malformed action"]
    assert result == action()


def test_real_deep_harness_transmits_contract_feedback_without_parameter_pruning():
    captured = []

    class OfflineModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.extend(
                json.loads(m.content)
                for m in messages
                if isinstance(m, HumanMessage)
                and isinstance(m.content, str)
                and m.content.startswith("{")
            )
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    invalid = action(parameters=[{"invalid": True}])
    client = create_proposal_client(
        OfflineModel(
            responses=[
                AIMessage(content=json.dumps(invalid)),
                AIMessage(content=json.dumps(action())),
            ]
        )
    )
    result = request_with_langgraph(client, payload(human_comment="Use my current evidence"))
    assert len(captured) == 2
    correction = expand_evidence_tables(captured[1])
    assert correction["validation_errors"]
    assert correction["human_comment"] == "Use my current evidence"
    assert correction["current_proposal"]["parameters"] == invalid["parameters"]
    assert "_llm_usage" not in correction["current_proposal"]
    assert result["parameters"] == action()["parameters"]
    assert result["_llm_usage"]["calls"] == 2


def test_saved_review_validation_never_reasks_the_model():
    from phase_agent.decisions.agent.post_dft_review import request_review_repair

    client = Mock(side_effect=AssertionError("saved review cannot call a model"))
    with pytest.raises(ValueError, match="post_dft_review"):
        request_review_repair(client, {}, {"tool": "update_mlip"})
    client.assert_not_called()


def test_review_choice_consistency_is_not_a_schema_retry():
    from tests.test_post_dft_review import review
    from phase_agent.decisions.agent.post_dft_review import request_validated_action

    client = Mock(return_value={"tool": "update_mlip", "post_dft_review": review()})
    with pytest.raises(ValueError, match="finetune"):
        request_validated_action(
            client,
            payload(
                allowed_tools=["update_mlip"],
                decision_context={"post_dft_assessment": {"status": "evaluated"}},
            ),
        )
    client.assert_called_once()


def test_existing_review_wire_alias_is_validated_on_a_copy_without_extra_call():
    from tests.test_post_dft_review import review
    from phase_agent.decisions.agent.post_dft_review import request_validated_action

    original = {
        "tool": "update_mlip",
        "parameters": {"prepare_inputs_only": True},
        "post_dft_review": {**review(), "choice": "update_mlip", "finetune_recommendation": "now"},
    }
    before = deepcopy(original)
    client = Mock(return_value=original)
    result = request_validated_action(
        client,
        payload(
            allowed_tools=["update_mlip"],
            decision_context={"post_dft_assessment": {"status": "evaluated"}},
        ),
    )
    assert result["post_dft_review"]["choice"] == "finetune"
    assert original == before
    client.assert_called_once()


@pytest.mark.parametrize("harness", [None, "deepagents"])
def test_contract_correction_keeps_the_existing_hybrid_harness(harness):
    from phase_agent.decisions.agent.hybrid_proposal import create_hybrid_client

    legacy = Mock(side_effect=[action(budget="30"), action()])
    deep = Mock(side_effect=[action(budget="30"), action()])
    factory = Mock(return_value=deep)
    client = create_hybrid_client(legacy, factory)
    request = payload(mode="autonomous_search")
    if harness:
        request["analysis_harness"] = harness
    result = request_with_langgraph(client, request)
    used, unused = (deep, legacy) if harness else (legacy, deep)
    assert used.call_count == 2
    assert used.call_args.args[0]["mode"] == "proposal_contract_repair"
    unused.assert_not_called()
    assert result["_proposal_harness"] == (harness or "legacy")
    assert factory.call_count == (1 if harness else 0)


def test_valid_generation_plan_derives_only_missing_totals_without_extra_model_call():
    original = {
        "tool": "generate_branches",
        "parameters": {
            "generation_plan": [
                {"strategy": "coverage", "quota": 2, "phase": "O3", "reason": "coverage"}
            ]
        },
    }
    before = deepcopy(original)
    client = Mock(return_value=original)
    result = request_with_langgraph(client, payload(allowed_tools=["generate_branches"]))
    client.assert_called_once()
    assert result["parameters"]["quotas"] == {"coverage": 2}
    assert result["parameters"]["total_quota"] == 2
    assert original == before


@pytest.mark.parametrize("by_policy", [False, True])
def test_generation_count_schema_correction_never_handles_region_mode(by_policy):
    from phase_agent.decisions.agent.proposal_schema_repair import repair_proposal_schema
    from phase_agent.tools.policy.region_sampling import MODE, VERSION

    original = {
        "tool": "generate_branches",
        "parameters": {
            "generation_plan": [
                {"strategy": "coverage", "quota": 2, "phase": "O3", "reason": "coverage"}
            ],
            "quotas": {"coverage": 3},
            "total_quota": 2,
        },
    }
    request = payload(allowed_tools=["generate_branches"])
    if by_policy:
        request["decision_context"]["branch_count_policy"] = {"version": VERSION}
    else:
        original["parameters"]["sampling_mode"] = MODE
    before = deepcopy(original)
    client = Mock(side_effect=AssertionError("region checks must remain with their existing path"))
    assert repair_proposal_schema(client, request, original) == before
    client.assert_not_called()
    assert original == before

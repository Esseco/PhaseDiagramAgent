from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from phase_agent.decisions.agent.transport_references import (
    compact_transport_references,
    expand_transport_references,
)
from phase_agent.decisions.agent.transport_tables import (
    compact_evidence_tables,
    expand_evidence_tables,
)
from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
from phase_agent.decisions.agent.hybrid_proposal import create_hybrid_client, choose_harness
from phase_agent.runtime.token_usage import cached_request
from phase_agent.runtime.turn_process import _current


def evidence():
    return {
        "configuration": {
            "nested": {"threshold": 0.05, "human_constraints": ["never submit"] * 30}
        },
        "qbc_candidates": [
            {
                "candidate_id": str(i),
                "ehull": i / 100,
                "structure_path": "E:/project/results/epoch0/path/" + str(i) + ".vasp",
            }
            for i in range(175)
        ],
    }


def test_aliases_restore_every_value_after_tables_across_interaction_shapes():
    data = evidence()
    for mode in [
        "configuration_dialogue",
        "configuration_patch_repair",
        "autonomous_search",
        "dialogue_contract_repair",
        "scientific_decision_repair",
    ]:
        original = {
            "mode": mode,
            "state": {"facts": data},
            "decision_context": {"facts": deepcopy(data)},
            "conversation_facts": {"facts": deepcopy(data)},
            "instruction": "keep this verbatim",
            "output_contracts": {"facts": deepcopy(data)},
        }
        before = deepcopy(original)
        encoded = compact_evidence_tables(compact_transport_references(original))
        decoded = expand_evidence_tables(encoded)
        assert decoded == original and original == before
        assert (
            encoded["instruction"] == original["instruction"]
            and expand_evidence_tables(encoded["output_contracts"]) == original["output_contracts"]
        )
        assert len(json.dumps(encoded)) < len(json.dumps(original)) * 0.75
        assert decoded["state"]["facts"]["qbc_candidates"][-1]["ehull"] == 1.74


def test_pointer_escaping_and_changed_metric_are_not_aliased():
    value = {"complex/~key": [{"id": str(i), "energy": i} for i in range(30)]}
    payload = {
        "state": {"a/~": value},
        "decision_context": {"copy": deepcopy(value), "changed": deepcopy(value)},
    }
    payload["decision_context"]["changed"]["complex/~key"][-1]["energy"] = 999
    encoded = compact_transport_references(payload)
    assert expand_transport_references(encoded) == payload
    assert encoded["decision_context"]["copy"]["format"] == "evidence_ref_v1"
    assert encoded["decision_context"]["changed"] != encoded["decision_context"]["copy"]


def test_invalid_or_cyclic_transport_reference_is_an_error():
    with pytest.raises(ValueError):
        expand_transport_references(
            {"state": {"format": "evidence_ref_v1", "source_ref": "/state"}}
        )
    with pytest.raises(ValueError):
        expand_transport_references(
            {"state": {"format": "evidence_ref_v1", "source_ref": "/missing"}}
        )


def test_cache_is_scoped_to_client_turn_and_exact_current_facts():
    called = []

    def provider(payload):
        called.append(payload)
        return {"kind": "answer", "answer": "ok", "_llm_usage": {"calls": 1}}

    record = {}
    token = _current.set(record)
    try:
        cached_request({"v": 1}, provider, namespace="a")
        assert "_llm_usage" not in cached_request({"v": 1}, provider, namespace="a")
        cached_request({"v": 1}, provider, namespace="b")
        cached_request({"v": 2}, provider, namespace="a")
        assert len(called) == 3
    finally:
        _current.reset(token)
    token = _current.set({})
    try:
        cached_request({"v": 1}, provider, namespace="a")
        assert len(called) == 4
    finally:
        _current.reset(token)


def test_hybrid_topic_planning_does_not_build_complex_agent_then_preserves_analysis():
    base = Mock(return_value={"kind": "inspect_evidence", "topics": ["budget", "model"]})
    deep = Mock(return_value={"tool": "check_convergence"})
    factory = Mock(return_value=deep)
    client = create_hybrid_client(base, factory)
    request = {
        "mode": "autonomous_search",
        "analysis_harness": "deepagents",
        "evidence_reader": {"phase": "select_topics"},
    }
    client(request)
    factory.assert_not_called()
    base.assert_called_once()
    request["evidence_reader"]["phase"] = "analyze"
    client(request)
    factory.assert_called_once()
    deep.assert_called_once()


def test_two_legacy_clients_never_share_cached_responses_for_different_models():
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()

    a = create_deepseek_client(api_key="fake", model="first")
    b = create_deepseek_client(api_key="fake", model="second")
    token = _current.set({"id": "fake", "events": [], "timings": []})
    try:
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            return_value=Response(),
        ) as provider:
            a({"mode": "configuration_dialogue"})
            a({"mode": "configuration_dialogue"})
            b({"mode": "configuration_dialogue"})
            assert provider.call_count == 2
    finally:
        _current.reset(token)


def test_actual_deep_client_reuses_only_identical_successful_call():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from phase_agent.decisions.agent.deepagents_proposal import create_proposal_client

    class OfflineModel(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs):
            return self

    client = create_proposal_client(
        OfflineModel(responses=[AIMessage(content='{"tool":"check_convergence"}')])
    )
    token = _current.set({"id": "fake-deep", "events": [], "timings": []})
    try:
        request = {
            "decision_context": {
                "qbc_candidates": [{"candidate_id": "a", "ehull": 0.01}],
                "human_constraints": ["never submit"],
            },
            "allowed_tools": ["check_convergence"],
        }
        assert "_llm_usage" in client(request)
        assert "_llm_usage" not in client(request)
        changed = deepcopy(request)
        changed["decision_context"]["qbc_candidates"][0]["ehull"] = 0.02
        assert "_llm_usage" in client(changed)
    finally:
        _current.reset(token)


def test_human_constraints_and_instructions_never_turn_into_aliases():
    facts = {"human_constraints": ["keep this explicit"] * 100, "instruction": "User authority"}
    value = {"state": {"facts": facts}, "decision_context": {"facts": deepcopy(facts)}}
    encoded = compact_transport_references(value)
    assert encoded == value


def test_alias_candidate_count_preserves_output_headroom():
    from phase_agent.decisions.agent.transport_tables import evidence_row_count

    rows = [{"candidate_id": str(i), "ehull": i / 100} for i in range(175)]
    encoded = compact_transport_references(
        {"state": {"first": rows}, "decision_context": {"qbc_candidates": deepcopy(rows)}}
    )
    assert evidence_row_count(encoded["decision_context"]["qbc_candidates"]) == 175
    assert (
        expand_evidence_tables(compact_evidence_tables(encoded))["decision_context"][
            "qbc_candidates"
        ]
        == rows
    )


def test_deep_approved_knowledge_is_sent_once_and_memory_keeps_the_source_pointer():
    from langchain_core.messages import AIMessage
    from phase_agent.decisions.agent.deepagents_proposal import create_proposal_client

    agent = Mock()
    agent.invoke.return_value = {"messages": [AIMessage(content='{"tool":"check_convergence"}')]}
    with patch("deepagents.create_deep_agent", return_value=agent):
        client = create_proposal_client(Mock())
        client(
            {
                "decision_context": {
                    "relevant_approved_knowledge": [
                        {"knowledge_id": "k", "statement": "unique-approved-scientific-fact"}
                    ]
                }
            }
        )
    invocation = agent.invoke.call_args.args[0]
    assert "unique-approved-scientific-fact" in invocation["messages"][0]["content"]
    assert "unique-approved-scientific-fact" not in json.dumps(invocation["files"])
    assert "relevant_approved_knowledge" in json.dumps(invocation["files"])

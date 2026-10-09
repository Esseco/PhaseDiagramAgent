import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from decision_layer.agent.deepagents_proposal import create_proposal_client, proposal_material


class OfflineModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_real_harness_offline_json_and_memory():
    client = create_proposal_client(OfflineModel(responses=[AIMessage(content='{"tool":"check_convergence"}')]))
    result = client({"allowed_tools": ["check_convergence"], "decision_context": {
        "relevant_approved_knowledge": [{"knowledge_id": "k", "statement": "Check evidence"}]}})
    assert result["tool"] == "check_convergence"
    assert result["_proposal_harness"] == "deepagents"
    assert client.decision_backend == "langgraph"


def test_transport_allowlist_drops_raw_state_and_paths():
    result = proposal_material({"state": {"secret": "x"}, "decision_context": {
        "current_phase_diagram": {"mlip": {"stable_count": 3, "structure": [1], "path": "private"}},
        "unknown": "private"}})
    assert result["decision_context"] == {"current_phase_diagram": {"mlip": {"stable_count": 3}}}
    assert "state" not in result


def test_delegation_is_denied():
    client = create_proposal_client(OfflineModel(responses=[AIMessage(content="", tool_calls=[
        {"name": "task", "args": {"description": "work", "subagent_type": "general-purpose"}, "id": "a"}])]))
    with pytest.raises(ValueError, match="cannot delegate"):
        client({})


def test_contract_and_repair_material_preserved_without_raw_action():
    payload = {"output_contracts": {"action": {"type": "object"}},
               "validation_errors": ["action.tool: invalid"],
               "invalid_action": {"structure": [1], "path": "private"},
               "decision_context": {"qbc_candidates": [{"candidate_id": "a", "ehull_per_atom": .01,
                                                            "forces": [1], "path": "private"}]}}
    material = proposal_material(payload)
    assert material["output_contracts"] == payload["output_contracts"]
    assert material["validation_errors"] == payload["validation_errors"]
    assert material["decision_context"]["qbc_candidates"] == [{"candidate_id": "a", "ehull_per_atom": .01}]
    assert "invalid_action" not in material


def test_proposal_passes_real_langgraph_contract():
    from decision_layer.agent.decision_backend import request_decision_action
    client = create_proposal_client(OfflineModel(responses=[AIMessage(content=
        '{"tool":"select_dft_candidates","parameters":{"decisions":[{"candidate_id":"a",'
        '"action":"DFT_SINGLE_POINT","reason":"near hull"}]},"budget":30}')]))
    result = request_decision_action(client, {"instruction": "Choose one action", "allowed_tools": ["select_dft_candidates"],
        "decision_context": {"qbc_candidates": [{"candidate_id": "a", "ehull_per_atom": .01}]}})
    assert result["parameters"]["decisions"][0]["candidate_id"] == "a"
    assert result["_llm_usage"]["calls"] == 1

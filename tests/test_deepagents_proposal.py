import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from phase_agent.decisions.agent.deepagents_proposal import create_proposal_client, proposal_material


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
    from phase_agent.decisions.agent.decision_backend import request_decision_action
    client = create_proposal_client(OfflineModel(responses=[AIMessage(content=
        '{"tool":"select_dft_candidates","parameters":{"decisions":[{"candidate_id":"a",'
        '"action":"DFT_SINGLE_POINT","reason":"near hull"}]},"budget":30}')]))
    result = request_decision_action(client, {"instruction": "Choose one action", "allowed_tools": ["select_dft_candidates"],
        "decision_context": {"qbc_candidates": [{"candidate_id": "a", "ehull_per_atom": .01}]}})
    assert result["parameters"]["decisions"][0]["candidate_id"] == "a"
    assert result["_llm_usage"]["calls"] == 1



def _dft_sampling_payload():
    from phase_agent.analysis.phase.phase_dft_sampling import dft_sampling_reference

    evidence = {
        "status": "available", "decision_made": False, "geometry_key": "g1",
        "model_version": "m1", "hull_version": "h1", "snapshot_evidence_digest": "digest",
        "D_rule": "fixed-v2", "composition_per_O2": {"Na": .5, "Co": 1, "O": 2},
        "phase": "O3", "D_score": 7., "D_status": "available", "D_unit": "dimensionless",
        "D_gap_meV_per_O2": 70., "D_candidate_id": "cluster1", "D_candidate_size": 1,
        "D_candidate_geometry_keys": ["g1"], "D_candidate_structure_paths": ["bound-low.vasp"],
        "D_reference_structure_paths": ["bound-upper.vasp"], "H_unit": "meV/O2",
        "H_layers": [1, 2, 3], "H_single_sequence": [0., 40., None], "H_single_max": 40.,
        "H_D_candidate_union_sequence": [65., 105., None], "H_D_candidate_union_max": 105.,
        "distance_to_layer_hull_meV_per_O2": [20., 0., None],
        "H_path_reference": "bound-review.json", "H_path_scope": "maxima_are_conditional",
        "formal_Ehull_reference": "current_formal_hull_after_executed_exclusions",
        "quality_priority_eligible": True, "sampling_evidence_scope": "recursive_near_hull",
        "endpoint_protected": False, "structure_review": "uncertain",
    }
    qbc = {"status": "available", "energy_per_atom_std": .04, "f_std_max": .3}
    return {
        "allowed_tools": ["select_dft_candidates"], "mode": "autonomous_search",
        "decision_context": {
            "dft_selection_policy": {"reference_framework": dft_sampling_reference()},
            "qbc_candidates": [
                {"candidate_id": "c1", "structure_sha256": "g1", "predicted_Ehull": .02,
                 "ehull_unit": "eV/atom", "phase_diagram_version": "h1", "qbc": qbc,
                 "dft_sampling_evidence": evidence, "structure": [1], "path": "private"},
                {"candidate_id": "c2", "dft_sampling_evidence": {
                    "status": "unavailable", "decision_made": False,
                    "structure_file_status": "unavailable", "reason": "structure_file_unavailable"}},
            ],
        },
    }


def _assert_dft_material(material, payload):
    from phase_agent.decisions.agent.transport_tables import expand_evidence_tables

    actual = expand_evidence_tables(material)["decision_context"]
    expected = payload["decision_context"]
    assert actual["dft_selection_policy"] == expected["dft_selection_policy"]
    assert len(actual["qbc_candidates"]) == 2
    for row, original in zip(actual["qbc_candidates"], expected["qbc_candidates"], strict=True):
        assert row == {k: v for k, v in original.items() if k not in {"structure", "path"}}
    assert "D_score" not in actual["qbc_candidates"][1]["dft_sampling_evidence"]


def test_D_H_and_reference_survive_deep_projection_and_table_roundtrip():
    from copy import deepcopy
    from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
    from phase_agent.decisions.agent.transport_tables import compact_evidence_tables

    payload = _dft_sampling_payload()
    before = deepcopy(payload)
    material = compact_evidence_tables(proposal_material(prepare_llm_request(payload)))
    _assert_dft_material(material, before)
    assert payload == before


def test_offline_deep_model_receives_D_H_QBC_and_adjustable_reference_once():
    import json
    from langchain_core.messages import HumanMessage

    captured = []
    class InspectingModel(OfflineModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            captured.extend(json.loads(m.content) for m in messages if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.startswith('{'))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    payload = _dft_sampling_payload()
    client = create_proposal_client(InspectingModel(responses=[AIMessage(content='{"tool":"check_convergence"}')]))
    result = client(payload)
    assert result["tool"] == "check_convergence"
    assert len(captured) == 1
    _assert_dft_material(captured[0], payload)

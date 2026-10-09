import json
import pytest
from decision_layer.agent.dft_contracts import DFTDecision, dft_contract_errors
from execution_layer.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def test_schema_and_legacy_reason_are_supported():
    json.dumps(DFTDecision.model_json_schema())
    assert not dft_contract_errors([{"candidate_id": "S-1", "action": "DFT_SINGLE_POINT"}])


@pytest.mark.parametrize("change", [{"candidate_id": []}, {"candidate_id": 1},
    {"candidate_id": " "}, {"action": "TRAIN"}, {"reason": 12}, {"energy": -1}])
def test_invalid_wire_fields_never_reach_cost_or_reservation(change):
    state = {"budget_reservations": {}}
    row = {"candidate_id": "S-1", "action": "DFT_SINGLE_POINT", **change}
    result = validate_dft_agent_decisions({"decisions": [row]}, [], state,
        config={}, config_version="v", remaining_budget=100)
    assert not result["valid"]
    assert result["accepted"] == []
    assert state == {"budget_reservations": {}}


def test_schema_does_not_authorize_unknown_candidate():
    row = {"candidate_id": "S-unknown", "action": "DFT_SINGLE_POINT"}
    assert not dft_contract_errors([row])
    result = validate_dft_agent_decisions({"decisions": [row]}, [], {},
        config={}, config_version="v", remaining_budget=100)
    assert "unknown_candidate:S-unknown" in result["errors"]


def test_post_dft_repair_uses_same_dft_schema_once():
    from decision_layer.agent.post_dft_review import request_validated_action
    from tests.test_post_dft_review import review
    requests = []
    def client(payload):
        requests.append(payload)
        return {"tool": "select_dft_candidates",
            "parameters": {"decisions": [{"candidate_id": [] if len(requests) == 1 else "S-1",
                "action": "DFT_SINGLE_POINT"}]},
            "post_dft_review": {**review(), "choice": "supplement_dft"}}
    result = request_validated_action(client, {"instruction": "test",
        "allowed_tools": ["select_dft_candidates"],
        "decision_context": {"post_dft_assessment": {"status": "evaluated"}}})
    assert len(requests) == 2
    assert requests[0]["output_contracts"] == requests[1]["output_contracts"]
    assert "dft_decision_item" in requests[0]["output_contracts"]
    assert any("candidate_id" in e for e in requests[1]["validation_errors"])
    assert result["parameters"]["decisions"][0]["candidate_id"] == "S-1"

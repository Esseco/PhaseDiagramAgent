"""DFT re-selection reads current point costs and limits through the DFT contract."""

from copy import deepcopy
import pytest
from phase_agent.decisions.agent.summary_request import summary_payload, request_summary_first
from phase_agent.decisions.agent.hybrid_proposal import create_hybrid_client
from tests.test_operation_evidence_layers import fixture

FIELDS = {
    "dft_candidate_costs": [
        {"candidate_id": "57", "single_point_cost": 12.3456789012345, "relax_cost": 40}
    ],
    "dft_stage_limits": {"dft_single_point": {"max_tasks": 100, "max_cost": 6000}},
    "independent_audit_candidate_ids": ["57"],
}


@pytest.mark.parametrize("field", FIELDS)
def test_current_selection_evidence_is_indexed_under_dft(field):
    payload = fixture()
    payload["decision_context"].update(deepcopy(FIELDS))
    request, _ = summary_payload(payload)
    assert field in request["evidence_reader"]["field_index"]["dft"]
    assert field not in request["evidence_reader"]["field_index"].get("history", {})


def test_hybrid_reselection_reads_exact_costs_without_contract_repair():
    payload = fixture()
    payload["state"]["config_version"] = "current-config"
    payload["decision_context"].update(deepcopy(FIELDS))
    before = deepcopy(payload)
    seen = []

    def provider(request):
        seen.append(deepcopy(request))
        assert "validation_feedback" not in request["evidence_reader"]
        if len(seen) == 1:
            return {"kind": "inspect_evidence", "topics": ["dft"], "fields": list(FIELDS)}
        if len(seen) == 2:
            assert request["state"]["config_version"] == "current-config"
            for field, value in FIELDS.items():
                assert request["decision_context"][field] == value
            assert "available_branches" not in request["decision_context"]
            return {
                "kind": "inspect_evidence",
                "topics": ["dft", "configuration", "budget", "model", "tasks", "convergence"],
            }
        return {"tool": "select_dft_candidates", "target_ids": ["57"]}

    def forbidden():
        raise AssertionError("No harness switch")

    action = request_summary_first(create_hybrid_client(provider, forbidden), payload)
    assert action["target_ids"] == ["57"]
    assert len(seen) == 3 and payload == before

from copy import deepcopy

import pytest

from phase_agent.decisions.agent.summary_request import TOPICS, request_summary_first
from tests.test_operation_evidence_layers import fixture


@pytest.mark.parametrize(
    "invalid,detail",
    [
        ({"topics": ["DFT"]}, "topics"),
        ({"topics": ["model"], "fields": ["qbc_candidates"]}, "qbc_candidates"),
        ({"full_for": ["submit_everything"]}, "submit_everything"),
        ({"execute": True}, "execute"),
        (
            {
                "analysis_plan": {
                    "stage": "DFT",
                    "question": "choose",
                    "reasoning": "skip_validation",
                }
            },
            "reasoning",
        ),
        (
            {
                "analysis_plan": {
                    "stage": "DFT",
                    "question": "choose",
                    "reasoning": "scientific",
                    "comparison_tools": ["submit_everything"],
                }
            },
            "submit_everything",
        ),
    ],
)
def test_one_invalid_read_can_be_corrected_without_expanding_evidence(invalid, detail):
    payload = fixture()
    before = deepcopy(payload)
    seen = []

    def client(request):
        seen.append(deepcopy(request))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["dft"],
                **invalid,
                "_llm_usage": {"calls": 1, "input_tokens": 10},
            }
        if len(seen) == 2:
            feedback = request["evidence_reader"]["validation_feedback"]
            assert detail in str(feedback["errors"])
            assert feedback["attempt"] == 1 and feedback["max_attempts"] == 1
            assert request["evidence_reader"]["phase"] == "select_topics"
            assert "qbc_candidates" not in request["decision_context"]
            assert request["allowed_tools"] == payload["allowed_tools"]
            return {
                "kind": "inspect_evidence",
                "topics": list(TOPICS),
                "_llm_usage": {"calls": 1, "input_tokens": 20},
            }
        assert "validation_feedback" not in request["evidence_reader"]
        assert (
            request["decision_context"]["qbc_candidates"]
            == payload["decision_context"]["qbc_candidates"]
        )
        return {
            "tool": "select_dft_candidates",
            "target_ids": ["57"],
            "_llm_usage": {"calls": 1, "input_tokens": 30},
        }

    action = request_summary_first(client, payload)
    assert action["target_ids"] == ["57"] and len(seen) == 3
    assert action["_llm_usage"]["calls"] == 3
    assert action["_llm_usage"]["input_tokens"] == 60
    assert payload == before


def test_persistent_invalid_read_stops_after_one_correction_and_keeps_cause():
    calls = []

    def client(request):
        calls.append(deepcopy(request))
        return {
            "kind": "inspect_evidence",
            "topics": ["dft"],
            "full_for": ["submit_everything"],
            "_llm_usage": {"calls": 1, "input_tokens": 5},
        }

    with pytest.raises(ValueError, match="只读契约.*submit_everything") as error:
        request_summary_first(client, fixture())
    assert len(calls) == 2 and error.value.llm_usage["calls"] == 2
    assert all("qbc_candidates" not in request["decision_context"] for request in calls)


def test_correction_cannot_return_scientific_action_instead_of_read_request():
    calls = []

    def client(request):
        calls.append(deepcopy(request))
        return (
            {"kind": "inspect_evidence", "topics": ["wrong"]}
            if len(calls) == 1
            else {"tool": "select_dft_candidates", "target_ids": ["57"]}
        )

    with pytest.raises(ValueError, match="只读.*inspect_evidence"):
        request_summary_first(client, fixture())
    assert len(calls) == 2


def test_validation_diagnostic_records_field_errors_without_raw_input_values():
    from phase_agent.runtime.turn_process import _current

    record = {"events": []}
    token = _current.set(record)
    seen = []

    def client(request):
        seen.append(deepcopy(request))
        return {
            "kind": "inspect_evidence",
            "topics": ["dft"],
            "unexpected_field": "PRIVATE-VALUE-DO-NOT-LOG",
        }

    try:
        with pytest.raises(ValueError, match="unexpected_field"):
            request_summary_first(client, fixture())
        assert record["events"]
        assert "unexpected_field" in str(record["events"])
        assert "PRIVATE-VALUE-DO-NOT-LOG" not in str(record)
        assert "PRIVATE-VALUE-DO-NOT-LOG" not in str(
            seen[-1]["evidence_reader"]["validation_feedback"]
        )
    finally:
        _current.reset(token)


def test_correction_allowance_is_not_reset_after_a_successful_read():
    calls = []

    def client(request):
        calls.append(deepcopy(request))
        if len(calls) == 2:
            return {"kind": "inspect_evidence", "topics": ["model"]}
        return {"kind": "inspect_evidence", "topics": ["wrong"]}

    with pytest.raises(ValueError, match="一次纠正后仍无效"):
        request_summary_first(client, fixture())
    assert len(calls) == 3
    assert "validation_feedback" not in calls[-1]["evidence_reader"]


def test_hybrid_client_metadata_does_not_invalidate_valid_evidence_request():
    from phase_agent.decisions.agent.hybrid_proposal import create_hybrid_client

    seen = []

    def provider(request):
        seen.append(deepcopy(request))
        if len(seen) == 1:
            return {"kind": "inspect_evidence", "topics": list(TOPICS)}
        return {"tool": "select_dft_candidates", "target_ids": ["57"]}

    def unexpected_deep_client():
        raise AssertionError("no model/harness switch for this routine request")

    client = create_hybrid_client(provider, unexpected_deep_client)
    action = request_summary_first(client, fixture())
    assert action["target_ids"] == ["57"] and len(seen) == 2
    assert all("validation_feedback" not in request["evidence_reader"] for request in seen)
    assert action["_proposal_harness"] == "legacy"
    assert action["_analysis_route"]["harness"] == "legacy"


def test_hybrid_metadata_exception_does_not_allow_other_private_fields():
    from phase_agent.decisions.agent.hybrid_proposal import create_hybrid_client

    calls = []

    def provider(request):
        calls.append(deepcopy(request))
        return {"kind": "inspect_evidence", "topics": ["model"], "_execute": True}

    client = create_hybrid_client(provider, lambda: None)
    with pytest.raises(ValueError, match="_execute"):
        request_summary_first(client, fixture())
    assert len(calls) == 2

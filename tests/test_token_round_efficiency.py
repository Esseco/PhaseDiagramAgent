from copy import deepcopy
import json
from unittest.mock import patch
from phase_agent.decisions.agent.summary_request import summary_payload, request_summary_first
from phase_agent.decisions.agent.prepare_llm_request import (
    prepare_llm_request,
    needs_deep_reasoning,
)
from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
from phase_agent.decisions.agent.transport_tables import (
    table_rows,
    expand_evidence_tables,
    evidence_row_count,
)
from phase_agent.runtime.turn_process import _current


def payload():
    return {
        "mode": "autonomous_search",
        "allowed_tools": ["check_convergence"],
        "instruction": "scientific analysis required",
        "output_contracts": {"scientific_decision": {"required": ["rationale"]}},
        "state": {
            "current_project": {"configuration": {"limits": 3}},
            "available_budget": 12,
            "mlip_status": {"active_version": "m1"},
            "human_constraints": ["do not create tasks"],
        },
        "decision_context": {
            "qbc_candidates": [{"candidate_id": str(i), "ehull": i / 100} for i in range(175)],
            "current_phase_diagram": {"status": "completed"},
            "task_stage_evidence": {"completed": 897},
        },
    }


def test_planning_contract_has_no_scientific_schemas_then_restores_them():
    original = payload()
    before = deepcopy(original)
    seen = []

    def model(req):
        seen.append(deepcopy(req))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["configuration", "budget", "model", "tasks", "dft", "convergence"],
            }
        return {"tool": "check_convergence"}

    request_summary_first(model, original)
    first = prepare_llm_request(seen[0])
    last = prepare_llm_request(seen[-1])
    assert len(seen) == 2 and set(first["output_contracts"]) == {"evidence_request"}
    assert not needs_deep_reasoning(seen[0]) and needs_deep_reasoning(seen[-1])
    assert "scientific_decision" in last["output_contracts"]
    assert seen[-1]["instruction"] == original["instruction"]
    assert first["state"]["human_constraints"] == ["do not create tasks"]
    assert len(seen[-1]["decision_context"]["qbc_candidates"]) == 175
    assert original == before


def test_path_prefix_tables_are_lossless_for_repeats_missing_and_unicode():
    rows = [
        {
            "candidate_id": str(i),
            "structure_path": r"E:\project\submissions\verified\epoch0_RelaxMC_Batch01\results/"
            + str(i // 2)
            + ".vasp",
            "ehull": i / 100,
        }
        for i in range(175)
    ]
    for values in [rows, rows + [{"candidate_id": "特殊", "structure_path": None}]]:
        original = deepcopy(values)
        table = table_rows(values)
        assert expand_evidence_tables(table) == original and values == original
        assert evidence_row_count(table) == len(values)
        assert len(json.dumps(table)) < len(json.dumps(values)) * 0.65


def test_provider_uses_two_phases_and_records_finish_reasons():
    responses = [
        {
            "kind": "inspect_evidence",
            "topics": ["configuration", "budget", "model", "tasks", "dft", "convergence"],
        },
        {"tool": "check_convergence"},
    ]
    bodies = []

    def provider(request, **kwargs):
        bodies.append(json.loads(request.data))
        action = responses[len(bodies) - 1]

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": json.dumps(action)}}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ).encode()

        return Response()

    record = {"id": "fake-turn", "events": [], "timings": []}
    token = _current.set(record)
    try:
        client = create_deepseek_client(api_key="fake", thinking="enabled", max_tokens=2400)
        with patch(
            "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
            side_effect=provider,
        ):
            result = request_summary_first(client, payload())
        assert result["tool"] == "check_convergence" and len(bodies) == 2
        assert bodies[0]["thinking"] == {"type": "disabled"} and bodies[0]["max_tokens"] == 1600
        assert bodies[1]["thinking"] == {"type": "enabled"} and bodies[1]["max_tokens"] >= 8192
        assert [r["phase"] for r in record["model_requests"]] == ["evidence_topics", "analysis"]
        assert all(
            r["finish_reason"] == "stop" and r["response_chars"] > 0
            for r in record["model_requests"]
        )
    finally:
        _current.reset(token)


def test_exact_reference_index_is_available_without_expanding_historical_reports():
    original = payload()
    original["decision_context"]["evidence_catalog"] = {
        "references": {
            "phase_diagram:mlip:v1": {"kind": "phase_diagram", "report": {"rows": ["large"] * 200}}
        }
    }
    summary, sources = summary_payload(original)
    index = summary["decision_context"]["evidence_reference_index"]
    assert list(index["references"]) == ["phase_diagram:mlip:v1"]
    assert "report" not in index["references"]["phase_diagram:mlip:v1"]
    assert sources["evidence_catalog"] == original["decision_context"]["evidence_catalog"]

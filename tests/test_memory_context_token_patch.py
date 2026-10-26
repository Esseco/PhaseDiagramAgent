from copy import deepcopy

from phase_agent.analysis.phase import phase_quality_reports
from phase_agent.decisions.agent.prepare_llm_request import needs_deep_reasoning, prepare_llm_request
from phase_agent.decisions.agent.summary_request import request_summary_first


def test_phase_quality_report_reuses_fingerprint(monkeypatch):
    calls = []

    def diagnose(*args, **kwargs):
        calls.append(1)
        return {"status": "ok", "structures": []}

    monkeypatch.setattr(phase_quality_reports, "diagnose_phase_diagram", diagnose)
    monkeypatch.setattr(phase_quality_reports, "phase_boundary_rule", lambda state: {})
    state = {"active_model_version": "m1", "tasks": [], "phase_diagrams": {}}
    snapshot = {"entries": []}
    first = phase_quality_reports.get_phase_quality_report(state, snapshot=snapshot)
    second = phase_quality_reports.get_phase_quality_report(state, snapshot=snapshot)
    assert len(calls) == 1
    assert first == second
    assert first is not second


def test_transport_drops_exact_context_aliases_but_keeps_constraints_and_approval_facts():
    payload = {
        "state": {
            "user_message": "继续",
            "human_constraints": ["预算不变"],
            "pending_plans": {"p": {"plan_id": "p"}},
            "shared_evidence": {"count": 1},
        },
        "decision_context": {
            "shared_evidence": {"count": 1},
            "human_constraints": ["预算不变"],
            "pending_plans": {"p": {"plan_id": "p"}},
        },
    }
    original = deepcopy(payload)
    result = prepare_llm_request(payload)
    assert payload == original
    assert "shared_evidence" not in result["state"]
    assert result["decision_context"]["shared_evidence"] == {"count": 1}
    assert result["state"]["user_message"] == "继续"
    assert result["state"]["human_constraints"] == ["预算不变"]
    assert result["state"]["pending_plans"] == {"p": {"plan_id": "p"}}


def test_evidence_reader_and_large_history_do_not_enable_reasoning_by_themselves():
    payload = {
        "mode": "unified_dialogue",
        "evidence_reader": {"phase": "analyze", "field_index": {"history": {}}},
        "state": {"event_history": [{"text": "x" * 100000}]},
        "decision_context": {},
    }
    assert not needs_deep_reasoning(payload)
    payload["evidence_reader"]["phase"] = "select_topics"
    payload["state"]["qbc_candidates"] = [{"candidate_id": "a"}]
    assert not needs_deep_reasoning(payload)


def test_routine_verification_does_not_preload_unselected_comparison_pools():
    payload = {
        "mode": "autonomous_search",
        "allowed_tools": ["select_dft_candidates", "generate_branches"],
        "state": {"current_project": {"configuration": {"cost": 1}}},
        "decision_context": {
            "qbc_candidates": [{"candidate_id": "q"}],
            "available_branches": [{"branch_id": "b"}],
            "budget_usage": {},
            "available_budget": 1,
            "mlip_status": {"active_version": "m1"},
            "tasks": [],
            "current_convex_hull": {},
        },
    }
    seen = []

    def model(request):
        seen.append(request)
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "view": "full",
                "topics": ["dft"],
                "analysis_plan": {
                    "stage": "verify saved selection",
                    "question": "check existing candidate",
                    "reasoning": "routine",
                    "analysis_kind": "verification",
                    "comparison_tools": ["select_dft_candidates", "generate_branches"],
                },
            }
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, payload)
    assert "qbc_candidates" in seen[-1]["decision_context"]
    assert "available_branches" not in seen[-1]["decision_context"]

from copy import deepcopy

import pytest

from tests.test_stage_aware_evidence import fixture
from phase_agent.decisions.agent.summary_request import request_summary_first, summary_payload
from phase_agent.decisions.agent.prepare_llm_request import needs_deep_reasoning


def plan(tools=None, kind="scientific_comparison", reasoning="scientific"):
    return {
        "stage": "registered branches awaiting search",
        "question": "Compare scientific alternatives",
        "analysis_kind": kind,
        "reasoning": reasoning,
        "comparison_tools": tools or [],
    }


def test_registration_and_search_milestones_are_distinct_and_version_bound():
    p = fixture()
    request, _ = summary_payload(p)
    m = request["decision_context"]["stage_snapshot"]["branch_milestones"]
    assert m["registration_completed"] is True
    assert m["registered_branch_count"] == 58
    assert m["first_search_completed"] is None
    p["decision_context"]["new_model_generation"]["first_search_completed"] = False
    request, _ = summary_payload(p)
    assert request["decision_context"]["stage_snapshot"]["branch_milestones"][
        "registration_completed"
    ]
    p["decision_context"]["new_model_generation"]["model_version"] = "old"
    request, _ = summary_payload(p)
    assert (
        request["decision_context"]["stage_snapshot"]["branch_milestones"]["registration_completed"]
        is None
    )


def test_generation_policy_not_applying_is_not_proof_of_registered_branches():
    p = fixture()
    p["decision_context"]["new_model_generation"].update(
        required=False, ready=True, new_branch_count=0
    )
    request, _ = summary_payload(p)
    assert (
        request["decision_context"]["stage_snapshot"]["branch_milestones"]["registration_completed"]
        is False
    )


def test_compared_route_prerequisites_are_loaded_before_the_first_analysis():
    p = fixture()
    before = deepcopy(p)
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["model"],
                "fields": ["new_model_generation"],
                "analysis_plan": plan(["select_dft_candidates", "prepare_local_batch_files"]),
            }
        return {"tool": "select_dft_candidates"}

    result = request_summary_first(model, p)
    assert len(seen) == 2 and result["tool"] == "select_dft_candidates"
    assert seen[1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]
    assert (
        seen[1]["decision_context"]["available_branches"]
        == p["decision_context"]["available_branches"]
    )
    assert p == before and seen[1]["allowed_tools"] == p["allowed_tools"]


def test_agent_can_change_to_a_route_not_declared_in_the_plan():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["model"],
                "analysis_plan": plan(["prepare_local_batch_files"]),
            }
        return {"tool": "select_dft_candidates"}

    assert request_summary_first(model, p)["tool"] == "select_dft_candidates"
    assert len(seen) == 3
    assert seen[-1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]


@pytest.mark.parametrize(
    "kind,tools",
    [
        ("scientific_comparison", []),
        ("verification", ["generate_branches"]),
        ("input_preparation", ["select_dft_candidates"]),
        ("verification", ["prepare_local_batch_files", "pause_search"]),
    ],
)
def test_scientific_or_ambiguous_comparison_cannot_disable_reasoning(kind, tools):
    p = fixture()
    p["analysis_plan"] = plan(tools, kind, reasoning="routine")
    assert needs_deep_reasoning(p)


def test_genuine_input_preparation_can_stay_routine():
    p = fixture()
    p["analysis_plan"] = plan(["prepare_local_batch_files"], "input_preparation", "routine")
    assert not needs_deep_reasoning(p)


def test_old_scope_without_a_declared_kind_falls_back_to_scientific():
    p = fixture()
    p["analysis_plan"] = {"stage": "review", "question": "choose", "reasoning": "routine"}
    assert needs_deep_reasoning(p)


def test_unknown_comparison_tool_is_rejected_before_analysis():
    seen = []

    def model(r):
        seen.append(r)
        return {
            "kind": "inspect_evidence",
            "topics": ["model"],
            "analysis_plan": plan(["submit_everything"]),
        }

    with pytest.raises(ValueError):
        request_summary_first(model, fixture())
    assert len(seen) == 2


def test_preflight_normalizes_effective_reasoning_and_preserves_legal_tools():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["mc"],
                "analysis_plan": plan(["generate_branches"], reasoning="routine"),
            }
        return {"tool": "generate_branches"}

    request_summary_first(model, p)
    assert len(seen) == 2
    assert seen[1]["analysis_plan"]["reasoning"] == "scientific"
    assert seen[1]["allowed_tools"] == p["allowed_tools"]


def test_agent_can_escalate_thinking_with_loaded_evidence_without_rereading_it():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) <= 2:
            scope = plan(["generate_branches"])
            scope.update(
                reasoning_effort="low" if len(seen) == 1 else "high",
                reasoning_basis="check an unresolved scientific conflict",
            )
            return {"kind": "inspect_evidence", "topics": ["mc"], "analysis_plan": scope}
        return {"tool": "generate_branches"}

    assert request_summary_first(model, p)["tool"] == "generate_branches"
    assert len(seen) == 3
    assert seen[1]["decision_context"] == seen[2]["decision_context"]
    assert seen[2]["analysis_plan"]["reasoning_effort"] == "high"
    assert seen[2]["allowed_tools"] == p["allowed_tools"]

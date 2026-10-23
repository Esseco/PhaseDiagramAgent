from copy import deepcopy
import pytest
from phase_agent.decisions.agent.summary_request import request_summary_first, summary_payload
from phase_agent.decisions.agent.prepare_llm_request import needs_deep_reasoning


def fixture():
    return {
        "mode": "autonomous_search",
        "allowed_tools": [
            "prepare_local_batch_files",
            "select_dft_candidates",
            "generate_branches",
        ],
        "state": {
            "active_model_version": "m2",
            "current_project": {"configuration": {"cost_limit": 100}},
            "available_budget": 100,
            "human_constraints": ["preserve all constraints"],
            "tasks": [{"task_id": "t1", "model_version": "m2", "status": "completed"}],
        },
        "decision_context": {
            "relax_mc_batches": [
                {
                    "bundle_id": "b1",
                    "model_version": "m2",
                    "return_error": "unverified",
                    "plan_checksum": "p",
                }
            ],
            "new_model_generation": {"model_version": "m2", "ready": True, "new_branch_count": 58},
            "scientific_decision_brief": {"coverage": {"measured": 0.12345678901234567}},
            "decision_feedback": [{"historical_rationale": "x" * 10000}],
            "available_branches": [
                {"branch_id": str(i), "energy": -i, "model_version": "m2"} for i in range(238)
            ],
            "qbc_candidates": [
                {"candidate_id": str(i), "ehull": i / 1000, "sigma": i / 10000} for i in range(58)
            ],
            "current_convex_hull": {"status": "unknown"},
            "task_stage_evidence": {"m2": {"completed": 1}},
        },
    }


def read(operation="recover_relax_mc_batch"):
    return {
        "kind": "inspect_evidence",
        "topics": ["history"],
        "fields": ["relax_mc_batches"],
        "analysis_plan": {
            "stage": "registered batch",
            "question": "verify existing returned result",
            "reasoning": "routine",
            "analysis_kind": "verification",
            "comparison_tools": ["prepare_local_batch_files"],
            "comparison_operations": [operation],
        },
    }


def test_existing_return_verification_preloads_receipts_without_reopening_scientific_pools():
    p = fixture()
    before = deepcopy(p)
    seen = []

    def model(request):
        seen.append(deepcopy(request))
        return read() if len(seen) == 1 else {"operation": "recover_relax_mc_batch"}

    assert request_summary_first(model, p)["operation"] == "recover_relax_mc_batch"
    assert len(seen) == 2 and not needs_deep_reasoning(seen[1])
    context = seen[1]["decision_context"]
    assert context["relax_mc_batches"] == p["decision_context"]["relax_mc_batches"]
    assert context["tasks"] == p["state"]["tasks"]
    assert seen[1]["state"]["human_constraints"] == p["state"]["human_constraints"]
    for key in [
        "qbc_candidates",
        "available_branches",
        "scientific_decision_brief",
        "decision_feedback",
    ]:
        assert (
            key not in context
            and key
            in seen[1]["evidence_reader"]["field_index"][
                "dft"
                if key == "qbc_candidates"
                else "mc"
                if key == "available_branches"
                else "configuration"
                if key == "scientific_decision_brief"
                else "history"
            ]
        )
    assert seen[1]["allowed_tools"] == p["allowed_tools"] and p == before


def test_changed_scientific_route_still_loads_complete_pool_and_original_prerequisites():
    p = fixture()
    seen = []

    def model(request):
        seen.append(deepcopy(request))
        return read() if len(seen) == 1 else {"tool": "select_dft_candidates"}

    assert request_summary_first(model, p)["tool"] == "select_dft_candidates"
    assert len(seen) == 3
    assert seen[-1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]
    assert (
        seen[-1]["decision_context"]["scientific_decision_brief"]
        == p["decision_context"]["scientific_decision_brief"]
    )
    assert (
        seen[-1]["decision_context"]["current_convex_hull"]
        == p["decision_context"]["current_convex_hull"]
    )


def test_new_batch_preparation_keeps_original_full_scientific_evidence_and_thinking():
    p = fixture()
    seen = []

    def model(request):
        seen.append(deepcopy(request))
        return (
            read("prepare_relax_mc_batch")
            if len(seen) == 1
            else {"operation": "prepare_relax_mc_batch"}
        )

    request_summary_first(model, p)
    assert len(seen) == 2 and needs_deep_reasoning(seen[1])
    assert (
        seen[1]["decision_context"]["available_branches"]
        == p["decision_context"]["available_branches"]
    )
    assert (
        seen[1]["decision_context"]["scientific_decision_brief"]
        == p["decision_context"]["scientific_decision_brief"]
    )


@pytest.mark.parametrize("operation", ["not_an_operation", "activate_mlip_candidate"])
def test_invalid_or_disallowed_operation_never_reaches_analysis(operation):
    calls = []

    def model(r):
        calls.append(r)
        return read(operation)

    with pytest.raises(ValueError, match="只读契约"):
        request_summary_first(model, fixture())
    assert len(calls) == 2


def test_direct_recovery_proposal_cannot_skip_receipts_but_does_not_load_candidate_pools():
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        return {"operation": "recover_relax_mc_batch"}

    request_summary_first(model, fixture())
    assert len(seen) == 2
    assert "relax_mc_batches" in seen[-1]["decision_context"]
    assert "qbc_candidates" not in seen[-1]["decision_context"]


def test_precise_optional_scientific_field_remains_readable_during_verification():
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            action = read()
            action["topics"] += ["configuration"]
            action["fields"] += ["scientific_decision_brief"]
            return action
        return {"operation": "recover_relax_mc_batch"}

    request_summary_first(model, fixture())
    assert (
        seen[-1]["decision_context"]["scientific_decision_brief"]
        == fixture()["decision_context"]["scientific_decision_brief"]
    )


def test_operation_index_does_not_remove_other_legal_scientific_choices():
    p = fixture()
    r, _ = summary_payload(p)
    assert r["allowed_tools"] == p["allowed_tools"]
    assert "generate_branches" in r["evidence_reader"]["operation_scopes"]
    assert "recover_relax_mc_batch" in r["evidence_reader"]["operation_scopes"]


def test_supporting_memory_is_indexed_but_scientific_metrics_and_unknown_evidence_stay_loaded():
    p = fixture()
    p["decision_context"]["unclassified_scientific_observation"] = {"energy": -0.12345678901234567}
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            action = read("generate_branches")
            action["analysis_plan"]["comparison_tools"] = ["generate_branches"]
            return action
        return {"tool": "generate_branches"}

    request_summary_first(model, p)
    c = seen[-1]["decision_context"]
    assert (
        c["unclassified_scientific_observation"]
        == p["decision_context"]["unclassified_scientific_observation"]
    )
    assert c["available_branches"] == p["decision_context"]["available_branches"]
    assert "decision_feedback" not in c
    assert seen[-1]["evidence_reader"]["field_index"]["history"]["decision_feedback"][
        "supporting_memory"
    ]
    assert c["scientific_decision_brief"] == p["decision_context"]["scientific_decision_brief"]


def test_supporting_memory_can_be_read_completely_during_a_scientific_comparison():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            action = read("generate_branches")
            action["analysis_plan"]["comparison_tools"] = ["generate_branches"]
            action["fields"].append("decision_feedback")
            return action
        return {"tool": "generate_branches"}

    request_summary_first(model, p)
    assert (
        seen[-1]["decision_context"]["decision_feedback"]
        == p["decision_context"]["decision_feedback"]
    )


def test_post_refresh_prompts_acknowledge_completed_registration_without_weakening_the_gate():
    from tests.test_new_model_generation_gate import switched, generated
    from phase_agent.tools.policy.model_generation import (
        new_model_generation_facts,
        new_model_search_errors,
    )
    from phase_agent.analysis.state.training_context import post_refresh_search_context

    before = switched()
    after = generated()
    assert "先生成" in post_refresh_search_context(before)["instruction"]
    assert new_model_search_errors(before) == ["new_model_branch_generation_required"]
    assert "首次生成前置已满足" in post_refresh_search_context(after)["instruction"]
    assert "首次生成前置已满足" in new_model_generation_facts(after)["instruction"]
    assert new_model_search_errors(after) == []
    after["generation_history"][-1]["model_version"] = "stale"
    assert new_model_search_errors(after) == ["new_model_branch_generation_required"]


def test_ordering_recovery_reads_real_registered_plan_field_without_loading_new_candidates():
    p = fixture()
    p["decision_context"]["ordering_searches"] = {
        "plan1": {"model_version": "m2", "status": "prepared", "checksum": "checked"}
    }
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        return (
            read("recover_ordering_search")
            if len(seen) == 1
            else {"operation": "recover_ordering_search", "parameters": {"plan_id": "plan1"}}
        )

    request_summary_first(model, p)
    assert (
        seen[-1]["decision_context"]["ordering_searches"]
        == p["decision_context"]["ordering_searches"]
    )
    assert "available_branches" not in seen[-1]["decision_context"]

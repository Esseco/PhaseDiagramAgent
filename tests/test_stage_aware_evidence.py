from copy import deepcopy
import json

import pytest

from phase_agent.decisions.agent.summary_request import (
    summary_payload,
    request_summary_first,
    expand_topics,
)
from phase_agent.decisions.agent.prepare_llm_request import needs_deep_reasoning


def fixture():
    return {
        "mode": "autonomous_search",
        "allowed_tools": [
            "prepare_local_batch_files",
            "select_dft_candidates",
            "generate_branches",
        ],
        "instruction": "Choose independently, using current scientific evidence.",
        "output_contracts": {"scientific_decision": {"type": "object"}},
        "state": {
            "active_model_version": "m2",
            "confirmed_config_version": "c2",
            "current_status": "branches_registered",
            "available_budget": 100,
            "tasks": [
                {"task_id": "t", "stage": "relax", "status": "completed", "model_version": "m1"}
            ],
            "action_records": [{"nested_old_state": "x" * 2000}] * 20,
            "event_history": [{"nested_old_state": "y" * 2000}] * 20,
            "human_constraints": ["preserve constraints"],
        },
        "decision_context": {
            "new_model_generation": {"ready": True, "model_version": "m2", "new_branch_count": 58},
            "model_refresh": {"status": "completed", "new_model_version": "m2"},
            "post_dft_assessment": {"model_version": "m1", "error": 0.12},
            "confirmed_configuration": {"limits": {"cost": 100}},
            "task_stage_evidence": {"completed": 1},
            "available_branches": [{"branch_id": "b1", "ehull": 0.2, "model_version": "m2"}],
            "qbc_candidates": [
                {"candidate_id": str(i), "energy": -i, "ehull": i / 10, "sigma": i / 100}
                for i in range(25)
            ],
            "current_convex_hull": {"stable_entries": [{"candidate_id": "0", "energy": 0}]},
            "unclassified_scientific_fact": {"important": True},
        },
    }


def test_phase_facts_precede_selection_without_forcing_an_action():
    p = fixture()
    before = deepcopy(p)
    request, _ = summary_payload(p)
    stage = request["decision_context"]["stage_snapshot"]
    assert stage["active_model_version"] == "m2"
    assert stage["new_model_generation"]["new_branch_count"] == 58
    assert stage["post_dft_assessment"]["matches_active_model"] is False
    assert stage["task_counts_by_model"]["m1"]["completed"] == 1
    assert request["allowed_tools"] == p["allowed_tools"]
    assert request["state"]["human_constraints"] == ["preserve constraints"]
    assert p == before


def test_history_defaults_exclude_repeated_execution_snapshots_but_keep_unknown_science():
    p = fixture()
    request, sources = summary_payload(p)
    expand_topics(request, sources, {"history"})
    text = json.dumps(request)
    assert "nested_old_state" not in text
    assert request["decision_context"]["unclassified_scientific_fact"] == {"important": True}
    assert request["decision_context"]["evidence_catalog"]["history"]["action_records"]["on_demand"]
    assert sources["action_records"] == p["state"]["action_records"]
    assert len(text) < len(json.dumps(p)) * 0.4


def test_exact_history_field_can_be_loaded_in_full_and_cannot_be_executed():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {"kind": "inspect_evidence", "topics": ["history"], "fields": ["action_records"]}
        raise RuntimeError("capture only")

    with pytest.raises(RuntimeError, match="capture only"):
        request_summary_first(model, p)
    assert seen[1]["decision_context"]["action_records"] == p["state"]["action_records"]
    assert "event_history" not in seen[1]["decision_context"]


def test_partial_candidate_read_cannot_bypass_full_scientific_prerequisites():
    p = fixture()
    before = deepcopy(p)
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["dft"],
                "fields": ["post_dft_assessment"],
            }
        return {"tool": "select_dft_candidates"}

    assert request_summary_first(model, p)["tool"] == "select_dft_candidates"
    assert len(seen) == 3
    assert seen[-1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]
    assert (
        seen[-1]["decision_context"]["current_convex_hull"]
        == p["decision_context"]["current_convex_hull"]
    )
    assert p == before


def test_model_chooses_routine_then_can_escalate_without_rereading_evidence():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) <= 2:
            return {
                "kind": "inspect_evidence",
                "topics": ["model"],
                "analysis_plan": {
                    "stage": "branch input review",
                    "question": "Check current evidence",
                    "reasoning": "routine" if len(seen) == 1 else "scientific",
                    "analysis_kind": "verification",
                },
            }
        raise RuntimeError("capture only")

    with pytest.raises(RuntimeError, match="capture only"):
        request_summary_first(model, p)
    assert not needs_deep_reasoning(seen[1])
    assert needs_deep_reasoning(seen[2])
    assert seen[1]["decision_context"] == seen[2]["decision_context"]


@pytest.mark.parametrize(
    "extra",
    [
        {"fields": ["not_a_field"]},
        {"fields": ["qbc_candidates"]},
        {"analysis_plan": {"stage": "review", "question": "check", "reasoning": "skip_validation"}},
        {"tool": "prepare_local_batch_files"},
    ],
)
def test_bad_field_or_plan_never_reaches_analysis(extra):
    calls = []

    def model(r):
        calls.append(r)
        return {"kind": "inspect_evidence", "topics": ["model"], **extra}

    with pytest.raises(ValueError):
        request_summary_first(model, fixture())
    assert len(calls) == 2


def test_source_revision_includes_deferred_history_and_candidates():
    p = fixture()
    first, _ = summary_payload(p)
    p["state"]["action_records"][0]["nested_old_state"] = "changed"
    second, _ = summary_payload(p)
    assert first["evidence_revision"] != second["evidence_revision"]


def test_no_plan_keeps_scientific_reasoning_and_configuration_stays_routine():
    assert needs_deep_reasoning(fixture())
    p = fixture()
    p["mode"] = "configuration_dialogue"
    p["analysis_plan"] = {"stage": "edit", "question": "check", "reasoning": "scientific"}
    assert not needs_deep_reasoning(p)


def test_cross_stage_comparison_retains_all_tools_and_full_candidates():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "topics": ["dft", "convergence", "configuration", "budget", "model", "tasks"],
                "analysis_plan": {
                    "stage": "new branches available",
                    "question": "Compare DFT against search",
                    "reasoning": "scientific",
                },
            }
        return {"tool": "select_dft_candidates", "target_ids": ["24"]}

    result = request_summary_first(model, p)
    assert len(seen) == 2 and result["tool"] == "select_dft_candidates"
    assert needs_deep_reasoning(seen[1])
    assert seen[1]["allowed_tools"] == p["allowed_tools"]
    assert seen[1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]


def test_field_index_survives_loading_full_evidence_reference_catalog():
    p = fixture()
    p["decision_context"]["evidence_catalog"] = {
        "references": {"r": {"kind": "observation", "metrics": [1, 2]}}
    }
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) <= 2:
            return {
                "kind": "inspect_evidence",
                "topics": ["history"],
                "fields": ["evidence_catalog"] if len(seen) == 1 else ["action_records"],
            }
        raise RuntimeError("capture only")

    with pytest.raises(RuntimeError, match="capture only"):
        request_summary_first(model, p)
    assert (
        seen[2]["decision_context"]["evidence_catalog"] == p["decision_context"]["evidence_catalog"]
    )
    assert seen[2]["evidence_reader"]["field_index"]["history"]["action_records"]["on_demand"]
    assert seen[2]["decision_context"]["action_records"] == p["state"]["action_records"]


def test_reading_same_topic_can_fetch_additional_fields_without_false_repetition():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) <= 2:
            return {
                "kind": "inspect_evidence",
                "topics": ["model"],
                "fields": ["model_refresh"] if len(seen) == 1 else ["new_model_generation"],
            }
        raise RuntimeError("capture only")

    with pytest.raises(RuntimeError, match="capture only"):
        request_summary_first(model, p)
    assert "new_model_generation" not in seen[1]["decision_context"]
    assert seen[2]["decision_context"]["new_model_generation"]["new_branch_count"] == 58


def test_invalid_analysis_scope_keeps_conservative_scientific_reasoning():
    p = fixture()
    p["analysis_plan"] = {"reasoning": "routine"}
    assert needs_deep_reasoning(p)


@pytest.mark.parametrize(
    "reasoning, expected_thinking, expected_tokens",
    [
        ("routine", "disabled", 4096),
        ("scientific", "enabled", 8192),
    ],
)
def test_provider_budget_follows_typed_model_scope_without_an_extra_planner_call(
    reasoning, expected_thinking, expected_tokens
):
    from unittest.mock import patch
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
    from tests.test_deepseek_client import FakeResponse, completion

    p = fixture()
    bodies = []

    def provider(request, **kwargs):
        body = json.loads(request.data)
        bodies.append(body)
        if len(bodies) == 1:
            action = {
                "kind": "inspect_evidence",
                "topics": ["configuration", "budget", "model", "tasks", "mc", "convergence"],
                "analysis_plan": {
                    "stage": "branch input review",
                    "question": "Assess the next proposal",
                    "reasoning": reasoning,
                    "analysis_kind": "input_preparation",
                },
            }
        else:
            action = {
                "tool": "prepare_local_batch_files",
                "parameters": {"mode": "relax_mc_inputs"},
            }
        return FakeResponse(completion(json.dumps(action)))

    with patch("urllib.request.urlopen", side_effect=provider):
        result = request_summary_first(create_deepseek_client(api_key="fake"), p)
    assert len(bodies) == 2 and result["tool"] == "prepare_local_batch_files"
    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert bodies[1]["thinking"] == {"type": expected_thinking}
    assert bodies[1]["max_tokens"] == expected_tokens


def test_absent_and_explicitly_empty_candidate_pools_remain_distinct():
    p = fixture()
    p["decision_context"].pop("qbc_candidates")
    with pytest.raises(ValueError, match="缺少完整"):
        request_summary_first(lambda r: {"tool": "select_dft_candidates"}, p)
    p["decision_context"]["qbc_candidates"] = []
    seen = []
    request_summary_first(
        lambda r: seen.append(deepcopy(r)) or {"tool": "select_dft_candidates"}, p
    )
    assert seen[-1]["decision_context"]["qbc_candidates"] == []


def test_default_history_does_not_hide_arbitrary_new_scientific_fields():
    p = fixture()
    p["decision_context"]["new_bias_diagnostic"] = {
        "model": "m2",
        "uncertainty": None,
        "errors": [0.1, 0.2],
    }
    request, sources = summary_payload(p)
    expand_topics(request, sources, {"history"})
    assert (
        request["decision_context"]["new_bias_diagnostic"]
        == p["decision_context"]["new_bias_diagnostic"]
    )


def test_provider_analysis_scope_does_not_trim_human_constraints_or_dft_values():
    from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
    from phase_agent.decisions.agent.transport_tables import expand_evidence_tables

    p = fixture()
    p["analysis_plan"] = {
        "stage": "DFT",
        "question": "Compare every candidate",
        "reasoning": "scientific",
    }
    p["state"]["available_branches"] = [{"branch_id": "b", "energy": -1.234, "structure_path": "x"}]
    p["decision_context"]["qbc_candidates"][-1]["sigma"] = None
    prepared = expand_evidence_tables(prepare_llm_request(p))
    assert prepared["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]
    assert prepared["state"]["human_constraints"] == p["state"]["human_constraints"]


def test_current_instruction_scientific_reviews_and_reserved_budget_are_not_hidden():
    p = fixture()
    p["state"]["user_message"] = (
        "Compare DFT with existing branches; do not assume the usual next step."
    )
    p["state"]["decisions"] = [
        {"model_version": "m2", "scientific_finding": "preserve this assessment"}
    ]
    p["state"]["budget_reservations"] = {"pending": {"cost": 11, "status": "reserved"}}
    request, sources = summary_payload(p)
    assert request["state"]["user_message"] == p["state"]["user_message"]
    expand_topics(request, sources, {"history"})
    for key in ("decisions", "budget_reservations"):
        assert request["decision_context"][key] == p["state"][key]


def test_repeated_execution_snapshots_add_only_index_cost_after_transport_compaction():
    from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request

    p = fixture()

    def prepared_size(payload):
        request, sources = summary_payload(payload)
        expand_topics(request, sources, {"history"})
        return len(json.dumps(prepare_llm_request(request), ensure_ascii=False))

    with_history = prepared_size(p)
    for key in ("action_records", "event_history"):
        p["state"].pop(key)
    assert with_history < prepared_size(p) * 1.2

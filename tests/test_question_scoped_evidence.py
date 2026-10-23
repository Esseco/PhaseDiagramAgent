from copy import deepcopy
import json
import pytest
from phase_agent.decisions.agent.summary_request import request_summary_first
from tests.test_operation_evidence_layers import fixture


def read(view="overview"):
    return {
        "kind": "inspect_evidence",
        "view": view,
        "topics": ["mc", "dft", "configuration", "budget", "tasks", "convergence"],
        "analysis_plan": {
            "stage": "registered",
            "question": "identify the unresolved scientific decision",
            "reasoning": "scientific",
            "comparison_tools": ["generate_branches", "select_dft_candidates"],
        },
    }


def test_overview_does_not_masquerade_as_complete_candidate_evidence():
    original = fixture()
    before = deepcopy(original)
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            return read()
        return {
            "tool": "select_dft_candidates",
            "target_ids": ["57"],
            "reason": "provisional inference",
        }

    action = request_summary_first(model, original)
    assert len(seen) == 3 and action["target_ids"] == ["57"]
    assert "qbc_candidates" not in seen[1]["decision_context"]
    overview = seen[1]["decision_context"]["field_overviews"]["qbc_candidates"]
    assert overview["complete_field"] is False and overview["summary"]["count"] == 58
    assert (
        seen[2]["decision_context"]["qbc_candidates"]
        == original["decision_context"]["qbc_candidates"]
    )
    assert "available_branches" not in seen[2]["decision_context"]
    assert (
        seen[2]["provisional_proposal"]["unvalidated_action"]["reason"] == "provisional inference"
    )
    assert seen[1]["allowed_tools"] == original["allowed_tools"] and original == before
    assert all(
        p["state"]["human_constraints"] == original["state"]["human_constraints"] for p in seen
    )


def test_exact_full_read_after_overview_restores_extreme_metric_and_last_candidate():
    original = fixture()
    original["decision_context"]["qbc_candidates"][-1]["ehull"] = -123.12345678901234
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            return read()
        if len(seen) == 2:
            return {
                "kind": "inspect_evidence",
                "view": "full",
                "topics": ["dft"],
                "fields": ["qbc_candidates"],
                "full_for": ["select_dft_candidates"],
            }
        return {"tool": "select_dft_candidates", "target_ids": ["57"]}

    request_summary_first(model, original)
    assert len(seen) == 3
    assert (
        seen[-1]["decision_context"]["qbc_candidates"]
        == original["decision_context"]["qbc_candidates"]
    )
    assert "qbc_candidates" not in seen[-1]["decision_context"]["field_overviews"]


def test_agent_can_change_to_another_scientific_route_after_narrow_supplementation():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return read()
        return (
            {"tool": "select_dft_candidates"} if len(seen) == 2 else {"tool": "generate_branches"}
        )

    assert request_summary_first(model, p)["tool"] == "generate_branches"
    assert len(seen) == 4
    assert (
        seen[-1]["decision_context"]["available_branches"]
        == p["decision_context"]["available_branches"]
    )
    assert seen[-1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]


def test_overview_has_exact_missing_null_ranges_and_no_unobserved_success():
    p = fixture()
    pool = p["decision_context"]["qbc_candidates"]
    pool[0].pop("sigma")
    pool[1]["sigma"] = None
    pool[-1]["ehull"] = -0.12345678901234567
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return read()
        raise RuntimeError("capture")

    with pytest.raises(RuntimeError, match="capture"):
        request_summary_first(model, p)
    fields = seen[-1]["decision_context"]["field_overviews"]["qbc_candidates"]["summary"]["fields"]
    assert fields["sigma"]["missing_count"] == 1 and fields["sigma"]["null_count"] == 1
    assert fields["ehull"]["finite_min"] == -0.12345678901234567
    assert "complete candidate" in seen[-1]["evidence_reader"]["view_rule"]


def test_repeated_identical_overview_is_rejected_without_endless_rereads():
    calls = []

    def model(p):
        calls.append(p)
        return read()

    with pytest.raises(ValueError, match="上限|重复"):
        request_summary_first(model, fixture())
    assert len(calls) == 2


def test_default_full_and_explicit_full_remain_identical_scientific_reads():
    def run(explicit):
        seen = []

        def model(p):
            seen.append(deepcopy(p))
            if len(seen) == 1:
                action = read("full")
                if not explicit:
                    action.pop("view")
                else:
                    action["full_for"] = action["analysis_plan"]["comparison_tools"]
                return action
            return {"tool": "select_dft_candidates"}

        request_summary_first(model, fixture())
        return seen

    a, b = run(False), run(True)
    assert len(a) == len(b) == 2 and a[-1]["decision_context"] == b[-1]["decision_context"]


def test_unknown_view_is_rejected_before_analysis():
    calls = []

    def model(p):
        calls.append(p)
        return read("hide_required_evidence")

    with pytest.raises(ValueError, match="只读契约"):
        request_summary_first(model, fixture())
    assert len(calls) == 2


def test_overview_does_not_preload_other_routes_just_because_they_are_compared():
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            r = read()
            r["topics"] = ["model"]
            return r
        raise RuntimeError("capture")

    with pytest.raises(RuntimeError, match="capture"):
        request_summary_first(model, fixture())
    c = seen[-1]["decision_context"]
    assert "qbc_candidates" not in c and "qbc_candidates" not in c.get("field_overviews", {})
    assert "available_branches" not in c and "available_branches" not in c.get(
        "field_overviews", {}
    )


def test_overviews_never_summarize_explicit_human_constraints_out_of_rows():
    from phase_agent.decisions.agent.evidence_overview import summarize

    rows = [
        {"id": str(i), "human_constraints": ["do not lose this human constraint"]}
        for i in range(20)
    ]
    assert summarize(rows)["fields"]["human_constraints"]["values"] == [
        r["human_constraints"] for r in rows
    ]


def test_current_full_for_scope_does_not_load_unselected_scientific_alternative_pools():
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            r = read("full")
            r.update(topics=["dft"], fields=["qbc_candidates"], full_for=["select_dft_candidates"])
            return r
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, fixture())
    assert len(seen) == 2
    assert (
        seen[-1]["decision_context"]["qbc_candidates"]
        == fixture()["decision_context"]["qbc_candidates"]
    )
    assert "available_branches" not in seen[-1]["decision_context"]
    assert seen[-1]["allowed_tools"] == fixture()["allowed_tools"]


def test_nested_field_request_loads_complete_root_without_silent_metric_or_row_loss():
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            r = read("full")
            r.update(
                topics=["dft"], fields=["qbc_candidates.ehull"], full_for=["select_dft_candidates"]
            )
            return r
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, fixture())
    assert (
        seen[-1]["decision_context"]["qbc_candidates"]
        == fixture()["decision_context"]["qbc_candidates"]
    )
    assert seen[-1]["evidence_reader"]["expanded_field_requests"] == {
        "qbc_candidates.ehull": "qbc_candidates"
    }


def test_narrow_preload_does_not_prevent_later_route_change_and_original_full_reads():
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            r = read("full")
            r.update(topics=["dft"], full_for=["select_dft_candidates"])
            return r
        return {"tool": "generate_branches"}

    request_summary_first(model, fixture())
    assert (
        len(seen) == 3
        and seen[-1]["decision_context"]["available_branches"]
        == fixture()["decision_context"]["available_branches"]
    )


def test_first_read_has_bound_overview_and_stage_progress_without_pretending_a_full_pool():
    from phase_agent.decisions.agent.summary_request import summary_payload

    p = fixture()
    p["state"].pop("tasks")
    p["decision_context"]["task_round_evidence"] = {
        "groups": [
            {
                "lineage": {"model_version": "m2"},
                "stage_evidence": [{"stage": "deep_search", "status_counts": {"cancelled": 5}}],
            }
        ]
    }
    r, _ = summary_payload(p)
    assert r["decision_context"]["scientific_overview"]["qbc_candidates"]["summary"]["count"] == 58
    assert (
        r["decision_context"]["stage_snapshot"]["task_stage_counts_by_model"]["m2"]["deep_search"][
            "cancelled"
        ]
        == 5
    )
    assert "qbc_candidates" not in r["decision_context"]


def test_full_source_read_removes_only_its_redundant_initial_preview():
    seen = []

    def model(p):
        seen.append(deepcopy(p))
        if len(seen) == 1:
            r = read("full")
            r.update(topics=["dft"], full_for=["select_dft_candidates"])
            return r
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, fixture())
    assert "qbc_candidates" in seen[0]["decision_context"]["scientific_overview"]
    assert "qbc_candidates" not in seen[-1]["decision_context"]["scientific_overview"]
    assert (
        seen[-1]["decision_context"]["qbc_candidates"]
        == fixture()["decision_context"]["qbc_candidates"]
    )


def large_brief_fixture():
    p = fixture()
    p["decision_context"]["scientific_decision_brief"] = {
        "policy": {"schema_version": 1, "human_rule": "must keep this"},
        "model_version": "m2",
        "config_version": "c2",
        "limits": {"budget": 100},
        "unknowns": ["unverified is unknown"],
        "coverage": [
            {"phase": "O3", "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "count": i} for i in range(60)
        ],
        "new_unclassified_scientific_fact": {"exact": -0.12345678901234567},
    }
    return p


def test_scoped_current_pool_decision_keeps_exact_policy_limits_unknowns_and_full_candidate_metrics():
    p = large_brief_fixture()
    before = deepcopy(p)
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            a = read("full")
            a.update(topics=["dft"], full_for=["select_dft_candidates"])
            return a
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, p)
    assert len(seen) == 2
    c = seen[-1]["decision_context"]
    brief = c["scientific_decision_brief"]
    for key in [
        "policy",
        "limits",
        "unknowns",
        "model_version",
        "config_version",
        "new_unclassified_scientific_fact",
    ]:
        assert brief[key] == p["decision_context"]["scientific_decision_brief"][key]
    assert c["qbc_candidates"] == p["decision_context"]["qbc_candidates"]
    assert (
        "coverage" not in brief
        and c["field_overviews"]["scientific_decision_brief"]["complete_field"] is False
    )
    assert p == before


def test_deferred_coverage_can_be_read_exactly_and_is_mandatory_for_new_region_generation():
    p = large_brief_fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            a = read("full")
            a.update(topics=["dft"], full_for=["select_dft_candidates"])
            return a
        return {"tool": "generate_branches"}

    request_summary_first(model, p)
    assert len(seen) == 3
    assert (
        seen[-1]["decision_context"]["scientific_decision_brief"]
        == p["decision_context"]["scientific_decision_brief"]
    )
    assert (
        seen[-1]["decision_context"]["available_branches"]
        == p["decision_context"]["available_branches"]
    )


def test_preflight_topic_labels_do_not_expand_unrequested_configuration_components():
    p = large_brief_fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            a = read("full")
            a.update(topics=["dft"], full_for=["select_dft_candidates"])
            return a
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, p)
    assert "configuration" in seen[-1]["decision_context"]["loaded_topics"]
    assert "coverage" not in seen[-1]["decision_context"]["scientific_decision_brief"]
    assert (
        seen[-1]["decision_context"]["current_project"]
        if "current_project" in seen[-1]["decision_context"]
        else seen[-1]["state"]["current_project"]
    )


def test_exact_authoritative_configuration_already_supplied_is_not_requested_again():
    p = large_brief_fixture()
    p["conversation_facts"] = {"current_project": deepcopy(p["state"]["current_project"])}
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            return {
                "kind": "inspect_evidence",
                "view": "full",
                "topics": ["mc", "tasks", "model", "budget", "convergence"],
                "full_for": [],
                "analysis_plan": {
                    "stage": "registered",
                    "question": "choose current MC allocation",
                    "reasoning": "scientific",
                    "comparison_tools": ["allocate_mc_bohb"],
                },
            }
        return {"tool": "allocate_mc_bohb"}

    p["allowed_tools"].append("allocate_mc_bohb")
    request_summary_first(model, p)
    assert len(seen) == 2
    assert "current_project" in seen[1]["evidence_reader"]["initial_full_fields"]
    assert (
        seen[1]["decision_context"]["available_branches"]
        == p["decision_context"]["available_branches"]
    )
    assert "scientific_decision_brief" not in seen[1]["decision_context"]["scientific_overview"]


def test_nonmatching_initial_configuration_does_not_count_as_an_exact_read():
    p = fixture()
    p["conversation_facts"] = {"current_project": {"configuration": {"cost_limit": 999}}}
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            a = read("full")
            a.update(topics=["dft"], full_for=["select_dft_candidates"])
            return a
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, p)
    assert "current_project" not in seen[-1]["evidence_reader"]["initial_full_fields"]
    assert seen[-1]["state"]["current_project"] == p["state"]["current_project"]


def test_discovery_index_alias_roundtrips_exactly_without_dropping_fields_or_human_rules():
    from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request
    from phase_agent.decisions.agent.transport_tables import expand_evidence_tables

    p = fixture()
    index = {
        "dft": {"qbc_candidates": {"count": 58, "serialized_bytes": 9999}},
        "history": {"new_scientific_fact": {"count": 1, "serialized_bytes": 123}},
    }
    p["decision_context"]["evidence_catalog"] = deepcopy(index)
    p["evidence_reader"] = {"phase": "select_topics", "field_index": deepcopy(index)}
    before = deepcopy(p)
    wire = prepare_llm_request(p)
    assert (
        wire["decision_context"]["evidence_catalog"]["source_ref"] == "/evidence_reader/field_index"
    )
    decoded = expand_evidence_tables(wire)
    assert decoded["decision_context"]["evidence_catalog"] == index
    assert decoded["state"]["human_constraints"] == p["state"]["human_constraints"] and p == before


def test_mc_selection_has_first_call_room_for_branch_envelope_without_disabling_thinking():
    from unittest.mock import patch
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
    from tests.test_deepseek_client import FakeResponse, completion

    payload = {
        "decision_kind": "strategy",
        "output_contracts": {"scientific_decision": {"type": "object"}},
        "decision_context": {
            "mc_entry_conditions": {
                "available_allocations": [
                    {"branch_id": str(i), "planned_relative_cost": 1.5} for i in range(52)
                ]
            }
        },
    }

    def provider(request, **kwargs):
        body = json.loads(request.data)
        assert body["thinking"] == {"type": "enabled"}
        assert body["max_tokens"] > 8192
        return FakeResponse(completion('{"tool":"allocate_mc_bohb"}'))

    with patch("urllib.request.urlopen", side_effect=provider) as request:
        result = create_deepseek_client(api_key="fake")(payload)
    assert result["tool"] == "allocate_mc_bohb" and request.call_count == 1


def test_topic_prefixed_fields_resolve_exact_sources_and_missing_names_are_not_empty_facts():
    p = fixture()
    seen = []

    def model(r):
        seen.append(deepcopy(r))
        if len(seen) == 1:
            a = read("full")
            a.update(
                topics=["dft"],
                fields=["dft.qbc_candidates.ehull", "dft.nonexistent_observation"],
                full_for=["select_dft_candidates"],
            )
            return a
        return {"tool": "select_dft_candidates"}

    request_summary_first(model, p)
    assert len(seen) == 2
    assert seen[1]["decision_context"]["qbc_candidates"] == p["decision_context"]["qbc_candidates"]
    assert "nonexistent_observation" not in seen[1]["decision_context"]
    feedback = seen[1]["evidence_reader"]["unresolved_field_requests"]
    assert feedback["fields"] == ["dft.nonexistent_observation"]
    assert "unobserved" in feedback["status"]
    assert "nonexistent_observation" not in seen[1]["evidence_reader"]["fully_loaded_fields"]

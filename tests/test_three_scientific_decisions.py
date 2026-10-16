"""Fake-runtime evidence/approval tests; no scientific computation or submission."""

from copy import deepcopy
from unittest.mock import patch
import pytest
from phase_agent.analysis.state.scientific_decision_brief import decision_brief
from phase_agent.decisions.agent.scientific_decisions import (
    ScientificDecision,
    attach_decision_contract,
    scientific_decision_errors,
)
from phase_agent.analysis.state.decision_feedback import recent_decision_feedback
from phase_agent.tools.workflows.scientific_module_scope import (
    attach_module_scope,
    prepare_generated_module,
)
from phase_agent.tools.workflows.propose_followup import propose_after_generation
from phase_agent.runtime.proposal_presentation import concise_generation_lines


def review(point="branch_sampling"):
    return dict(
        point=point,
        config_version="c1",
        model_version="m1",
        bottleneck="Na coverage is uneven",
        rationale="Fill the recorded gap",
        alternatives=[dict(route="larger cells", reason="Defer until current coverage improves")],
        expected_outcome="More useful low-energy competitors",
        success_criterion="Compare valid near-hull discoveries and measured cost",
        limitations="No wall-time observations yet",
        selection_purposes=[],
    )


def state():
    return dict(
        active_model_version="m1",
        confirmed_config_version="c1",
        scientific_decision_policy={"schema_version": 1},
        tasks=[],
    )


def action():
    return dict(
        tool="generate_branches",
        task_key="branch1",
        scientific_decision=review(),
        parameters={
            "generation_cost_preview": {"selected_branch_upper": 2, "relax_mc_cost_upper": 10}
        },
    )


@pytest.mark.parametrize(
    "field,value", [("model_version", "old"), ("config_version", "old"), ("point", "dft_sampling")]
)
def test_stale_or_wrong_decision_rejected(field, value):
    a = action()
    a["scientific_decision"][field] = value
    assert scientific_decision_errors(a, decision_brief(state()))


def test_good_branch_decision_has_no_invented_results():
    assert not scientific_decision_errors(action(), decision_brief(state()))
    assert "energy" not in ScientificDecision.model_json_schema()["properties"]


@pytest.mark.parametrize("ids", [[], ["x", "x"], ["other"]])
def test_dft_purposes_must_match_selected_batch(ids):
    a = action()
    a.update(tool="select_dft_candidates", scientific_decision=review("dft_sampling"))
    a["parameters"] = {"decisions": [{"candidate_id": "x", "action": "DFT_SINGLE_POINT"}]}
    a["scientific_decision"]["selection_purposes"] = [
        {"candidate_id": i, "purpose": "coverage"} for i in ids
    ]
    assert scientific_decision_errors(a, decision_brief(state()))


def test_direction_can_select_dft_but_assigns_each_selected_purpose():
    a = action()
    a.update(tool="select_dft_candidates", scientific_decision=review("round_direction"))
    a["parameters"] = {"decisions": [{"candidate_id": "x", "action": "DFT_SINGLE_POINT"}]}
    a["scientific_decision"]["selection_purposes"] = [
        {"candidate_id": "x", "purpose": "bias_check"}
    ]
    assert not scientific_decision_errors(
        a, decision_brief(state()), {"post_dft_assessment": {"status": "evaluated"}}
    )


def test_legacy_and_maintenance_not_forced_into_scientific_contract():
    payload = {"instruction": "old", "decision_context": {}}
    assert attach_decision_contract(payload) == payload
    assert not scientific_decision_errors(
        {"tool": "prepare_local_batch_files"}, decision_brief(state())
    )


def test_brief_excludes_old_model_and_does_not_claim_exhaustive_coverage():
    s = state()
    s["tasks"] = [dict(task_id="old", model_version="m0", branch_id="b", status="completed")]
    s["qbc_candidates"] = [
        dict(candidate_id="old", model_version="m0"),
        dict(candidate_id="new", model_version="m1"),
    ]
    brief = decision_brief(s)
    assert not brief["task_status_by_branch"]
    assert sum(r["count"] for r in brief["candidate_strata"]) == 1
    assert "not exhaustive" in brief["instruction"]
    assert s["tasks"][0]["model_version"] == "m0"


def test_feedback_matches_child_ids_and_parent_keys_once():
    s = state()
    s["action_records"] = [
        dict(
            record_id="r1",
            status="completed",
            agent_proposal={"decision_summary": {"scientific_decision": review()}},
            final_action={"tool": "select_dft_candidates", "task_key": "pick"},
            execution_result={"status": "completed", "result": {"tasks": [{"task_id": "t1"}]}},
        )
    ]
    s["tasks"] = [
        dict(
            task_id="t1",
            task_key="child",
            scientific_parent_task_key="pick",
            status="completed",
            actual_cost=2,
        ),
        dict(task_id="unrelated", status="completed", actual_cost=99),
    ]
    result = recent_decision_feedback(s)[0]["outcome"]
    assert result["matched_task_status"] == {"completed": 1}
    assert result["measured_cost"] == 2


def test_scope_not_added_to_legacy_or_stepwise():
    a = action()
    assert "module_execution_scope" not in attach_module_scope(deepcopy(a), {}, {})["parameters"]
    s = state()
    s["scientific_decision_policy"]["mlip_execution"] = "stepwise"
    assert "module_execution_scope" not in attach_module_scope(deepcopy(a), s, {})["parameters"]


def legacy_scope(a, s, config):
    a = deepcopy(a)
    preview = a["parameters"]["generation_cost_preview"]
    a["parameters"]["module_execution_scope"] = {
        "schema_version": 1,
        "operation": "prepare_relax_mc_batch",
        "config_version": s.get("confirmed_config_version"),
        "model_version": s.get("active_model_version"),
        "branch_upper": preview["selected_branch_upper"],
        "relative_cost_upper": 3 * preview["relax_mc_cost_upper"],
        "maximum_mc_segments": 3,
        "submit_authorized": False,
        "mc_policy": deepcopy(config.get("mc_policy") or {}),
    }
    return a


def module_case():
    config = {"mc_policy": {"max_segments_per_branch": 3}}
    s = state()
    a = legacy_scope(action(), s, config)
    result = {
        "status": "completed",
        "state": s,
        "registered": [{"branch_id": "b1"}],
        "summary": {"registered_structures": 1},
    }
    context = {"effective_config": config, "config_version": "c1"}
    return a, result, context


@pytest.mark.parametrize("change", ["cost", "model", "config", "rules", "branches", "submission"])
def test_module_scope_cannot_expand(change):
    a, r, c = module_case()
    scope = a["parameters"]["module_execution_scope"]
    if change == "model":
        scope["model_version"] = "old"
    if change == "config":
        scope["config_version"] = "old"
    if change == "rules":
        scope["mc_policy"] = {}
    if change == "branches":
        scope["branch_upper"] = 0
    if change == "submission":
        scope["submit_authorized"] = True
    child_cost = 31 if change == "cost" else 10
    with (
        patch(
            "phase_agent.tools.relax_mc_batch.plan.attach_preview",
            side_effect=lambda a, *x: {**a, "budget": child_cost},
        ),
        patch("phase_agent.tools.relax_mc_batch.prepare.prepare_batch") as prepare,
    ):
        out = prepare_generated_module(a, r, c)
    prepare.assert_not_called()
    assert out["module_preparation"]["status"] == "blocked"
    assert out["registered"] == r["registered"]
    assert out["state"]["module_preparation_reports"][0]["branch_ids"] == ["b1"]


def test_scoped_module_prepares_once_without_new_model_or_approval():
    a, r, c = module_case()
    with (
        patch(
            "phase_agent.tools.relax_mc_batch.plan.attach_preview",
            side_effect=lambda a, *x: {**a, "budget": 10},
        ),
        patch(
            "phase_agent.tools.policy.validate_tool_action.validate_tool_action",
            return_value={"valid": True},
        ),
        patch(
            "phase_agent.tools.relax_mc_batch.prepare.prepare_batch",
            return_value={"status": "prepared", "state": r["state"], "bundle_id": "bundle"},
        ) as prepare,
    ):
        out = prepare_generated_module(a, r, c)
    prepare.assert_called_once()
    assert out["submitted"] is False
    assert out["module_preparation"]["status"] == "prepared"
    assert (
        propose_after_generation(
            {}, out, event_loop=lambda *a, **k: pytest.fail("No second model call")
        )
        == out
    )
    lines = concise_generation_lines(a["parameters"], {})
    assert any("审批包含" in line and "30" in line for line in lines)


def test_one_repair_then_invalid_reasoning_stops_before_execution():
    from phase_agent.decisions.agent.post_dft_review import request_validated_action

    calls = []
    payload = {
        "allowed_tools": ["generate_branches"],
        "instruction": "choose",
        "decision_context": {"scientific_decision_brief": decision_brief(state())},
    }

    def model(request):
        calls.append(request)
        return {"tool": "generate_branches", "parameters": {}, "reason": "no reasoning"}

    with pytest.raises(ValueError, match="Invalid scientific decision"):
        request_validated_action(model, payload)
    assert len(calls) == 1
    assert "scientific_decision" in calls[0]["output_contracts"]


def test_valid_reasoning_requires_no_extra_model_call():
    from phase_agent.decisions.agent.post_dft_review import request_validated_action

    calls = []

    def model(request):
        calls.append(request)
        return action()

    payload = {
        "allowed_tools": ["generate_branches"],
        "instruction": "choose",
        "decision_context": {"scientific_decision_brief": decision_brief(state())},
    }
    assert (
        request_validated_action(model, payload)["scientific_decision"]["point"]
        == "branch_sampling"
    )
    assert len(calls) == 1


def test_explicit_stepwise_override_prevents_automatic_packaging():
    a = action()
    a["scientific_decision"]["module_execution"] = "stepwise"
    assert "module_execution_scope" not in attach_module_scope(a, state(), {})["parameters"]


def test_missing_record_identity_cannot_match_unrelated_tasks():
    s = state()
    s["action_records"] = [
        dict(
            agent_proposal={"decision_summary": {}},
            final_action={"tool": "generate_branches", "task_key": "p"},
        )
    ]
    s["action_records"][0]["agent_proposal"]["decision_summary"] = {"rationale": "expectation"}
    s["tasks"] = [dict(task_id="unrelated", status="completed", actual_cost=100)]
    assert recent_decision_feedback(s)[0]["outcome"]["measured_cost"] is None


def test_revised_model_request_does_not_copy_program_approval_scope():
    from phase_agent.decisions.agent.prepare_llm_request import prepare_llm_request

    payload = {
        "state": {},
        "current_proposal": {
            "parameters": {
                "generation_plan": [],
                "generation_cost_preview": {"cost": 1},
                "module_execution_scope": {"submit_authorized": False},
            }
        },
    }
    out = prepare_llm_request(payload)
    assert "module_execution_scope" not in out["current_proposal"]["parameters"]
    assert "module_execution_scope" in payload["current_proposal"]["parameters"]


def test_real_module_preview_and_validator_prepare_only_fake_fixture(tmp_path):
    from tests.test_relax_mc_batch import fixture

    manager, s, config, context, _ = fixture(tmp_path, count=2)
    s["scientific_decision_policy"] = {"schema_version": 1}
    a = {
        "tool": "generate_branches",
        "task_key": "branch-fixture",
        "parameters": {
            "generation_cost_preview": {"selected_branch_upper": 2, "relax_mc_cost_upper": 1000000}
        },
    }
    a = legacy_scope(a, s, config)
    registered = [
        {"branch_id": bid, "structure_id": sid}
        for bid, b in manager.data["branches"].items()
        for sid in b["structure_ids"]
    ]
    out = prepare_generated_module(
        a,
        {
            "status": "completed",
            "state": s,
            "registered": registered,
            "summary": {"registered_structures": len(registered)},
        },
        context,
    )
    assert out["module_preparation"]["status"] == "prepared", out.get("reason")
    assert out["submitted"] is False
    assert len(out["state"]["relax_mc_batches"]) == 1
    assert all(t["scientific_parent_task_key"] == "branch-fixture" for t in out["state"]["tasks"])


def test_three_point_policy_survives_unified_workflow_migration():
    from phase_agent.tools.policy.workflow_policy import normalize_workflow_state

    s = state()
    s["search_mode"] = "Sequential"
    out = normalize_workflow_state(s)
    assert "search_mode" not in out
    assert out["scientific_decision_policy"] == s["scientific_decision_policy"]
    assert out["workflow_version"] == "unified-v1"


def test_post_training_decision_uses_same_contract_and_keeps_approval_boundary():
    from phase_agent.tools.local.training_agent_review import review_training_choice

    s = state()
    s["candidate_models"] = {"new": {"validation": {}}}
    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "evidence_type": "grouped_cross_validation_review",
    }
    calls = []

    def model(payload):
        calls.append(payload)
        return {
            "tool": "adjust_strategy",
            "parameters": {"choice": "activate", "alternatives": "Defer more DFT"},
            "reason": "Use returned candidate evidence",
            "scientific_decision": review("round_direction"),
        }

    out, waits = review_training_choice(s, [handoff], model)
    assert calls == []
    assert out["active_model_version"] == "m1"
    assert out["candidate_models"]["new"]["agent_review"]["choice"] == "activate"
    assert waits[0]["stage"] == "awaiting_activation_approval"


def test_agent_catalog_separates_scientific_choices_from_module_maintenance():
    from phase_agent.tools.dispatch.scientific_operations import with_operation_contracts

    payload = {
        "allowed_tools": ["generate_branches", "allocate_mc_bohb", "prepare_local_batch_files"],
        "decision_context": {"scientific_decision_brief": decision_brief(state())},
    }
    out = with_operation_contracts(payload)
    assert [r["operation"] for r in out["scientific_operations"]] == ["generate_branches"]
    assert any(r["operation"] == "allocate_mlip_mc" for r in out["module_operations"])
    assert len(out["scientific_decision_points"]) == 3


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1, "30"])
def test_malformed_module_cost_envelope_never_prepares(value):
    a, r, c = module_case()
    a["parameters"]["module_execution_scope"]["relative_cost_upper"] = value
    with patch("phase_agent.tools.relax_mc_batch.prepare.prepare_batch") as prepare:
        out = prepare_generated_module(a, r, c)
    prepare.assert_not_called()
    assert out["module_preparation"]["status"] == "blocked"

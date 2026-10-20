from copy import deepcopy

import pytest

from phase_agent.analysis.state.decision_evidence_catalog import decision_evidence_catalog
from phase_agent.decisions.agent.branch_pool_review import review_errors
from phase_agent.decisions.agent.post_dft_review import request_validated_action
from phase_agent.decisions.agent.deepagents_proposal import proposal_material
from phase_agent.decisions.agent.summary_request import request_summary_first
from phase_agent.runtime.concise_dialogue import concise_body
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
from tests.test_on_demand_branch import context, review
from tests.test_model_refresh_workflow import recovered_state


def test_branch_pool_has_versioned_citable_source():
    facts = {"active_model_version": "new", "existing_branches": [{"branch_id": "b"}]}
    catalog = decision_evidence_catalog({"branch_pool_facts": facts})
    row = next(row for row in catalog["references"].values() if row["kind"] == "branch_pool")
    assert row["source_ref"] == "decision_context.branch_pool_facts"
    assert row["model_version"] == "new"
    assert catalog != decision_evidence_catalog(
        {"branch_pool_facts": {**facts, "existing_branches": []}}
    )


def test_branch_reference_contract_survives_initial_and_correction_transports():
    calls = []
    valid = review()
    valid.update(
        task_key="proposal", target_ids=[], budget=0, reason="审阅", expected_purpose="审阅"
    )
    valid["parameters"].update(sampling_mode="region_adaptive", defer_reason="暂缓")
    bad = deepcopy(valid)
    bad["branch_pool_review"]["evidence_refs"] = ["decision_context.branch_pool_facts"]
    bad["branch_pool_review"]["existing_branch_groups"][0]["evidence_refs"] = ["fake"]

    def client(payload):
        calls.append(payload)
        for transmitted in (payload, proposal_material(payload)):
            contract = transmitted["output_contracts"]["branch_review_evidence"]
            assert set(contract["references"]) == {"current"}
        if len(calls) == 1:
            return deepcopy(bad)
        assert any("fake" in item for item in payload["validation_errors"])
        assert any(
            "decision_context.branch_pool_facts" in item for item in payload["validation_errors"]
        )
        return deepcopy(valid)

    before = deepcopy(valid)

    def transport(payload):
        def provider(request):
            if (request.get("evidence_reader") or {}).get("phase") == "select_topics":
                return {
                    "kind": "inspect_evidence",
                    "topics": list(request["evidence_reader"]["topics"]),
                }
            return client(request)

        return request_summary_first(provider, payload)

    from tests.test_token_round_efficiency import payload as summary_fixture

    payload = summary_fixture()
    payload.update(
        allowed_tools=["generate_branches"],
        decision_context={**payload["decision_context"], **context()},
    )
    result = request_validated_action(transport, payload)
    assert not review_errors(result, context())
    assert len(calls) == 2 and valid == before


def changed_supplement(tmp_path):
    state, session, ctx, registry = recovered_state(tmp_path)
    session["confirmed_snapshot"]["config_version"] = "new-config"
    ctx["config_version"] = "new-config"
    ctx["effective_config"]["mlip"]["relax_parameters"] = {
        "relax_steps": 300,
        "optimizer": "LBFGS",
        "fmax": 0.05,
    }
    session["confirmed_snapshot"]["config"]["mlip"]["relax_parameters"] = deepcopy(
        ctx["effective_config"]["mlip"]["relax_parameters"]
    )
    return state, session, ctx, registry


def test_config_changed_supplement_is_reviewable_and_uses_current_settings(tmp_path):
    state, session, ctx, registry = changed_supplement(tmp_path)
    before = deepcopy(state)
    proposed = run_model_refresh_gate(
        state, session, registry=registry, context=ctx, human_feedback="同意"
    )
    assert proposed["status"] == "awaiting_approval", proposed.get("reason")
    assert proposed["state"]["tasks"] == before["tasks"] and state == before
    assert (
        proposed["action"]["parameters"]["supplemental_relax_parameters"]
        == ctx["effective_config"]["mlip"]["relax_parameters"]
    )
    prepared = run_model_refresh_gate(
        proposed["state"], session, registry=registry, context=ctx, human_feedback="同意"
    )
    assert prepared["status"] == "prepared", prepared.get("reason")
    task = prepared["state"]["tasks"][-1]
    assert task["parameters"]["relax_steps"] == 300 and task["parameters"]["optimizer"] == "LBFGS"
    assert prepared["state"]["model_refresh"]["plan"] == before["model_refresh"]["plan"]


def test_another_config_change_invalidates_supplement_approval(tmp_path):
    state, session, ctx, registry = changed_supplement(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=ctx)
    session["confirmed_snapshot"]["config_version"] = "another-config"
    ctx["config_version"] = "another-config"
    result = run_model_refresh_gate(
        first["state"], session, registry=registry, context=ctx, human_feedback="同意"
    )
    assert result["status"] == "awaiting_approval", result.get("reason")
    assert result["action"]["task_key"] != first["action"]["task_key"]
    assert len(result["state"]["tasks"]) == 1


def test_supplement_input_boundary_rejects_tampered_settings(tmp_path):
    from phase_agent.tools.local.prepare_model_refresh import prepare_model_refresh
    from phase_agent.tools.workflows.model_refresh_state import fingerprint

    state, session, ctx, registry = changed_supplement(tmp_path)
    proposed = run_model_refresh_gate(state, session, registry=registry, context=ctx)
    current = proposed["state"]
    action = deepcopy(proposed["action"])
    current["model_refresh"]["supplemental_input_authorization"] = {
        "approved_by": action["task_key"],
        "config_version": ctx["config_version"],
        "checksum": fingerprint(current["model_refresh"]["supplemental_candidates"]),
        "relax_parameters_checksum": fingerprint(
            action["parameters"]["supplemental_relax_parameters"]
        ),
    }
    action["parameters"]["supplemental_relax_parameters"]["relax_steps"] = 150
    result = prepare_model_refresh(
        action=action,
        context={**ctx, "event_state": current, "approval_record_id": action["task_key"]},
    )
    assert result["status"] == "rejected"
    assert result["state"]["tasks"] == current["tasks"]


def test_invalid_branch_citations_remain_rejected_after_one_correction():
    from phase_agent.decisions.agent.branch_review_repair import repair_branch_review

    invalid = review()
    invalid["branch_pool_review"]["evidence_refs"] = ["fake"]
    with pytest.raises(ValueError, match="invalid_after_one_correction.*fake"):
        repair_branch_review(
            lambda payload: deepcopy(invalid), {"decision_context": context()}, invalid
        )


@pytest.mark.parametrize("fault", ["model", "wave", "scope"])
def test_config_reapproval_cannot_bypass_other_scope_errors(tmp_path, fault):
    state, session, ctx, registry = changed_supplement(tmp_path)
    if fault == "model":
        state["active_model"]["model_path"] = "other.model"
    elif fault == "wave":
        state["model_refresh"]["approved_scope"]["remaining_waves"] = 0
    else:
        state["model_refresh"]["approved_scope"]["eligible_structure_ids"] = []
    result = run_model_refresh_gate(
        state, session, registry=registry, context=ctx, human_feedback="同意"
    )
    assert result["status"] in {"confirmation_required", "rejected"}, result
    assert len(result["state"]["tasks"]) == 1


def test_failed_new_proposal_does_not_claim_it_was_saved():
    text = concise_body(
        "本轮未执行：workflow 未能准备（branch_review_invalid_after_one_correction）。",
        {"pending_execution_policies": {"old": {}}},
    )
    assert "已保存输入或计算方案" not in text
    assert "未生成有效新方案" in text


def test_cached_snapshot_reindexes_current_branch_pool_without_mutating_history():
    from phase_agent.tools.state.state_manager import agent_state_summary

    state = {
        "active_model_version": "new",
        "current_state_snapshot": {
            "decision_context": {"evidence_catalog": {"references": {"old": {}}}}
        },
        "branch_candidates": [{"branch_id": "b", "P": "O3", "x": "1/2"}],
    }
    before = deepcopy(state)
    summary = agent_state_summary(state)
    refs = summary["decision_context"]["evidence_catalog"]["references"]
    assert any(row["kind"] == "branch_pool" and row["branch_count"] == 1 for row in refs.values())
    assert state == before


def test_supplement_review_displays_only_current_batch_and_settings(tmp_path):
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    from phase_agent.runtime.review_presentation import review_card
    from phase_agent.analysis.cost.estimate_proposal_cost import estimate_proposal_cost
    from phase_agent.analysis.cost.estimate_proposal_runtime import estimate_proposal_runtime

    state, session, ctx, registry = changed_supplement(tmp_path)
    proposed = run_model_refresh_gate(state, session, registry=registry, context=ctx)
    action = deepcopy(proposed["action"])
    # Original first-wave scope is larger and has a different historical runtime.
    action["parameters"]["refresh_preview"]["candidates"] *= 10
    action["parameters"]["refresh_preview"]["runtime_budget"] = {"serial_seconds": 99999}
    proposal = deepcopy(proposed["agent_proposal"])
    proposal["raw_action"] = action
    card = review_card("p", {"agent_proposal": proposal})
    assert card["scope"] == "追加弛豫 1 个单点候选；首批结果保留"
    cost = estimate_proposal_cost(action, state)
    assert cost["workload"][0]["tasks"] == 1
    assert cost["workload"][0]["maximum_supplemental_tasks"] == 0
    assert estimate_proposal_runtime(action, state)["serial_seconds"] is None
    text = format_workflow_reply({**proposed, "agent_proposal": proposal}, tmp_path / "state.json")
    assert "300 步，LBFGS" in text and "1 个单点候选" in text
    assert "首批：弛豫" not in text

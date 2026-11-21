"""Small synthetic contexts; no project model, server or scientific job."""

from copy import deepcopy
import json

import pytest

from phase_agent.evidence.snapshot import evidence_snapshot, role_context
from tests.test_codex_native_team import make_control, team, advice


def context_payload():
    common = {
        "evidence_catalog": {"references": {"e1": {"kind": "synthetic"}}},
        "memory_context": {"approved_knowledge": [], "authority": "reviewed advice only"},
        "human_constraints": ["do not retry DFT"],
        "execution_authorization_policy": {"source": "none"},
        "budget_limits": {"total": 50},
        "partial_recovery": {"retryable_task_ids": ["t1"], "missing": ["t2"]},
        "reliability_extrapolation_rounds": {"rounds": [{"direct_sampling_required": True}]},
        "unknown_future_risk": {"blocked": True},
        "qbc_candidates": [{"candidate_id": "c1", "D": 0.1, "H": 0.2}],
        "branch_pool_facts": {"existing_branches": [{"branch_id": "b1"}]},
        "generation_region_contract": "generation protocol " * 100,
        "dft_selection_policy": {
            "reference_framework": "DFT protocol " * 100,
            "max_tasks_per_round": 5,
        },
    }
    return {
        "state": {
            "current_project": {"project_path": "synthetic"},
            "confirmed_config_version": "cfg1",
            "active_model_version": "model1",
            "tasks": [{"task_id": "t1", "status": "failed", "input_sha256": "hash"}],
            "decision_context": deepcopy(common),
        },
        "decision_context": deepcopy(common),
        "allowed_tools": ["prepare_local_batch_files"],
        "user_instruction": "prepare only",
    }


def test_projection_keeps_constraints_risks_versions_and_all_candidate_metrics():
    payload = context_payload()
    original = deepcopy(payload)
    snapshot = evidence_snapshot(payload)
    for role in ("exploration", "reliability"):
        view = role_context(snapshot, role)["evidence_snapshot"]
        facts = view["facts"]
        assert view["snapshot_id"] == snapshot["snapshot_id"]
        assert view["config_version"] == "cfg1"
        assert view["model_version"] == "model1"
        assert facts["decision_context"]["unknown_future_risk"]["blocked"]
        assert facts["decision_context"]["human_constraints"] == ["do not retry DFT"]
        assert (
            facts["decision_context"]["qbc_candidates"]
            == payload["decision_context"]["qbc_candidates"]
        )
        assert "decision_context" not in facts["state"]
        assert view["context_projection"]["source_snapshot_id"] == snapshot["snapshot_id"]
        assert len(json.dumps(view)) < len(json.dumps(snapshot))
    assert payload == original
    assert (
        "generation_region_contract"
        not in role_context(snapshot, "reliability")["evidence_snapshot"]["facts"][
            "decision_context"
        ]
    )
    policy = role_context(snapshot, "exploration")["evidence_snapshot"]["facts"][
        "decision_context"
    ]["dft_selection_policy"]
    assert policy == {"max_tasks_per_round": 5}


def test_conflicting_nested_facts_are_not_silently_deduplicated():
    payload = context_payload()
    payload["state"]["decision_context"]["budget_limits"] = {"total": 1}
    view = role_context(evidence_snapshot(payload), "exploration")["evidence_snapshot"]
    assert view["facts"]["state"]["decision_context"]["budget_limits"] == {"total": 1}
    assert view["facts"]["decision_context"]["budget_limits"] == {"total": 50}


def test_native_response_is_self_contained_without_duplicate_top_snapshot(tmp_path):
    control, context = make_control(tmp_path)
    context.update({k: v for k, v in context_payload().items() if k != "allowed_tools"})
    t = team(control)
    view = t.start(question="review")
    assert "facts" not in view["snapshot"]
    for role, request in view["requests"].items():
        assert request["consultation_binding"] == {
            "team_id": view["team_id"],
            "revision": 1,
            "snapshot_id": view["snapshot"]["snapshot_id"],
        }
        assert request["evidence_snapshot"]["facts"]["user_instruction"]
    full = t.read(request_role="strategic")
    assert full["snapshot"]["facts"]
    assert full["requests"] == {}
    first = t.advise(view["team_id"], 1, advice(view))
    assert set(first["requests"]) == {"reliability"}
    assert all("advice" not in event for event in first["events"])
    assert t._load()["events"][-1]["advice"]
    revised = t.feedback(view["team_id"], 1, "revise", "new question")
    with pytest.raises(ValueError, match="snapshot"):
        t.advise(view["team_id"], revised["revision"], advice(view, "reliability"))
    assert t.read(include_history=True)["events"][0]["advice"]


def test_native_task_pack_reads_current_scientific_facts_and_binds_them(tmp_path):
    control, context = make_control(tmp_path)
    state = {
        "qbc_candidates": [{"candidate_id": "c1", "ehull": 0.01}],
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
    }
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    t = team(control)
    view = t.start(question="review full evidence")
    facts = view["requests"]["reliability"]["evidence_snapshot"]["facts"]
    assert facts["decision_context"]["qbc_candidates"][0]["candidate_id"] == "c1"
    state["qbc_candidates"][0]["ehull"] = 0.2
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    assert t.read()["stale"] is True


def test_simulation_paths_share_fact_contract_and_operation_scope(tmp_path):
    from phase_agent.agents.simulation.agent import prepare_proposal
    from tests.test_simulation_agent import SimulationModel
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["prepare_local_batch_files"]
    context["scientific_operations"] = operation_catalog(["prepare_local_batch_files"])
    state = {
        "current_status": "awaiting_inputs",
        "confirmed_config_version": "c1",
        "active_model_version": "m1",
        "budget_remaining": 12,
        "tasks": [{"task_id": "t1", "status": "failed", "input_sha256": "hash"}],
        "human_constraints": ["no retry"],
    }
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    t = team(control)
    view = t.start(question="prepare only", consult=['exploration', 'reliability'])
    from tests.test_codex_native_team import register_opinions
    register_opinions(t, view, ['exploration', 'reliability'])
    action = {
        "tool": "prepare_local_batch_files",
        "parameters": {"mode": "relax_mc_results", "batch_id": "b"},
    }
    draft = t.draft(view["team_id"], 1, action, rationale="inspect")
    native = draft["requests"]["simulation"]
    model = SimulationModel()
    prepare_proposal(
        {"raw_action": action},
        model,
        state=state,
        config={"budgets": {"total": 20}},
        registry={"prepare_local_batch_files": {"handler": lambda **kw: None}},
    )
    ordinary = next(r for r in model.calls if r["mode"] == "simulation_plan")
    for request in (native, ordinary):
        facts = request["facts"]
        assert facts["snapshot_id"]
        assert facts["model_version"] == "m1"
        assert facts["facts"]["state"]["tasks"][0]["input_sha256"] == "hash"
        assert facts["facts"]["state"]["human_constraints"] == ["no retry"]
        assert facts["facts"]["decision_context"]["memory_context"]["authority"]
        assert facts["context_projection"]["role"] == "simulation"
        assert len(request["registered_operations"]) == 1
        assert request["registered_operations"][0]["parameters"]["mode"] == "relax_mc_results"
        assert request["allowed_tools"] == []


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0)])
def test_deduplication_preserves_json_types(left, right):
    payload = context_payload()
    payload["state"]["decision_context"]["unknown_future_risk"] = left
    payload["decision_context"]["unknown_future_risk"] = right
    view = role_context(evidence_snapshot(payload), "reliability")["evidence_snapshot"]
    assert type(view["facts"]["state"]["decision_context"]["unknown_future_risk"]) is type(left)
    assert type(view["facts"]["decision_context"]["unknown_future_risk"]) is type(right)


def test_native_context_read_rejects_concurrent_source_change(tmp_path):
    control, _ = make_control(tmp_path)
    original = control.codex_context

    def changed():
        context = original()
        control.state_path.write_text('{"external_update":true}', encoding="utf-8")
        return context

    control.codex_context = changed
    with pytest.raises(ValueError, match="sources_changed_during_read"):
        team(control).start(question="review")
    assert not (tmp_path / "codex_team.json").exists()


def test_mcp_read_selects_a_single_task_and_preserves_full_history(tmp_path):
    from phase_agent.runtime.codex.team import call_team_tool

    control, _ = make_control(tmp_path)
    t = team(control)
    view = t.start(question="review")
    response = call_team_tool(
        control, "phase_team_read", {"request_role": "reliability", "include_history": True}
    )
    assert set(response["requests"]) == {"reliability"}
    assert response["snapshot"]["snapshot_id"] == view["snapshot"]["snapshot_id"]


def test_empty_current_pool_cannot_resurrect_cached_candidates(tmp_path):
    control, context = make_control(tmp_path)
    context["decision_context"]["qbc_candidates"] = [{"candidate_id": "old"}]
    control.state_path.write_text('{"qbc_candidates":[]}', encoding="utf-8")
    view = team(control).start(question="review")
    assert (
        view["requests"]["reliability"]["evidence_snapshot"]["facts"]["decision_context"][
            "qbc_candidates"
        ]
        == []
    )


def test_normal_team_uses_projection_without_losing_final_proposal_facts():
    from tests.test_decision_team import FakeTeam
    from phase_agent.graphs.decision_team.graph import request_with_langgraph

    payload = context_payload()
    payload["allowed_tools"] = ["pause_search"]
    payload["mode"] = "autonomous_search"
    model = FakeTeam()
    request_with_langgraph(model, payload)
    for request in model.calls[:3]:
        assert request["evidence_snapshot"]["context_projection"]
    final = model.calls[-1]
    assert final["state"] == payload["state"]
    assert final["decision_context"]["unknown_future_risk"]["blocked"]


def test_simulation_keeps_action_readiness_limits_and_cross_stage_risks():
    from phase_agent.evidence.snapshot import simulation_facts
    from phase_agent.agents.simulation.agent import planning_request
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog

    action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_mc_results"}}
    state = {
        "tasks": [{"task_id": "t", "status": "completed", "accepted": False}],
        "budget_remaining": 10,
        "pending_execution_policies": {"blocked": {"status": "awaiting_approval"}},
    }
    context = {
        "qbc_candidates": [{"candidate_id": "unrelated"}],
        "partial_recovery": {"missing": ["t"]},
        "dft_failure_guard": {"no_retry": True},
        "unknown_risk": {"blocked": True},
        "memory_context": {"approved_knowledge": [], "authority": "advice only"},
    }
    facts = simulation_facts(
        action, state=state, config={"budgets": {"total": 20}}, context=context
    )
    assert facts["facts"]["state"]["tasks"] == state["tasks"]
    assert (
        facts["facts"]["state"]["pending_execution_policies"] == state["pending_execution_policies"]
    )
    shared = facts["facts"]["decision_context"]
    assert shared["budget_limits"] == {"total": 20}
    assert shared["unknown_risk"]["blocked"]
    assert shared["dft_failure_guard"]["no_retry"]
    assert shared["partial_recovery"]["missing"] == ["t"]
    assert "qbc_candidates" not in shared
    request = planning_request(
        action, facts=facts, operations=operation_catalog(["prepare_local_batch_files"])
    )
    assert request["registered_operations"]
    assert all(
        row["parameters"]["mode"] == "relax_mc_results" for row in request["registered_operations"]
    )


def test_native_tables_preserve_complete_rows_and_reference_binding(tmp_path):
    from phase_agent.models.transport_tables import expand_evidence_tables

    control, context = make_control(tmp_path)
    candidates = [
        {
            "candidate_id": f"c{i}",
            "model_version": "m1",
            "ehull": i / 1000,
            "D": i / 10,
            "H": i / 20,
            "phase": "O3",
            "limitations": ["same-frame only"],
        }
        for i in range(8)
    ]
    state = {
        "qbc_candidates": candidates,
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
    }
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    t = team(control)
    view = t.start(question="review")
    request = view["requests"]["reliability"]
    assert request["evidence_encoding"]
    restored = expand_evidence_tables(request["evidence_snapshot"])
    assert restored["facts"]["decision_context"]["qbc_candidates"] == candidates
    authoritative = t._snapshot(t._context(), t._load())
    assert restored == role_context(authoritative, "reliability")["evidence_snapshot"]
    assert restored["snapshot_id"] == view["snapshot"]["snapshot_id"]


def test_reviewed_memory_alias_keeps_one_exact_value_and_no_mutation():
    payload = context_payload()
    memory = {
        "approved_knowledge": [
            {"knowledge_id": "k", "statement": "advice", "approved_by_user": True}
        ],
        "short_term": {"last_observation": "incomplete"},
    }
    context = payload["decision_context"]
    context["memory_context"] = memory
    context["relevant_approved_knowledge"] = deepcopy(memory["approved_knowledge"])
    context["short_term_memory"] = deepcopy(memory["short_term"])
    original = deepcopy(payload)
    for role in ("exploration", "reliability", "strategic"):
        result = role_context(evidence_snapshot(payload), role)["evidence_snapshot"]
        projected = result["facts"]["decision_context"]
        assert projected["memory_context"] == memory
        assert "relevant_approved_knowledge" not in projected
        assert "short_term_memory" not in projected
        assert (
            result["context_projection"]["duplicate_fields"][
                "decision_context.relevant_approved_knowledge"
            ]
            == "decision_context.memory_context.approved_knowledge"
        )
    assert payload == original


def test_old_simulation_opinion_cannot_cross_revision_with_same_action(tmp_path):
    from tests.test_simulation_agent import SimulationModel

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["prepare_local_batch_files"]
    t = team(control)
    view = t.start(question="prepare", consult=['exploration'])
    from tests.test_codex_native_team import register_opinions
    register_opinions(t, view, ['exploration'])
    action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_inputs"}}
    draft = t.draft(view["team_id"], 1, action, rationale="review")
    old_plan = SimulationModel()(draft["requests"]["simulation"])
    revised = t.feedback(view["team_id"], 1, "revise", "new readiness constraint")
    register_opinions(t, revised, ['exploration'])
    t.draft(view["team_id"], revised["revision"], action, rationale="review again")
    with pytest.raises(ValueError, match="context"):
        t.simulate(view["team_id"], revised["revision"], old_plan)
    assert t._load()["status"] == "planning_simulation"


def test_new_ordinary_simulation_requires_echoed_context_binding():
    from phase_agent.agents.simulation.agent import prepare_proposal
    from tests.test_simulation_agent import SimulationModel

    class Unbound(SimulationModel):
        def __call__(self, request):
            result = super().__call__(request)
            result.pop("context_binding", None)
            return result

    with pytest.raises(ValueError, match="context"):
        prepare_proposal(
            {"raw_action": {"tool": "check_convergence"}},
            Unbound(),
            state={},
            config={},
            registry={"check_convergence": {"handler": lambda: None}},
        )


def test_role_operation_contracts_follow_existing_permissions():
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog
    from phase_agent.agents.permissions import ADVISORY_OPERATIONS

    payload = context_payload()
    payload["scientific_operations"] = operation_catalog(["prepare_local_batch_files"])
    source = evidence_snapshot(payload)
    for role in ("exploration", "reliability"):
        view = role_context(source, role)["evidence_snapshot"]
        visible = {row["operation"] for row in view["facts"]["scientific_operations"]}
        original = {row["operation"] for row in source["facts"]["scientific_operations"]}
        assert visible == original & ADVISORY_OPERATIONS[role]
        assert view["context_projection"]["omitted_operation_contracts"]
        assert (
            view["facts"]["decision_context"]["qbc_candidates"]
            == payload["decision_context"]["qbc_candidates"]
        )


def test_missing_config_budget_does_not_erase_context_resource_limit():
    from phase_agent.evidence.snapshot import simulation_facts

    facts = simulation_facts(
        {"tool": "check_convergence"},
        state={},
        config={},
        context={"budget_limits": {"total": 30}, "resource_guard": {"blocked": True}},
    )
    assert facts["facts"]["decision_context"]["budget_limits"] == {"total": 30}
    assert facts["facts"]["decision_context"]["resource_guard"]["blocked"]

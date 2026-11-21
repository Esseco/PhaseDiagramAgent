"""Decision-flow boundaries using tiny projects and fake experts only."""

from copy import deepcopy
import json

import pytest

from tests.test_codex_native_team import make_control, team, advice
from tests.test_decision_team import FakeTeam, payload
from phase_agent.agents.contracts import ConsultationPlan, validate_advice
from phase_agent.evidence.snapshot import evidence_snapshot, role_context
from phase_agent.graphs.decision_team.graph import request_with_langgraph
from phase_agent.runtime.codex.team import call_team_tool
from phase_agent.agents.simulation.agent import planning_request, inspect_execution


def tasks():
    return {
        "exploration": {
            "question": "Which coverage gaps remain?",
            "completion_criteria": ["Identify missing regions"],
        },
        "reliability": {
            "question": "Which stability claims are unsupported?",
            "completion_criteria": ["Identify failure evidence"],
        },
    }


def test_native_roles_share_constraints_but_receive_distinct_tasks(tmp_path):
    control, context = make_control(tmp_path)
    context["decision_context"]["human_constraints"] = ["no new computation"]
    view = call_team_tool(
        control,
        "phase_team_start",
        {
            "question": "Choose next route",
            "role_tasks": tasks(),
        },
    )
    for role, request in view["requests"].items():
        assert request["decision_goal"] == "Choose next route"
        assert request["question"] == tasks()[role]["question"]
        assert request["completion_criteria"] == tasks()[role]["completion_criteria"]
        assert request["allowed_tools"] == []
        assert request["evidence_snapshot"]["snapshot_id"] == view["snapshot"]["snapshot_id"]
        assert "no new computation" in str(request["evidence_snapshot"])
        assert "previous_opinion" not in request
    assert control.state_path.read_text() == "{}"


def test_ordinary_roles_use_same_task_contract():
    class Client(FakeTeam):
        def __call__(self, request):
            result = super().__call__(request)
            if request.get("mode") == "team_consultation":
                result["role_tasks"] = tasks()
            return result

    client = Client()
    request_with_langgraph(client, payload())
    experts = [r for r in client.calls if r.get("mode") == "expert_assessment"]
    assert len(experts) == 2
    for request in experts:
        assert request["question"] == tasks()[request["agent_role"]]["question"]
        assert request["decision_goal"] == "What evidence is still missing?"
        assert "previous_opinion" not in request


def test_unrequested_role_task_rejected():
    with pytest.raises(ValueError, match="unrequested"):
        ConsultationPlan(snapshot_id="s", consult=["exploration"], question="q", role_tasks=tasks())


def test_guidance_explicit_supersession_preserves_original_and_source(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="Choose route", consult=[])
    v = t.feedback(v["team_id"], 1, "revise", "Prefer broad search")
    g1 = v["effective_guidance"][0]
    v = call_team_tool(
        control,
        "phase_team_feedback",
        {
            "team_id": v["team_id"],
            "revision": 2,
            "decision": "revise",
            "feedback": "Prefer deeper search",
            "supersedes": [g1["guidance_id"]],
        },
    )
    assert v["guidance"] == ["Prefer deeper search"]
    assert v["effective_guidance"][0]["supersedes"] == ["g1"]
    assert v["effective_guidance"][0]["scope"] == v["team_id"]
    assert v["effective_guidance"][0]["source"] == "human_revise"
    facts = v["snapshot"]["facts"]
    assert "Prefer broad search" not in json.dumps(facts)
    stored = t._load()
    assert stored["guidance"] == ["Prefer broad search", "Prefer deeper search"]
    history = t.read(include_history=True)
    assert [r["text"] for r in history["guidance_history"]] == stored["guidance"]
    assert len([e for e in history["events"] if e["kind"] == "human_revise"]) == 2
    assert "direction_review" not in stored and stored["advice"] == []


def test_legacy_guidance_is_active_until_explicitly_superseded(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=[])
    s = t._load()
    s["guidance"] = ["legacy original"]
    t._save(s)
    v = t.read()
    assert v["effective_guidance"][0]["source"] == "legacy_guidance"
    assert v["effective_guidance"][0]["revision"] is None
    with pytest.raises(ValueError, match="current_guidance"):
        t.feedback(v["team_id"], 1, "revise", "new", supersedes=["invented"])
    assert t._load()["revision"] == 1
    v = t.feedback(v["team_id"], 1, "revise", "new", supersedes=["g1"])
    assert v["guidance"] == ["new"]
    assert t.read(include_history=True)["guidance_history"][0]["text"] == "legacy original"


def prepared_both(tmp_path):
    control, context = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q")
    for role in v["consult"]:
        t.advise(v["team_id"], 1, advice(v, role), compact=True)
    return control, context, t, v


def test_unaccepted_draft_edit_reuses_original_opinions_without_rebinding(tmp_path):
    _, _, t, v = prepared_both(tmp_path)
    original = deepcopy(t._load()["advice"])
    t.draft(v["team_id"], 1, {"tool": "pause_search"}, rationale="first")
    v2 = call_team_tool(
        t.control,
        "phase_team_draft",
        {
            "team_id": v["team_id"],
            "revision": 1,
            "action": {"tool": "pause_search"},
            "rationale": "clearer explanation",
            "replace_draft": True,
        },
    )
    assert v2["advice"] == original
    assert v2["revision"] == 1 and v2["snapshot"]["snapshot_id"] == v["snapshot"]["snapshot_id"]
    assert not v2["requests"] and not v2["execution_approved"]
    assert len([e for e in t._load()["events"] if e["kind"] == "draft"]) == 2
    assert len([e for e in t._load()["events"] if e["kind"] == "expert_advice"]) == 2
    t.feedback(v["team_id"], 1, "accept", "accept")
    with pytest.raises(ValueError, match="accepted"):
        t.draft(v["team_id"], 1, {"tool": "pause_search"}, rationale="edit", replace_draft=True)


@pytest.mark.parametrize(
    "change",
    [
        "model_version",
        "config_version",
        "context_digest",
        "budget",
        "input",
        "result",
        "goal",
        "permission",
    ],
)
def test_any_unproved_source_change_rejects_opinion_and_draft_reuse(tmp_path, change):
    control, context, t, v = prepared_both(tmp_path)
    t.draft(v["team_id"], 1, {"tool": "pause_search"}, rationale="first")
    if change in {"model_version", "config_version", "context_digest"}:
        context[change] += "-changed"
    elif change == "permission":
        context["allowed_tools"] = []
        context["context_digest"] = "permissions-changed"
    else:
        control.state_path.write_text(json.dumps({change: "changed"}), encoding="utf8")
    with pytest.raises(ValueError, match="stale"):
        t.draft(v["team_id"], 1, {"tool": "pause_search"}, rationale="edit", replace_draft=True)
    assert t._load()["revision"] == 1


def test_guidance_revision_invalidates_all_and_old_snapshot_cannot_answer(tmp_path):
    _, _, t, v = prepared_both(tmp_path)
    old = advice(v)
    v2 = t.feedback(v["team_id"], 1, "revise", "change goal")
    assert v2["advice"] == [] and v2["draft"] is None
    assert v2["snapshot"]["snapshot_id"] != v["snapshot"]["snapshot_id"]
    with pytest.raises(ValueError, match="revision"):
        t.advise(v["team_id"], 1, old)
    with pytest.raises(ValueError, match="snapshot"):
        t.advise(v["team_id"], 2, old)


def test_clarification_is_one_targeted_same_snapshot_round(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=["exploration"])
    opinion = advice(v)
    opinion.update(
        recommendation="insufficient_evidence",
        evidence_gaps=["Which covered region supports stopping?"],
    )
    t.advise(v["team_id"], 1, opinion)
    clarified = call_team_tool(
        control,
        "phase_team_clarify",
        {
            "team_id": v["team_id"],
            "revision": 1,
            "role": "exploration",
            "gap_index": 0,
        },
    )
    req = clarified["requests"]["exploration"]
    assert req["clarification"]["snapshot_id"] == v["snapshot"]["snapshot_id"]
    assert req["previous_opinion"]["recommendation"] == "insufficient_evidence"
    assert req["allowed_tools"] == [] and clarified["execution_approved"] is False
    with pytest.raises(ValueError, match="missing_expert"):
        t.draft(v["team_id"], 1, {"tool": "pause_search"}, rationale="pending")
    with pytest.raises(ValueError, match="limit"):
        t.clarify(v["team_id"], 1, "exploration", 0)
    followup = deepcopy(opinion)
    followup["assessment"] = "Still unknown; do not compute."
    t.advise(v["team_id"], 1, followup)
    assert not t.read()["requests"]
    with pytest.raises(ValueError, match="limit"):
        t.clarify(v["team_id"], 1, "exploration", 0)
    assert len([e for e in t._load()["events"] if e["kind"] == "expert_advice"]) == 2
    assert control.state_path.read_text() == "{}"


def test_new_facts_cannot_be_supplemented_via_same_version_clarification(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=["exploration"])
    a = advice(v)
    a.update(recommendation="insufficient_evidence", evidence_gaps=["new results needed"])
    t.advise(v["team_id"], 1, a)
    control.state_path.write_text('{"new_result":true}', encoding="utf8")
    with pytest.raises(ValueError, match="stale"):
        t.clarify(v["team_id"], 1, "exploration", 0)
    assert not t._load().get("clarification_counts")


def test_option_specific_refs_are_validated_and_types_preserved(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=["exploration"])
    snap = t._snapshot(t._context(), t._load())
    a = advice(v)
    a["options"][0].update(
        evidence_refs=["e1"],
        counterevidence_refs=["e1"],
        assumptions=["budget unchanged"],
        estimated_cost=1.0,
        cost_unit="GPU hour",
    )
    result = validate_advice(a, role="exploration", snapshot=snap)
    assert result["options"][0]["cost_unit"] == "GPU hour"
    a["options"][0]["counterevidence_refs"] = ["invented"]
    with pytest.raises(ValueError, match="unavailable evidence"):
        validate_advice(a, role="exploration", snapshot=snap)


def blocked_mc_snapshot():
    p = payload()
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog

    p["allowed_tools"] = ["allocate_mc_bohb"]
    p["scientific_operations"] = operation_catalog(p["allowed_tools"])
    p["decision_context"]["mc_entry_conditions"] = {"required": True, "ready": False}
    return evidence_snapshot(p)


def test_readiness_exposes_dependency_separately_from_authorization():
    s = blocked_mc_snapshot()
    readiness = role_context(s, "exploration")["operation_readiness"]["allocate_mlip_mc"]
    assert readiness["project_readiness"] == "waiting_dependency"
    assert readiness["authorization"].startswith("not_evaluated")
    assert readiness["input_readiness"] == "not_fully_verified"
    assert readiness["sources"] == ["decision_context.mc_entry_conditions"]
    with pytest.raises(ValueError, match="preflight_failed.*MC requires"):
        planning_request({"tool": "allocate_mc_bohb"}, facts=s)


def test_native_preflight_rejects_before_persisting_pseudo_plan(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    original = t._context

    def context(**kw):
        c = original(**kw)
        c["allowed_tools"] = ["allocate_mc_bohb"]
        c["team_decision_context"]["mc_entry_conditions"] = {"required": True, "ready": False}
        return c

    t._context = context
    v = t.start(question="q", consult=[])
    with pytest.raises(ValueError, match="coordinator_preflight_failed"):
        t.draft(v["team_id"], 1, {"tool": "allocate_mc_bohb"}, rationale="blocked")
    assert t._load()["draft"] is None
    assert t._load()["status"] == "consulting"


def test_retired_route_unavailable_even_via_mode_alias_but_not_frozen_validation():
    from phase_agent.tools.dispatch.scientific_operations import (
        operation_catalog,
        operation_readiness,
    )
    from phase_agent.agents.simulation.agent import validate_plan, action_hash

    p = {"scientific_operations": operation_catalog(["prepare_local_batch_files"])}
    r = operation_readiness(p)
    assert r["prepare_perturbation_recovery"]["project_readiness"] == "unsupported_new_proposal"
    a = {
        "tool": "prepare_local_batch_files",
        "parameters": {"mode": "perturbation_recovery_inputs"},
    }
    with pytest.raises(ValueError, match="parent_first"):
        planning_request(a)
    # Historical validation is structural; it does not apply the new-proposal policy.
    plan = {
        "action_hash": action_hash(a),
        "rationale": "historical",
        "steps": [
            {"kind": "preflight", "description": "old"},
            {"kind": "invoke", "description": "old", "tool": a["tool"]},
            {"kind": "verify", "description": "old"},
        ],
        "checks": ["old check"],
        "failure_policy": "return_to_strategic",
    }
    assert validate_plan(plan, a)["checks"] == ["old check"]


def test_simulation_planned_checks_not_claimed_as_executed():
    request = planning_request({"tool": "pause_search"})
    assert request["program_preflight"]["status"] == "passed"
    assert request["execution_checks"]["status"] == "not_executed"
    a = {"tool": "pause_search", "_simulation_plan": {"checks": ["Check artifact checksum"]}}
    report = inspect_execution(a, {"status": "completed", "result": {"status": "prepared"}})
    assert report["planned_checks"]["status"] == "not_evaluated"
    assert not report["scientific_completion_claimed"]
    failed = inspect_execution(
        a,
        {
            "status": "failed",
            "result": {
                "status": "blocked",
                "reason": "input checksum mismatch",
                "validation_errors": ["input_sha256"],
            },
        },
    )
    assert failed["observations"]["execution.result.reason"] == "input checksum mismatch"
    assert failed["observations"]["execution.result.validation_errors"] == ["input_sha256"]
    assert failed["next"] == "return_to_strategic"


def test_revision_archives_old_role_tasks_and_does_not_reuse_them(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", role_tasks=tasks())
    v = t.feedback(v["team_id"], 1, "revise", "new direction")
    assert v["requests"]["exploration"]["question"] != tasks()["exploration"]["question"]
    old = [
        e for e in t.read(include_history=True)["events"] if e["kind"] == "role_tasks_superseded"
    ]
    assert old[0]["role_tasks"] == tasks()


def test_replacing_action_discards_plan_and_never_keeps_direction_review(tmp_path):
    from phase_agent.agents.simulation.agent import action_hash

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["pause_search", "prepare_local_batch_files"]
    t = team(control)
    v = t.start(question="q", consult=['exploration', 'reliability'])
    from tests.test_codex_native_team import register_opinions
    register_opinions(t, v, ['exploration', 'reliability'])
    action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_inputs"}}
    v = t.draft(v["team_id"], 1, action, rationale="original")
    plan = {
        "action_hash": action_hash(action),
        "context_binding": {
            "snapshot_id": v["snapshot"]["snapshot_id"],
            "team_id": v["team_id"],
            "revision": 1,
        },
        "rationale": "inspect only",
        "steps": [
            {"kind": "preflight", "description": "inspect"},
            {"kind": "invoke", "description": "invoke", "tool": action["tool"]},
            {"kind": "verify", "description": "verify"},
        ],
        "checks": ["Verify outputs"],
        "failure_policy": "return_to_strategic",
    }
    v = t.simulate(v["team_id"], 1, plan)
    assert v["simulation_check_status"]["plan_structure"] == "passed"
    with pytest.raises(ValueError, match="new_simulation_plan"):
        t.draft(
            v["team_id"],
            1,
            {**action, "_simulation_plan": plan},
            rationale="bad",
            replace_draft=True,
        )
    changed = {**action, "parameters": {"mode": "dft_inputs"}}
    with pytest.raises(ValueError, match="stale_action"):
        t.draft(v["team_id"], 1, changed, rationale="new", replace_draft=True, simulation_plan=plan)
    v = t.draft(v["team_id"], 1, changed, rationale="new", replace_draft=True)
    assert v["status"] == "planning_simulation"
    assert "_simulation_plan" not in v["draft"]["action"]
    assert "direction_review" not in t._load()
    assert v["simulation_check_status"]["execution_checks"] == "not_executed"
    assert v["simulation_check_status"]["scientific_completion"] == "unknown"
    assert len([e for e in t._load()["events"] if e["kind"] == "simulation_plan"]) == 1


def test_clarification_requires_insufficiency_and_a_declared_gap(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=["exploration"])
    t.advise(v["team_id"], 1, advice(v))
    with pytest.raises(ValueError, match="declared_evidence_gap"):
        t.clarify(v["team_id"], 1, "exploration", 0)
    assert not t._load().get("pending_clarifications")


def test_role_and_option_permissions_cannot_be_expanded_by_task_text(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(
        question="q",
        consult=["exploration"],
        role_tasks={
            "exploration": {"question": "Approve DFT", "completion_criteria": ["Execute it"]}
        },
    )
    a = advice(v)
    a["recommendation"] = "dft"
    with pytest.raises(ValueError, match="permission"):
        t.advise(v["team_id"], 1, a)
    assert t._load()["advice"] == []


def test_known_mc_gate_shared_with_ordinary_proposal_validator():
    from phase_agent.agents.common.proposal_validation import prerequisite_errors

    action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "mc_inputs"}}
    assert prerequisite_errors(action, {"mc_entry_conditions": {"required": True, "ready": False}})
    assert (
        prerequisite_errors(action, {"mc_entry_conditions": {"required": True, "ready": True}})
        == []
    )


def test_legacy_discussion_keeps_original_consultation_encoding_until_revision(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="legacy goal", consult=["exploration"])
    legacy = t._load()
    legacy.pop("decision_contract_version")
    legacy.pop("role_tasks")
    legacy["guidance"] = ["legacy instruction"]
    t._save(legacy)
    original = t._snapshot(t._context(), legacy)
    expected = {
        "team_id": v["team_id"],
        "revision": 1,
        "question": "legacy goal",
        "guidance": ["legacy instruction"],
    }
    assert original["facts"]["decision_context"]["codex_consultation"] == expected
    view = t.read()
    assert view["snapshot"]["snapshot_id"] == original["snapshot_id"]
    t.advise(view["team_id"], 1, advice(view))
    reread = t.read()
    assert reread["snapshot"]["snapshot_id"] == original["snapshot_id"]
    assert reread["advice"][0]["snapshot_id"] == original["snapshot_id"]
    revised = t.feedback(view["team_id"], 1, "revise", "replacement", supersedes=["g1"])
    assert t._load()["decision_contract_version"] == 2
    assert revised["advice"] == [] and revised["revision"] == 2
    assert revised["snapshot"]["snapshot_id"] != original["snapshot_id"]


def test_compact_ack_keeps_other_pending_clarification_visible(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q")
    a = advice(v)
    a.update(recommendation="insufficient_evidence", evidence_gaps=["coverage evidence"])
    t.advise(v["team_id"], 1, a)
    t.clarify(v["team_id"], 1, "exploration", 0)
    ack = t.advise(v["team_id"], 1, advice(v, "reliability"), compact=True)
    assert ack["remaining_roles"] == ["exploration"]


def test_draft_cannot_claim_program_validated_simulation_via_raw_metadata(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=[])
    with pytest.raises(ValueError, match="simulation"):
        t.draft(
            v["team_id"],
            1,
            {
                "tool": "pause_search",
                "_simulation_plan": {"checks": ["Everything passed"], "action_hash": "invented"},
            },
            rationale="unvalidated model assertion",
        )
    assert t._load()["draft"] is None


def test_simulation_display_marks_stale_and_uncertain_execution_unknown(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    v = t.start(question="q", consult=[])
    t.draft(v["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    s = t._load()
    s["status"] = "submission_uncertain"
    t._save(s)
    t._recorded_submission = lambda session: None
    read = t.read()
    assert read["simulation_check_status"]["execution_checks"] == "unknown"
    control.state_path.write_text('{"changed":true}', encoding="utf8")
    read = t.read()
    assert read["simulation_check_status"]["context_validity"] == "stale"

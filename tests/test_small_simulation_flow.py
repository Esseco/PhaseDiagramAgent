"""Small fake-only checks of consultation floors and deterministic Simulation."""

from copy import deepcopy
import json
import pytest

from phase_agent.agents.permissions import necessary_roles, missing_role_reasons
from phase_agent.agents.simulation.agent import (
    prepare_proposal,
    validate_frozen_plan,
    inspect_execution,
)
from phase_agent.agents.simulation.program_checks import dft_input_record
from phase_agent.graphs.decision_team.graph import request_with_langgraph
from tests.test_codex_native_team import make_control, team, advice
from tests.test_decision_team import FakeTeam, payload
from tests.test_simulation_agent import SimulationModel


def fixed():
    return (
        {
            "tool": "prepare_local_batch_files",
            "target_ids": ["T"],
            "parameters": {"mode": "dft_inputs"},
        },
        {
            "confirmed_config_version": "C",
            "active_model_version": "M",
            "tasks": [
                {
                    "task_id": "T",
                    "stage": "dft_relax",
                    "status": "pending",
                    "model_version": "M",
                    "structure_id": "S",
                }
            ],
        },
    )


@pytest.mark.parametrize(
    "action,roles",
    [
        ({"tool": "generate_branches"}, {"exploration"}),
        ({"tool": "select_dft_candidates"}, {"reliability"}),
        ({"tool": "update_mlip"}, {"reliability"}),
        ({"tool": "check_convergence"}, {"exploration", "reliability"}),
        ({"tool": "adjust_strategy"}, {"exploration", "reliability"}),
        ({"tool": "pause_search"}, set()),
        (
            {"tool": "prepare_local_batch_files", "parameters": {"mode": "unknown"}},
            {"exploration", "reliability"},
        ),
    ],
)
def test_role_floor(action, roles):
    assert set(necessary_roles(action)) == roles


def test_fixed_dft_is_not_selecting_new_candidates():
    action, state = fixed()
    assert necessary_roles(action, state=state) == {}
    assert set(necessary_roles({"tool": "select_dft_candidates"}, state=state)) == {"reliability"}
    assert (
        dft_input_record({**action, "parameters": {"mode": "dft_inputs", "new_scope": True}}, state)
        is None
    )


def test_current_advice_and_handoff_not_cross_version():
    action = {"tool": "generate_branches"}
    opinion = {
        "role": "exploration",
        "snapshot_id": "S",
        "recommendation": "explore",
        "handoff_roles": ["reliability"],
    }
    assert set(missing_role_reasons(action, [opinion], snapshot_id="S")) == {"reliability"}
    assert set(missing_role_reasons(action, [opinion], snapshot_id="new")) == {
        "exploration",
        "reliability",
    }
    opinion["recommendation"] = "insufficient_evidence"
    assert "exploration" in missing_role_reasons(action, [opinion], snapshot_id="S")


def test_native_missing_roles_return_real_tasks_before_draft(tmp_path):
    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["check_convergence", "pause_search"]
    t = team(control)
    started = t.start(question="Judge convergence", consult=[])
    view = t.draft(started["team_id"], 1, {"tool": "check_convergence"}, rationale="test")
    assert set(view["missing_roles"]) == {"exploration", "reliability"}
    assert set(view["requests"]) == {"exploration", "reliability"}
    assert not view.get("draft")
    for role in view["requests"]:
        t.advise(view["team_id"], 1, advice(view, role))
    reviewed = t.draft(view["team_id"], 1, {"tool": "check_convergence"}, rationale="test")
    assert reviewed["status"] == "awaiting_human_input"


def test_native_fixed_inputs_skip_model_task_but_keep_review(tmp_path):
    action, state = fixed()
    control, context = make_control(tmp_path)
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    context["allowed_tools"] = ["prepare_local_batch_files"]
    t = team(control)
    start = t.start(question="Prepare registered inputs", consult=[])
    view = t.draft(start["team_id"], 1, action, rationale="fixed list")
    assert view["status"] == "awaiting_human_input"
    assert "simulation" not in view["requests"]
    assert view["draft"]["action"]["_simulation_program"]["handler"] == "program"
    assert view["execution_approved"] is False


def test_program_isolated_zero_models_and_stale_scope_blocked():
    action, state = fixed()
    proposal = {"raw_action": action}
    original = deepcopy(proposal)
    model = SimulationModel()
    prepared = prepare_proposal(
        proposal, model, state=state, config={}, registry={"prepare_local_batch_files": {}}
    )
    assert proposal == original and not model.calls
    assert "_simulation_plan" not in prepared["raw_action"]
    validate_frozen_plan(prepared["raw_action"], state=state)
    state["tasks"][0]["structure_id"] = "changed"
    with pytest.raises(ValueError, match="simulation_program"):
        validate_frozen_plan(prepared["raw_action"], state=state)


def test_known_blocker_no_model_unknown_exception_diagnosed():
    model = SimulationModel()
    report = inspect_execution(
        {"tool": "prepare_local_batch_files"},
        {"status": "completed", "result": {"status": "blocked", "reason": "missing_manifest"}},
        model,
    )
    assert report["diagnosis_source"] == "program" and not model.calls
    assert report["next"] == "return_to_strategic" and not report["scientific_completion_claimed"]
    inspect_execution(
        {"tool": "prepare_local_batch_files"},
        {"status": "failed", "error": "unexplained contradiction"},
        model,
    )
    assert len(model.calls) == 1 and model.calls[0]["mode"] == "simulation_diagnosis"


def test_ordinary_missing_specialist_fails_closed():
    class Choice(FakeTeam):
        def __call__(self, request):
            value = super().__call__(request)
            if request["mode"] == "autonomous_search":
                return {"tool": "check_convergence"}
            return value

    data = payload()
    data["allowed_tools"] = ["check_convergence", "pause_search"]
    client = Choice()
    client.roles = []
    with pytest.raises(ValueError, match="missing_required_role"):
        request_with_langgraph(client, data)


def test_legacy_no_expert_capability_does_not_fake_consultation():
    data = payload()
    data["allowed_tools"] = ["check_convergence"]
    with pytest.raises(ValueError, match="missing_required_role"):
        request_with_langgraph(lambda _: {"tool": "check_convergence"}, data)


def test_program_approval_rejection_and_receipt_use_existing_chain():
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from tests.test_event_loop import _session

    action, state = fixed()
    session = _session()
    state["confirmed_config_version"] = session["confirmed_snapshot"]["config_version"]
    calls, model_calls = [], []

    def proposal(request):
        model_calls.append(request["mode"])
        return deepcopy(action)

    registry = create_tool_registry(
        {
            "prepare_local_batch_files": lambda **kw: (
                calls.append(kw) or {"status": "prepared", "submitted": False}
            )
        }
    )
    opts = dict(
        session=session,
        registry=registry,
        invocation_id="fixed-program",
        execution_mode="interactive",
        agent_client=proposal,
    )
    pending = run_tool_step(state, **opts)
    assert pending["status"] == "awaiting_approval", pending
    assert pending["agent_proposal"]["simulation_program"]["handler"] == "program"
    assert not calls and model_calls == ["autonomous_search"]
    completed = run_tool_step(pending["state"], human_feedback="approve", **opts)
    assert completed["execution"]["status"] == "completed", completed
    assert len(calls) == 1 and model_calls == ["autonomous_search"]
    assert completed["simulation_report"]["handling"] == "program"
    replay = run_tool_step(completed["state"], human_feedback="approve", **opts)
    assert len(calls) == 1 and model_calls == ["autonomous_search"], replay

    denied = run_tool_step(state, **{**opts, "invocation_id": "denied"})
    rejected = run_tool_step(
        denied["state"], human_feedback="reject", **{**opts, "invocation_id": "denied"}
    )
    assert not rejected.get("execution") and len(calls) == 1


def test_program_known_missing_input_does_not_request_explanation_model():
    action, state = fixed()
    state["tasks"] = []
    model = SimulationModel()
    with pytest.raises(ValueError, match="unregistered_task"):
        prepare_proposal(
            {"raw_action": action},
            model,
            state=state,
            config={},
            registry={"prepare_local_batch_files": {}},
        )
    assert not model.calls


def test_modern_client_missing_simulation_capability_is_blocked():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="capability_unavailable"):
        prepare_proposal(
            {"raw_action": {"tool": "check_convergence"}},
            SimpleNamespace(supports_expert_advice=True),
            state={},
            config={},
            registry={},
        )


@pytest.mark.parametrize("change", ["config", "model", "task_model", "task_status", "parameters"])
def test_program_source_and_parameter_changes_invalidate(change):
    action, state = fixed()
    prepared = prepare_proposal(
        {"raw_action": action},
        None,
        state=state,
        config={},
        registry={"prepare_local_batch_files": {}},
    )
    frozen = prepared["raw_action"]
    if change == "config":
        state["confirmed_config_version"] = "new"
    elif change == "model":
        state["active_model_version"] = "new"
    elif change == "task_model":
        state["tasks"][0]["model_version"] = "new"
    elif change == "task_status":
        state["tasks"][0]["status"] = "completed"
    else:
        frozen["parameters"]["extra_selection"] = ["new"]
    with pytest.raises(ValueError, match="simulation_program"):
        validate_frozen_plan(frozen, state=state)


def test_cross_stage_roadmap_requires_both_even_when_active_step_is_exploration():
    action = {
        "operation": "generate_branches",
        "tool": "generate_branches",
        "research_plan": {
            "active_step_id": "a",
            "steps": [
                {
                    "step_id": "a",
                    "operation": "generate_branches",
                    "priority": 1,
                    "reason": "explore",
                },
                {
                    "step_id": "b",
                    "operation": "prepare_mlip_training",
                    "priority": 2,
                    "reason": "update",
                },
            ],
        },
    }
    assert set(necessary_roles(action)) == {"exploration", "reliability"}


def test_unresolved_extrapolation_cannot_be_hidden_by_fixed_input_protocol():
    action, state = fixed()
    context = {"reliability_extrapolation_rounds": {"rounds": [{"direct_sampling_required": True}]}}
    assert set(necessary_roles(action, state=state, context=context)) == {"reliability"}


def test_query_answer_does_not_start_consultation():
    calls = []

    def answer(request):
        calls.append(request["mode"])
        return {"kind": "answer", "answer": "Recorded facts only."}

    answer._semantic_dialogue = True
    data = payload()
    data["unified_dialogue"] = True
    result = request_with_langgraph(answer, data)
    assert result["kind"] == "answer" and calls == ["autonomous_search"]


@pytest.mark.parametrize(
    "tool,roles",
    [
        ("generate_branches", {"exploration"}),
        ("select_dft_candidates", {"reliability"}),
        ("update_mlip", {"reliability"}),
    ],
)
def test_native_specific_scientific_choices_block_before_previews(tmp_path, tool, roles):
    control, context = make_control(tmp_path)
    context["allowed_tools"] = [tool]
    t = team(control)
    view = t.start(question="New scientific choice", consult=[])
    result = t.draft(view["team_id"], 1, {"tool": tool}, rationale="proposed choice")
    assert set(result["missing_roles"]) == roles
    assert set(result["requests"]) == roles
    assert not result.get("draft")


def test_old_native_pause_exemption_does_not_call_project_model():
    from phase_agent.runtime.codex.no_project_model import NoProjectModel

    proposal = {"raw_action": {"tool": "pause_search"}}
    assert (
        prepare_proposal(proposal, NoProjectModel(), state={}, config={}, registry={}) == proposal
    )

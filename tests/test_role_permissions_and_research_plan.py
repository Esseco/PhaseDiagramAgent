"""Four-role permissions, roadmaps and confirmed authorization boundaries."""

from copy import deepcopy
import pytest

from phase_agent.agents.contracts import validate_advice
from phase_agent.evidence.snapshot import evidence_snapshot
from phase_agent.agents.common.action_contracts import action_contract_errors
from phase_agent.tools.dispatch.scientific_operations import operation_catalog
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from tests.test_event_loop import _session
from tests.test_simulation_agent import SimulationModel


def consultation(role, tool="generate_branches", **patch):
    snapshot = evidence_snapshot(
        {
            "allowed_tools": [
                "generate_branches",
                "update_mlip",
                "check_convergence",
                "select_dft_candidates",
            ],
            "scientific_operations": operation_catalog(),
        }
    )
    opinion = {
        "role": role,
        "snapshot_id": snapshot["snapshot_id"],
        "assessment": "Synthetic facts only",
        "recommendation": "defer",
        "options": [{"tool": tool, "reason": "synthetic", "parameters": {}}],
        "evidence_refs": [],
        "expected_outcome": "Unknown until executed",
        "limitations": "No measured benefit",
        **patch,
    }
    return snapshot, opinion


@pytest.mark.parametrize(
    "role,tool",
    [
        ("exploration", "update_mlip"),
        ("exploration", "select_dft_candidates"),
        ("reliability", "generate_branches"),
    ],
)
def test_specialist_cannot_propose_another_specialists_action(role, tool):
    snapshot, opinion = consultation(role, tool)
    with pytest.raises(ValueError, match="role|permission|scope"):
        validate_advice(opinion, role=role, snapshot=snapshot)


def test_exploration_can_reference_existing_registered_operation_and_estimate_cost():
    snapshot, opinion = consultation("exploration")
    opinion["options"][0].update(
        operation="generate_branches",
        estimated_cost=2.0,
        expected_benefit="Improve uncovered composition coverage",
    )
    result = validate_advice(opinion, role="exploration", snapshot=snapshot)
    assert result["options"][0]["operation"] == "generate_branches"
    assert result["options"][0]["estimated_cost"] == 2


def test_research_plan_cannot_select_a_different_active_execution():
    action = {
        "tool": "check_convergence",
        "operation": "check_convergence",
        "research_plan": {
            "active_step_id": "a",
            "steps": [
                {
                    "step_id": "a",
                    "operation": "generate_branches",
                    "priority": 1,
                    "reason": "explore",
                    "depends_on": [],
                }
            ],
        },
    }
    assert action_contract_errors(action)


def test_research_plan_rejects_dependency_cycles():
    action = {
        "tool": "check_convergence",
        "operation": "check_convergence",
        "research_plan": {
            "active_step_id": "a",
            "steps": [
                {
                    "step_id": "a",
                    "operation": "check_convergence",
                    "priority": 1,
                    "reason": "inspect",
                    "depends_on": ["b"],
                },
                {
                    "step_id": "b",
                    "operation": "generate_branches",
                    "priority": 2,
                    "reason": "explore",
                    "depends_on": ["a"],
                },
            ],
        },
    }
    assert action_contract_errors(action)


def test_confirmed_auto_operation_reuses_existing_execution_policy():
    session = _session()
    session["confirmed_snapshot"]["config"]["workflow"]["auto_approved_operations"] = [
        "check_convergence"
    ]
    calls = []
    result = run_tool_step(
        {},
        session,
        registry=create_tool_registry(
            {"check_convergence": lambda **kw: calls.append(kw) or {"converged": False}}
        ),
        agent_client=lambda _: {"tool": "check_convergence", "operation": "check_convergence"},
        execution_mode="interactive",
    )
    assert result["status"] == "completed" and len(calls) == 1
    assert result["execution_authorization"]["source"] == "confirmed_policy"
    assert result["human_feedback"] is None


def test_ordinary_manual_proposal_does_not_add_mandatory_direction_round():
    model = SimulationModel()
    model.supports_strategic_review = True
    first = run_tool_step(
        {},
        _session(),
        registry=create_tool_registry({"check_convergence": lambda **_: {"converged": False}}),
        agent_client=model,
        execution_mode="interactive",
    )
    assert (first["agent_proposal"].get("strategic_review") or {}).get(
        "status"
    ) != "awaiting_human_input"


def test_runtime_settings_cannot_grant_automatic_execution():
    session = _session()
    runtime = deepcopy(session["confirmed_snapshot"]["config"])
    runtime["workflow"]["auto_approved_operations"] = ["check_convergence"]
    calls = []
    result = run_tool_step(
        {},
        session,
        registry=create_tool_registry({"check_convergence": lambda **kw: calls.append(kw) or {}}),
        agent_client=lambda _: {"tool": "check_convergence", "operation": "check_convergence"},
        context={"effective_config": runtime},
        execution_mode="interactive",
    )
    assert result["status"] == "awaiting_approval" and not calls


def test_major_conflict_and_high_cost_override_policy_grants():
    from phase_agent.agents.strategic.human_review import policy_auto_approved

    config = {
        "workflow": {
            "auto_approved_operations": ["check_convergence"],
            "human_review_relative_cost": 5,
        }
    }
    action = {"operation": "check_convergence", "tool": "check_convergence", "budget": 0}
    assert not policy_auto_approved(action, config, estimated_cost=10)
    action["human_review_request"] = {
        "required": False,
        "expert_conflict": "major",
        "reason": "Experts disagree on the acceptable evidence",
    }
    assert not policy_auto_approved(action, config)


def test_sensitive_operation_cannot_be_auto_authorized():
    from phase_agent.agents.strategic.human_review import policy_auto_approved

    action = {
        "tool": "update_mlip",
        "operation": "activate_mlip_candidate",
        "parameters": {"action": "ACTIVATE_CANDIDATE_MODEL"},
    }
    assert not policy_auto_approved(
        action, {"workflow": {"auto_approved_operations": ["activate_mlip_candidate"]}}
    )


def test_future_plan_steps_do_not_execute_under_current_approval():
    action = {
        "tool": "check_convergence",
        "operation": "check_convergence",
        "research_plan": {
            "active_step_id": "a",
            "steps": [
                {
                    "step_id": "a",
                    "operation": "check_convergence",
                    "priority": 1,
                    "reason": "inspect",
                    "depends_on": [],
                },
                {
                    "step_id": "b",
                    "operation": "generate_branches",
                    "priority": 2,
                    "reason": "explore after inspection",
                    "depends_on": ["a"],
                },
            ],
        },
    }
    assert action_contract_errors(action) == []
    calls = []
    result = run_tool_step(
        {},
        _session(),
        registry=create_tool_registry(
            {
                "check_convergence": lambda **kw: calls.append("check") or {"converged": False},
                "generate_branches": lambda **kw: calls.append("generate") or {},
            }
        ),
        agent_client=lambda _: action,
        execution_mode="autonomous",
    )
    assert result["status"] == "completed" and calls == ["check"]


def test_native_direction_uses_only_confirmed_policy(tmp_path):
    from tests.test_codex_native_team import make_control, team

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["prepare_local_batch_files"]
    control.chat_handler.workflow_kwargs = {
        "config_session": {
            "status": "confirmed",
            "confirmed_snapshot": {
                "config_version": "c1",
                "config": {"workflow": {"auto_approved_operations": ["prepare_mlip_relax"]}},
            },
        }
    }
    native = team(control)
    started = native.start(question="prepare a permitted input", consult=[])
    draft = native.draft(
        started["team_id"],
        1,
        {
            "operation": "prepare_mlip_relax",
            "tool": "prepare_local_batch_files",
            "parameters": {"mode": "relax_inputs"},
        },
        rationale="Within the confirmed scope",
    )
    result = native.simulate(
        started["team_id"], 1, SimulationModel()(draft["requests"]["simulation"])
    )
    assert result["status"] == "ready_to_propose"
    assert result["direction_review"]["status"] == "policy_authorized"
    assert not any(row["kind"] == "human_accept" for row in result["events"])


def test_unknown_policy_operation_is_not_silently_accepted():
    from phase_agent.agents.strategic.human_review import authorization_settings

    with pytest.raises(ValueError, match="registered operations"):
        authorization_settings({"workflow": {"auto_approved_operations": ["invented_action"]}})


def test_exploration_sampling_adjustment_cannot_change_dft_policy():
    snapshot, opinion = consultation("exploration", "adjust_strategy")
    snapshot["allowed_tools"].append("adjust_strategy")
    opinion["options"][0]["parameters"] = {"patch": {"round_strategy.maximum_dft_budget": 999999}}
    with pytest.raises(ValueError, match="role|permission|scope"):
        validate_advice(opinion, role="exploration", snapshot=snapshot)


@pytest.mark.parametrize("explicit_operation", [True, False])
def test_specialist_cannot_propose_hidden_operation_on_a_shared_tool(explicit_operation):
    snapshot, opinion = consultation("reliability", "update_mlip")
    snapshot["facts"]["scientific_operations"] = [
        row for row in operation_catalog() if row["operation"] == "activate_mlip_candidate"
    ]
    opinion["options"][0]["parameters"] = {"action": "RETRAIN_MLIP", "prepare_inputs_only": True}
    if explicit_operation:
        opinion["options"][0]["operation"] = "prepare_mlip_training"
    with pytest.raises(ValueError, match="scope|catalog"):
        validate_advice(opinion, role="reliability", snapshot=snapshot)


def test_active_research_cost_can_only_strengthen_human_review():
    from phase_agent.agents.strategic.human_review import policy_auto_approved

    config = {
        "workflow": {
            "auto_approved_operations": ["check_convergence"],
            "human_review_relative_cost": 5,
        }
    }
    action = {
        "tool": "check_convergence",
        "operation": "check_convergence",
        "budget": 0,
        "research_plan": {
            "active_step_id": "a",
            "steps": [
                {
                    "step_id": "a",
                    "operation": "check_convergence",
                    "priority": 1,
                    "reason": "inspect",
                    "estimated_cost": 100,
                }
            ],
        },
    }
    assert not policy_auto_approved(action, config, estimated_cost=0)


def test_approved_retry_reuses_bounded_existing_task_recovery():
    from phase_agent.tools.workflows.create_workflow_handlers import create_workflow_handlers

    session = _session()
    config = session["confirmed_snapshot"]["config"]
    config["workflow"]["auto_approved_operations"] = ["restart_failed_task"]
    original = {
        "task_id": "T",
        "task_key": "failed-mc",
        "stage": "deep_search",
        "status": "failed",
        "model_version": "m1",
        "planned_relative_cost": 1,
    }
    action = {
        "tool": "restart_failed_task",
        "operation": "restart_failed_task",
        "task_key": "retry-T",
        "target_ids": ["T"],
        "parameters": {"max_retries": 1},
        "budget": 0,
    }
    result = run_tool_step(
        {"tasks": [original]},
        session,
        registry=create_tool_registry(create_workflow_handlers()),
        agent_client=lambda _: action,
        execution_mode="interactive",
        context={"effective_config": config},
    )
    assert result["status"] == "prepared"
    assert result["state"]["tasks"][0] == original
    retry = result["state"]["tasks"][1]
    assert retry["retry_of"] == "T" and retry["retry_index"] == 1
    assert not retry.get("job_id")


@pytest.mark.parametrize(
    "parameters",
    [{"max_retries": 2}, {"max_retries": 1, "repair_profile": "vasp_symmetry_single_gpu"}],
)
def test_retry_scope_or_parameter_expansion_requires_human(parameters):
    from phase_agent.agents.strategic.human_review import policy_auto_approved

    action = {
        "tool": "restart_failed_task",
        "operation": "restart_failed_task",
        "parameters": parameters,
    }
    assert not policy_auto_approved(
        action,
        {"workflow": {"auto_approved_operations": ["restart_failed_task"], "auto_retry_limit": 1}},
    )

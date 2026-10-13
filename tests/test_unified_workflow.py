from copy import deepcopy
import pytest

from phase_agent.tools.policy.workflow_policy import (
    settings,
    normalize_workflow_state,
    branch_batch_errors,
    action_errors,
    needs_human,
)
from phase_agent.tools.workflows.run_event_loop import run_action_turn


@pytest.mark.parametrize("mode", ["Sequential", "Adaptive"])
def test_old_mode_does_not_select_scheduler_or_approve(mode):
    from tests.test_event_loop import _session as make_session

    session = make_session()
    calls = []

    def agent(payload):
        calls.append(payload)
        return {"tool": "pause_search", "reason": "inspect evidence", "budget": 0}

    state = {
        "search_mode": mode,
        "model_update_epochs": [],
        "pending_execution_policies": {"__search_mode__": {"anything": "switch"}},
        "adaptive": {"action_plan": {"actions": []}},
    }
    result = run_action_turn(
        state,
        session,
        registry={"pause_search": {"handler": lambda **kw: {"status": "paused"}}},
        agent_client=agent,
        execution_mode="interactive",
    )
    assert result["status"] == "awaiting_approval"
    assert len(calls) == 1 and calls[0].get("mode") != "adaptive_plan"
    assert "search_mode" not in result["state"]
    assert "__search_mode__" not in result["state"]["pending_execution_policies"]


def test_migration_preserves_tasks_and_scientific_approvals_idempotently():
    state = {
        "search_mode": "Adaptive",
        "tasks": [{"task_id": "t", "status": "running"}],
        "adaptive": {"action_plan": {"actions": [{"status": "submitted"}]}},
        "pending_execution_policies": {"__search_mode__": {}, "science": {"hash": "frozen"}},
    }
    original = deepcopy(state)
    migrated = normalize_workflow_state(state)
    assert state == original
    assert migrated["tasks"] == original["tasks"]
    assert migrated["pending_execution_policies"] == {"science": {"hash": "frozen"}}
    assert normalize_workflow_state(migrated) == migrated


def test_settings_use_unified_parameters_and_legacy_fallback():
    config = {
        "run": {"total_quota": 900},
        "workflow": {
            "branch_batch_size": 120,
            "finetune_min_new_dft": 3,
            "finetune_max_new_dft": 8,
        },
    }
    assert settings(config) == config["workflow"]
    assert settings({}) == {
        "branch_batch_size": 900,
        "finetune_min_new_dft": 10,
        "finetune_max_new_dft": None,
    }
    assert settings(
        {
            "agent": {
                "adaptive": {
                    "N_min": 2,
                    "N_max": 4,
                    "branch_batch_fraction": 0.4,
                    "branch_batch_max": 300,
                }
            }
        }
    ) == {"branch_batch_size": 300, "finetune_min_new_dft": 2, "finetune_max_new_dft": 4}


@pytest.mark.parametrize(
    "patch",
    [
        {"branch_batch_size": 0},
        {"branch_batch_size": True},
        {"finetune_min_new_dft": 0},
        {"finetune_max_new_dft": 2},
        {"finetune_max_new_dft": True},
    ],
)
def test_invalid_settings_rejected(patch):
    with pytest.raises(ValueError):
        settings({"workflow": patch})


@pytest.mark.parametrize(
    "params",
    [
        {"total_quota": 11},
        {"quotas": {"coverage": 6, "composition": 5}},
        {
            "generation_plan": [
                {"strategy": "coverage", "quota": 6},
                {"strategy": "composition", "quota": 5},
            ]
        },
    ],
)
def test_batch_limit_cannot_be_bypassed(params):
    assert branch_batch_errors({"parameters": params}, {"workflow": {"branch_batch_size": 10}})


def test_training_minimum_and_inflight_gate():
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    config = {"workflow": {"finetune_min_new_dft": 1}}
    action = {"tool": "update_mlip", "parameters": {"prepare_inputs_only": True}}
    assert action_errors(action, {}, config) == ["finetune_min_new_dft_not_reached"]
    record = {
        "data_id": "d1",
        "status": "completed",
        "converged": True,
        "checks_passed": True,
        "energy": -1,
    }
    assert spin_standard_passed(record)
    state = {"new_dft_records": [record]}
    assert action_errors(action, state, config) == []
    state["tasks"] = [{"task_id": "pending", "status": "running"}]
    assert "fine_tune_waits_for_inflight_tasks" in action_errors(action, state, config)
    state["new_dft_records"][0]["energy"] = float("nan")
    assert "finetune_min_new_dft_not_reached" in action_errors(action, state, config)


def test_optional_max_counts_inflight_dft_and_does_not_force_training():
    config = {"workflow": {"finetune_min_new_dft": 1, "finetune_max_new_dft": 2}}
    state = {
        "tasks": [
            {"task_id": "a", "structure_id": "a", "status": "running", "stage": "dft_single_point"}
        ]
    }
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [{"candidate_id": x, "action": "DFT_SINGLE_POINT"} for x in ("b", "c")]
        },
    }
    assert action_errors(action, state, config) == ["dft_accumulation_exceeds_finetune_max"]
    assert (
        action_errors(
            {"tool": "generate_branches", "parameters": {"total_quota": 1}}, state, config
        )
        == []
    )


def test_lower_workflow_threshold_reaches_existing_training_backend():
    from phase_agent.tools.workflows.create_workflow_handlers import _update_mlip

    calls = []

    def trainer(**kwargs):
        calls.append(kwargs)
        return {"model_path": "fake-model", "artifact": "fake-model"}

    config = {
        "workflow": {"finetune_min_new_dft": 1},
        "mlip_finetune": {"enabled": True, "training": {"minimum_new_dft_records": 10}},
    }
    state = {
        "active_model_version": "m1",
        "active_model": {"version": "m1"},
        "new_dft_records": [
            {
                "data_id": "d1",
                "task_id": "t1",
                "structure_id": "s1",
                "model_version": "m1",
                "status": "completed",
                "converged": True,
                "checks_passed": True,
                "energy": -1.0,
            }
        ],
    }
    _update_mlip(
        action={"tool": "update_mlip", "parameters": {}},
        context={"effective_config": config, "event_state": state, "mlip_trainer": trainer},
    )
    assert calls
    assert config["mlip_finetune"]["training"]["minimum_new_dft_records"] == 10


def test_agent_expansion_proposal_survives_defaults_and_requires_human():
    from phase_agent.decisions.agent.propose_tool_action import _apply_generation_defaults

    config = {
        "system": {"H_generation": {"enabled": True, "first_round_max_det_H": 4}},
        "workflow": {"branch_batch_size": 20},
    }
    action = _apply_generation_defaults(
        {"tool": "generate_branches", "parameters": {"total_quota": 10, "max_det_H": 8}}, {}, config
    )
    assert action["parameters"]["max_det_H"] == 8
    assert needs_human(action, config)


def test_sensitive_approvals_remain():
    assert needs_human(
        {"tool": "update_mlip", "parameters": {"action": "ACTIVATE_CANDIDATE_MODEL"}}, {}
    )


def test_declining_dft_remains_allowed_above_new_accumulation_limit():
    record = {
        "data_id": "d1",
        "status": "completed",
        "converged": True,
        "checks_passed": True,
        "energy": -1,
    }
    state = {"new_dft_records": [record, {**record, "data_id": "d2"}]}
    action = {
        "tool": "select_dft_candidates",
        "parameters": {"decisions": [{"candidate_id": "candidate", "action": "DEFER"}]},
    }
    assert (
        action_errors(
            action, state, {"workflow": {"finetune_min_new_dft": 1, "finetune_max_new_dft": 1}}
        )
        == []
    )
    assert needs_human(
        {
            "tool": "select_dft_candidates",
            "parameters": {"decisions": [{"candidate_id": "d", "action": "DFT_RELAX"}]},
        },
        {},
    )
    assert needs_human(
        {"tool": "generate_branches", "parameters": {"max_det_H": 5}},
        {"generation_actions": {"first_round_max_det_H": 4}},
    )


@pytest.mark.parametrize("stage", ["dft_single_point", "dft_relax"])
def test_direct_dft_stage_cannot_bypass_accumulation_limit(stage):
    config = {"workflow": {"finetune_min_new_dft": 1, "finetune_max_new_dft": 1}}
    state = {
        "new_dft_records": [
            {
                "data_id": "old",
                "status": "completed",
                "converged": True,
                "checks_passed": True,
                "energy": -1,
            }
        ]
    }
    action = {
        "tool": "run_calculation_stage",
        "stage": stage,
        "target_ids": ["new"],
        "parameters": {},
    }
    assert "dft_accumulation_exceeds_finetune_max" in action_errors(action, state, config)
    if stage == "dft_relax":
        assert needs_human(action, config)


@pytest.mark.parametrize("busy", [False, True])
def test_confirmed_policy_revision_rebinds_only_idle_existing_project(busy):
    from phase_agent.configuration.runtime.authorize_budget_extension import (
        authorize_budget_extension,
    )

    old = {"mlip": {"version": "m1"}, "workflow": {"branch_batch_size": 50}}
    new = deepcopy(old)
    new["workflow"] = {
        "branch_batch_size": 100,
        "finetune_min_new_dft": 3,
        "finetune_max_new_dft": 20,
    }
    state = {
        "confirmed_config_version": "old",
        "confirmed_config": old,
        "tasks": [{"task_id": "t", "status": "running" if busy else "completed"}],
    }
    result = authorize_budget_extension(state, {"config_version": "new", "config": new})
    assert result["status"] == ("rejected_active_work" if busy else "rebound")
    assert result["state"]["tasks"] == state["tasks"]
    assert result["state"]["confirmed_config_version"] == ("old" if busy else "new")


def test_mode_ui_and_switch_api_are_removed():
    from phase_agent.runtime.agent_api import _approval_page
    from phase_agent.runtime.control_request_contracts import MODELS

    assert "/phase/search-mode" not in MODELS
    assert "searchMode" not in _approval_page()
    assert "renderMode" not in _approval_page()


def test_new_limits_are_confirmed_science_config_not_runtime_overrides():
    from tests.test_event_loop import _session
    from phase_agent.configuration.runtime.build_effective_run_config import (
        build_effective_run_config,
    )

    session = _session()
    expected = session["confirmed_snapshot"]["config"]["workflow"]
    assert expected["branch_batch_size"] == 900
    effective = build_effective_run_config(session, {"workflow": {"branch_batch_size": 1}})
    assert effective["workflow"] == expected


def test_branch_followup_proposes_package_without_carrying_approval():
    from phase_agent.tools.workflows.propose_followup import propose_after_generation

    calls = []

    def event_loop(state, session, **kwargs):
        calls.append(kwargs)
        return {"status": "awaiting_approval", "state": state, "events": []}

    frame = {
        "execution_mode": "interactive",
        "agent_client": lambda p: {},
        "config_session": {},
        "effective_registry": {},
        "effective_config": {},
        "context": {},
    }
    result = {
        "status": "completed",
        "state": {},
        "events": [
            {
                "action": {"tool": "generate_branches"},
                "execution": {
                    "status": "completed",
                    "result": {"summary": {"registered_structures": 12}},
                },
            }
        ],
    }
    response = propose_after_generation(frame, result, event_loop=event_loop)
    assert response["status"] == "awaiting_approval"
    assert calls[0]["human_feedback"] is None and calls[0]["execution_mode"] == "interactive"
    assert "prepare_relax_mc_batch" in calls[0]["context"]["user_message"]


def test_legacy_scientific_approval_reuses_frozen_action_without_model_call():
    from tests.test_event_loop import _session

    session = _session()
    registry = {"pause_search": {"handler": lambda **kw: {"status": "paused"}}}
    proposal = run_action_turn(
        {"search_mode": "Adaptive"},
        session,
        registry=registry,
        agent_client=lambda p: {"tool": "pause_search", "reason": "inspect", "budget": 0},
        execution_mode="interactive",
        invocation_id="frozen",
    )
    legacy = deepcopy(proposal["state"])
    legacy["search_mode"] = "Adaptive"
    legacy["adaptive"] = {
        "action_plan": {"actions": [{"invocation_id": "frozen", "status": "awaiting_approval"}]}
    }

    def forbidden(payload):
        raise AssertionError("approval must not ask the model again")

    response = run_action_turn(
        legacy,
        session,
        registry=registry,
        agent_client=forbidden,
        execution_mode="interactive",
        invocation_id="frozen",
        human_feedback={"decision": "approve", "comment": "同意"},
    )
    assert response["status"] == "paused"
    assert "frozen" not in response["state"]["pending_execution_policies"]

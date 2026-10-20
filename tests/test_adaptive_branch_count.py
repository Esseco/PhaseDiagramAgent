from copy import deepcopy
import pytest
from phase_agent.tools.policy.adaptive_branch_count import (
    policy_for,
    budget_limits_for_action,
)
from phase_agent.tools.policy.workflow_policy import branch_batch_errors
from phase_agent.decisions.agent.propose_tool_action import _apply_generation_defaults


def action(count):
    return {
        "tool": "generate_branches",
        "task_key": "new",
        "parameters": {"total_quota": count, "quotas": {"coverage": count}},
        "budget": 10000,
        "reason": "覆盖新区域",
        "expected_purpose": "生成本轮branch",
        "decision_context": {"branch_count_policy": policy_for({})},
    }


@pytest.mark.parametrize("count", [100, 250, 600, 1000])
def test_new_proposal_range_overrides_old_default_cap(count):
    config = {"run": {"total_quota": 300, "batch_size": 96}}
    assert branch_batch_errors(action(count), config) == []
    proposal = _apply_generation_defaults(action(count), {}, config)
    assert proposal["parameters"]["total_quota"] == count
    assert proposal["parameters"]["batch_size"] == count


@pytest.mark.parametrize("count", [99, 1001])
def test_outside_range_is_rejected_without_clamping(count):
    a = action(count)
    before = deepcopy(a)
    assert branch_batch_errors(a, {"run": {"total_quota": 300}})
    with pytest.raises(ValueError):
        _apply_generation_defaults(a, {}, {"run": {"total_quota": 300}})
    assert a == before


def test_no_default_count_for_new_proposal():
    a = action(100)
    a["parameters"] = {}
    with pytest.raises(ValueError, match="explicit_quota"):
        _apply_generation_defaults(a, {}, {"run": {"total_quota": 300}})


def test_old_approved_proposal_keeps_original_count_and_budget_rules():
    a = action(10)
    a.pop("decision_context")
    config = {"run": {"total_quota": 300, "batch_size": 96}}
    assert branch_batch_errors(a, config) == []
    assert _apply_generation_defaults(a, {}, config)["parameters"]["total_quota"] == 10
    limits = {"enforcement": "hard", "total_relative_cost": 0}
    assert budget_limits_for_action(a, limits) == limits
    assert budget_limits_for_action(action(600), limits)["enforcement"] == "advisory"
    assert limits["enforcement"] == "hard"


def test_first_round_guidance_and_following_round_are_distinct():
    assert policy_for({})["first_round"]
    assert not policy_for({"available_branches": [{"branch_id": "b"}]})["first_round"]


def test_budget_estimate_cannot_veto_new_branch_proposal():
    from phase_agent.tools.policy.validate_tool_action import validate_tool_action

    config = {
        "agent": {"allowed_tools": ["generate_branches"]},
        "run": {"total_quota": 300},
        "budgets": {"enforcement": "hard", "total_relative_cost": 0},
        "frozen_parameters": [],
    }
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {"config_version": "v1", "config": config},
    }
    result = validate_tool_action(
        action(600), {}, session, {"generate_branches": {"handler": lambda **kw: None}}
    )
    assert "branch_batch_exceeds_limit" not in result["errors"]
    assert "reserved_total_cost_limit" not in result["errors"]
    assert not any("budget" in error or "cost_limit" in error for error in result["errors"])


@pytest.mark.parametrize("guards", [None, {"max_per_framework": 8, "max_per_parent_branch": 2}])
def test_generation_entry_does_not_truncate_to_old_batch_or_budget(tmp_path, monkeypatch, guards):
    from types import SimpleNamespace
    from importlib import import_module

    module = import_module("phase_agent.tools.workflows.create_active_learning_handlers")
    import phase_agent.tools.workflows.scientific_module_scope as scope

    calls = []

    def fake(manager, references, **kwargs):
        calls.append(kwargs)
        return {"registered": [], "coverage": {}, "summary": {}}

    monkeypatch.setattr(module, "run_branch_generation", fake)
    monkeypatch.setattr(
        scope, "prepare_generated_module", lambda action, generated, context: generated
    )
    import phase_agent.analysis.cost.generation_preflight as preflight

    monkeypatch.setattr(preflight, "attach_generation_preflight", lambda a, *args: a)
    config = {
        "total_quota": 300,
        "batch_size": 96,
        "initial_states_per_branch": 4,
        "seed": 42,
        "structure_directory": str(tmp_path),
        "generation_options": {"cost_budget": 0},
    }
    if guards is not None:
        config["generation_options"]["selection_config"] = guards
    a = action(600)
    a["parameters"]["batch_size"] = 600
    a["parameters"]["generation_plan"] = [
        {
            "strategy": "coverage",
            "quota": 600,
            "phase": "O3",
            "reason": "覆盖",
            "min_det_H": 4,
            "max_det_H": 12,
            "na_min": 0.0,
            "na_max": 1.0,
        }
    ]
    context = {
        "effective_config": config,
        "event_state": {},
        "manager": SimpleNamespace(data={"branches": {}}, boundary={"P": ["O3"]}),
        "phase_references": {},
    }
    before = deepcopy(context["event_state"])
    module._generate_branches(action=a, context=context)
    assert calls[0]["batch_size"] == 600 and calls[0]["cost_budget"] is None
    assert calls[0]["selection_config"] == (
        guards or {"max_per_framework": 600, "max_per_parent_branch": 600}
    )
    assert context["event_state"] == before


def test_preflight_keeps_estimate_without_budget_rejection(monkeypatch):
    from importlib import import_module

    module = import_module("phase_agent.analysis.cost.generation_preflight")
    monkeypatch.setattr(module, "generation_framework_costs", lambda *a: [])
    monkeypatch.setattr(
        module,
        "preview_generation_cost",
        lambda *a: {
            "status": "over_budget",
            "relax_mc_cost_upper": 999,
            "remaining_budget": 0,
        },
    )
    a = action(600)
    a["parameters"]["generation_plan"] = [
        {"strategy": "coverage", "quota": 600, "phase": "all", "reason": "coverage"}
    ]
    result = module.attach_generation_preflight(
        a, {}, {"manager": object()}, {"budgets": {"enforcement": "hard"}}
    )
    preview = result["parameters"]["generation_cost_preview"]
    assert preview["status"] == "estimate_only" and preview["budget_reference_exceeded"]
    assert preview["relax_mc_cost_upper"] == 999


def test_new_model_proposal_uses_regions_and_counts_only_after_generation(monkeypatch):
    from importlib import import_module

    module = import_module("phase_agent.decisions.agent.propose_tool_action")
    seen = []

    def fake(client, payload):
        seen.append(payload)
        return {
            "tool": "generate_branches",
            "budget": 0,
            "reason": "探索假设",
            "expected_purpose": "生成后统计",
            "parameters": {
                "sampling_mode": "region_adaptive",
                "regions": [
                    {
                        "region_id": "r1",
                        "priority": 1,
                        "strategy": "coverage",
                        "phase": "O3",
                        "min_det_H": 4,
                        "max_det_H": 12,
                        "na_min": 0.25,
                        "na_max": 0.75,
                        "reason": "根据当前覆盖作探索",
                        "evidence_refs": [],
                    }
                ],
            },
        }

    monkeypatch.setattr(module, "request_validated_action", fake)
    proposal = module.propose_agent_tool_action(
        {},
        agent_client=object(),
        allowed_tools=["generate_branches"],
        config={"run": {"total_quota": 600, "batch_size": 96}},
    )
    assert seen[0]["branch_count_policy"]["count_is_result"] is True
    assert "minimum" not in seen[0]["branch_count_policy"]
    assert proposal["decision_source"] == "llm_agent"
    assert (
        not {"total_quota", "quotas", "batch_size", "generation_plan"}
        & proposal["parameters"].keys()
    )

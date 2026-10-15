from copy import deepcopy
from types import SimpleNamespace

import pytest
from phase_agent.tools.policy.adaptive_branch_count import policy_for, generation_review_errors
from phase_agent.tools.policy.execution_policy import build_agent_proposal, apply_execution_policy
from phase_agent.decisions.agent.propose_tool_action import _apply_generation_defaults
from phase_agent.decisions.agent.generation_contracts import GenerationAllocation
from phase_agent.science.structures.propose_branches import propose_branches
from phase_agent.analysis.cost.generation_preflight import preview_generation_cost
from phase_agent.tools.workflows.branch_review_view import readable_view, raw_record


def action():
    return {
        "tool": "generate_branches",
        "task_key": "test",
        "budget": 0,
        "reason": "重新覆盖",
        "expected_purpose": "新模型搜索",
        "decision_context": {"branch_count_policy": policy_for({}), "large_evidence": [1, 2]},
        "parameters": {
            "total_quota": 300,
            "batch_size": 300,
            "quotas": {"coverage": 300},
            "generation_plan": [
                {
                    "phase": "O3",
                    "strategy": "coverage",
                    "quota": 300,
                    "min_det_H": 4,
                    "max_det_H": 8,
                    "na_min": 0.25,
                    "na_max": 0.75,
                    "reason": "分布覆盖",
                }
            ],
        },
    }


def test_policy_survives_build_and_approval_without_large_context():
    proposal = build_agent_proposal(action(), {})
    result = apply_execution_policy(
        proposal, execution_mode="interactive", human_feedback="approve"
    )
    assert result["final_action"]["decision_context"] == {"branch_count_policy": policy_for({})}
    assert generation_review_errors(result["final_action"]) == []
    legacy = deepcopy(proposal)
    legacy["raw_action"].pop("decision_context")
    restored = apply_execution_policy(legacy, execution_mode="dry_run")["final_action"]
    assert generation_review_errors(restored) == []


def test_no_silent_96_override_or_legacy_proposal_execution():
    value = action()
    value["parameters"]["batch_size"] = 96
    assert "branch_target_mismatch" in generation_review_errors(value)
    with pytest.raises(ValueError, match="branch_target_mismatch"):
        _apply_generation_defaults(value, {}, {"run": {"total_quota": 300, "batch_size": 96}})
    legacy = deepcopy(value)
    legacy.pop("decision_context")
    assert generation_review_errors(legacy, {"branch_count_policy": policy_for({})}) == [
        "branch_count_policy_requires_proposal_revision"
    ]


def test_explicit_bounds_required_and_checked():
    value = action()
    value["parameters"]["generation_plan"][0].pop("min_det_H")
    assert generation_review_errors(value) == ["branch_allocation_bounds_missing:0"]
    row = action()["parameters"]["generation_plan"][0]
    row["min_det_H"] = 9
    with pytest.raises(ValueError, match="下限不能超过上限"):
        GenerationAllocation.model_validate(row)


def test_exact_phase_na_and_det_bounds_reach_generator():
    seen = []

    class Registry:
        def names(self):
            return ["coverage"]

        def get(self, name):
            def generate(context, quota, seed, options):
                seen.extend(context["frameworks"])
                return []

            return generate

    frameworks = [
        {"P": phase, "H": [[size, 0, 0], [0, 1, 0], [0, 0, 1]], "allowed_x": ["0", "1/2", "1"]}
        for phase in ["O3", "P3"]
        for size in [2, 4, 8, 12]
    ]
    propose_branches(
        SimpleNamespace(data={"branches": {}}, boundary={}),
        {},
        quotas={"coverage": 300},
        seed=1,
        register=False,
        registry=Registry(),
        framework_enumerator=lambda *a: frameworks,
        strategy_options={"_focus": action()["parameters"]["generation_plan"][0]},
    )
    assert len(seen) == 2
    assert [r["H"][0][0] for r in seen] == [4, 8]
    assert all(r["P"] == "O3" and r["allowed_x"] == ["1/2"] for r in seen)


def test_preflight_uses_same_lower_size_bound():
    rows = [
        {
            "phase": "O3",
            "det_H": size,
            "atom_count_upper": size * 4,
            "allowed_x": [0.5],
            "stage_costs": {"relax_and_feature": cost, "deep_search": cost},
            "stage_seconds": {"relax_and_feature": None, "deep_search": None},
        }
        for size, cost in [(2, 10000), (4, 1), (8, 2), (12, 20000)]
    ]
    result = preview_generation_cost(action(), rows, {}, {})
    assert result["selected_branch_upper"] == 300 and result["relax_mc_cost_upper"] == 1200


def test_readable_target_and_rows_preserve_executable_source():
    raw = {
        "status": "awaiting_approval",
        "plan": {
            "branch_count_policy": policy_for({}),
            "candidate_quota": 300,
            "selected_branch_upper": 300,
            "initial_states_per_branch": 3,
            "allocations": action()["parameters"]["generation_plan"],
            "supercells": [{"phase": "O3", "det_H": size} for size in [2, 4, 8, 12]],
        },
    }
    view = readable_view(raw)
    assert view["数量"]["本批目标branch数"] == 300
    assert "去重后选取branch上限" not in view["数量"]
    row = view["候选配额分配"][0]
    assert row["相"] == ["O3"] and row["允许超胞倍数_detH"] == [4, 8]
    assert row["钠含量_x范围"] == [0.25, 0.75]
    assert raw_record(view) == raw
    assert list(view)[-1] == "程序记录（可折叠）"


def test_model_contract_rejects_96_before_presenting_proposal():
    from phase_agent.decisions.agent.proposal_validation import proposal_errors

    value = action()
    value["parameters"]["batch_size"] = 96
    errors = proposal_errors(
        value,
        {"allowed_tools": ["generate_branches"], "decision_context": value["decision_context"]},
    )
    assert "branch_target_mismatch" in errors


def test_feedback_revision_keeps_policy_even_without_snapshot_context(monkeypatch):
    from phase_agent.decisions.agent.revise_tool_proposal import revise_tool_proposal
    import phase_agent.decisions.agent.decision_backend as backend

    monkeypatch.setattr(backend, "request_decision_action", lambda client, payload: action())
    revised = revise_tool_proposal(
        build_agent_proposal(action(), {}),
        "revise allocations",
        state={},
        allowed_tools=["generate_branches"],
        agent_client=object(),
    )
    assert revised["revision_status"] == "revised"
    assert generation_review_errors(revised["action"]) == []


def test_shared_intermediate_allocation_does_not_list_endpoint_only_phase():
    from phase_agent.tools.workflows.branch_review_view import allocation_phases

    plan = {
        "phases": {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O3", "P3", "OP2"]},
        "supercells": [{"phase": p} for p in ["O1", "O3", "P3", "OP2"]],
    }
    assert allocation_phases(plan, {"phase": "all", "na_min": 0.05, "na_max": 0.95}) == [
        "O3",
        "OP2",
        "P3",
    ]


def test_explicit_group_guards_are_distinct_from_implicit_defaults():
    from phase_agent.tools.policy.adaptive_branch_count import configured_selection_guards

    assert configured_selection_guards({"run": {"batch_size": 96}}) == {
        "max_per_framework": None,
        "max_per_parent_branch": None,
    }
    config = {
        "run": {
            "generation_options": {
                "selection_config": {"max_per_framework": 8, "max_per_parent_branch": 2}
            }
        }
    }
    assert configured_selection_guards(config) == {
        "max_per_framework": 8,
        "max_per_parent_branch": 2,
    }

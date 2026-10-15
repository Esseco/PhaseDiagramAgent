from copy import deepcopy
import json

import pytest

from phase_agent.analysis.state.training_context import (
    validation_planning_context,
    post_refresh_search_context,
    training_review_active,
)
from phase_agent.analysis.state.build_decision_context import build_decision_context
from phase_agent.decisions.agent.deepagents_proposal import proposal_material


def state():
    return {
        "active_model_version": "new",
        "model_refresh": {
            "status": "completed",
            "new_model_version": "new",
            "old_model_version": "old",
            "task_ids": ["refresh1"],
        },
        "tasks": [
            {
                "task_id": "refresh1",
                "model_version": "new",
                "stage": "relax_and_feature",
                "status": "completed",
            },
            {
                "task_id": "old_mc",
                "model_version": "old",
                "stage": "deep_search",
                "status": "completed",
            },
        ],
        "branch_candidates": [{"branch_id": "b1"}],
        "training_validation_planning": {
            "instruction": "OLD_DFT_MUST_FIRST",
            "reviews": [
                {
                    "new": {"energy_per_atom": {"mae": 0.004}},
                    "limitations": ["CV is not independent testing"],
                    "next_proposal": "OLD_DFT_MUST_FIRST",
                }
            ],
            "agent_model_reviews": [{"followup_action": {"tool": "select_dft_candidates"}}],
        },
    }


def test_completed_refresh_projects_legacy_directive_as_history_without_mutation():
    s = state()
    before = deepcopy(s)
    context = build_decision_context(s)
    plan = context["training_validation_planning"]
    assert plan["status"] == "historical"
    assert "OLD_DFT_MUST_FIRST" not in json.dumps(plan)
    assert plan["reviews"][0]["new_metrics"]["energy_per_atom"]["mae"] == 0.004
    assert not training_review_active(context)
    assert s == before
    facts = context["post_refresh_search"]
    assert facts["mc_completed"] == 0 and facts["relax_completed"] == 0
    assert facts["available_branch_count"] == 1


@pytest.mark.parametrize("phase", ["no_refresh", "pending", "wrong_model"])
def test_no_automatic_expiry_without_matching_completed_transition(phase):
    s = state()
    if phase == "no_refresh":
        s.pop("model_refresh")
    elif phase == "pending":
        s["model_refresh"]["status"] = "waiting_results"
    else:
        s["model_refresh"]["new_model_version"] = "another"
    plan = validation_planning_context(s)
    assert plan["status"] == "active" and plan["instruction"] == "OLD_DFT_MUST_FIRST"
    assert post_refresh_search_context(s) == {}


def test_explicit_current_validation_is_preserved_but_old_scope_is_historical():
    s = state()
    s["training_validation_planning"]["scope"] = {"model_version": "new"}
    plan = validation_planning_context(s)
    assert plan["status"] == "active" and plan["instruction"] == "OLD_DFT_MUST_FIRST"
    assert training_review_active({"training_validation_planning": plan})
    s["training_validation_planning"]["scope"]["model_version"] = "old"
    assert validation_planning_context(s)["status"] == "historical"


def test_cancelled_current_request_is_not_reactivated():
    s = state()
    s["training_validation_planning"].update(status="cancelled", model_version="new")
    assert validation_planning_context(s)["status"] == "historical"


def test_new_unactivated_candidate_still_enters_current_training_review():
    s = state()
    s["remote_finetune_jobs"] = {
        "round2": {
            "status": "results_received",
            "activated": False,
            "original_model_version": "new",
        }
    }
    s["training_validation_planning"]["status"] = "historical"
    assert post_refresh_search_context(s) == {}
    assert training_review_active(
        {
            "training_already_completed": True,
            "training_validation_planning": validation_planning_context(s),
            "post_refresh_search": {},
        }
    )


def test_cached_snapshot_cannot_reintroduce_obsolete_directive():
    from phase_agent.tools.state.state_manager import agent_state_summary

    s = state()
    s["current_state_snapshot"] = {
        "decision_context": {
            "training_validation_planning": deepcopy(s["training_validation_planning"])
        }
    }
    before = deepcopy(s)
    summary = agent_state_summary(s)
    assert summary["decision_context"]["training_validation_planning"]["status"] == "historical"
    assert summary["decision_context"]["post_refresh_search"]["mc_completed"] == 0
    assert s == before


def test_compact_transport_keeps_current_facts_and_scoped_status():
    c = build_decision_context(state())
    c["decision_authority"] = c["post_refresh_search"]["instruction"]
    compact = proposal_material({"decision_context": c})["decision_context"]
    assert compact["training_validation_planning"]["status"] == "historical"
    assert compact["post_refresh_search"]["model_version"] == "new"
    assert compact["post_refresh_search"]["mc_completed"] == 0
    assert (
        compact["training_validation_planning"]["reviews"][0]["new_metrics"]["energy_per_atom"][
            "mae"
        ]
        == 0.004
    )
    assert compact["decision_authority"] == c["decision_authority"]
    assert "OLD_DFT_MUST_FIRST" not in json.dumps(compact)


def test_scientific_request_exposes_search_and_dft_without_forcing_action():
    from phase_agent.decisions.agent.propose_tool_action import propose_agent_tool_action

    s = state()
    s["decision_context"] = build_decision_context(s)
    requests = []

    def agent(payload):
        requests.append(payload)
        return {
            "tool": "select_dft_candidates",
            "task_key": "new-evidence-dft",
            "target_ids": [],
            "parameters": {
                "decisions": [
                    {"candidate_id": "s1", "action": "DFT_SINGLE_POINT", "reason": "新发现近简并"}
                ]
            },
            "budget": 1,
            "reason": "新增近简并候选需核对排序",
            "expected_purpose": "核对具体候选",
            "evidence_refs": ["candidate:s1"],
        }

    result = propose_agent_tool_action(
        s,
        allowed_tools=["generate_branches", "select_dft_candidates", "pause_search"],
        agent_client=agent,
    )
    assert requests
    request = requests[0]
    assert (
        "generate_branches" in request["allowed_tools"]
        and "select_dft_candidates" in request["allowed_tools"]
    )
    assert request["decision_context"]["training_validation_planning"]["status"] == "historical"
    assert (
        request["decision_context"]["decision_authority"]
        == request["decision_context"]["post_refresh_search"]["instruction"]
    )
    assert result["tool"] == "select_dft_candidates"
    assert "严禁update_mlip" not in request["instruction"]

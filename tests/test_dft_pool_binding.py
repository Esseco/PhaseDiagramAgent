"""Current DFT pool and independent audit references must never disagree."""

from copy import deepcopy
import pytest
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)


def fixture():
    config = default_layered_search_config()
    candidates = [
        {
            "candidate_id": f"new-{i}",
            "structure_id": f"new-{i}",
            "phase": "O3",
            "x_Na_per_O2": i / 3,
            "atom_count": 12,
            "composition": {"Na": 1, "Co": 1, "O": 2},
        }
        for i in range(4)
    ]
    state = {
        "qbc_candidates": candidates,
        "dft_audit_candidate_ids": ["old"],
        "tasks": [],
        "active_model_version": "m",
        "confirmed_config_version": "v",
    }
    summary = {
        "qbc_candidates": [{"candidate_id": "stale"}],
        "decision_context": {
            "independent_audit_candidate_ids": ["old"],
            "qbc_candidates": [{"candidate_id": "stale"}],
        },
    }
    return state, summary, config


def test_refresh_uses_exact_current_pool_and_does_not_change_scientific_candidates():
    from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence

    state, summary, config = fixture()
    before = deepcopy(state["qbc_candidates"])
    refresh_dft_evidence(state, summary, config)
    ids = {r["candidate_id"] for r in before}
    assert state["dft_audit_candidate_ids"] and set(state["dft_audit_candidate_ids"]) <= ids
    assert (
        summary["decision_context"]["independent_audit_candidate_ids"]
        == state["dft_audit_candidate_ids"]
    )
    assert summary["qbc_candidates"] == summary["decision_context"]["qbc_candidates"] == before
    assert state["qbc_candidates"] == before
    assert {r["candidate_id"] for r in summary["decision_context"]["dft_candidate_costs"]} == ids
    assert set(summary["decision_context"]["dft_sampling_recommendation"]["candidate_ids"]) <= ids


def test_empty_pool_clears_stale_audit_without_inventing_candidates():
    from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence

    state, summary, config = fixture()
    state["qbc_candidates"] = []
    refresh_dft_evidence(state, summary, config)
    assert state["dft_audit_candidate_ids"] == []
    assert summary["decision_context"]["independent_audit_candidate_ids"] == []
    assert summary["decision_context"]["dft_candidate_costs"] == []
    assert summary["qbc_candidates"] == []


def test_repair_fields_are_in_dft_topic():
    from phase_agent.evidence.requests import source_topic

    assert source_topic("dft_validation_feedback") == "dft"
    assert source_topic("dft_selection_policy") == "dft"


def test_initial_proposal_refreshes_partial_mc_pool_before_model_call():
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    state, summary, config = fixture()
    original_pool = deepcopy(state["qbc_candidates"])
    calls = []

    def propose(evidence, **kwargs):
        calls.append(evidence)
        ids = {r["candidate_id"] for r in original_pool}
        assert evidence["qbc_candidates"] == original_pool
        assert set(evidence["decision_context"]["independent_audit_candidate_ids"]) <= ids
        assert "old" not in evidence["decision_context"]["independent_audit_candidate_ids"]
        return {"tool": "pause_search", "fallback_reason": "test stops after evidence inspection"}

    def forbidden(*args, **kwargs):
        raise AssertionError("No execution or additional model revision")

    result = select_initial_proposal(
        current=state,
        mode="interactive",
        config=config,
        context={"unified_dialogue": True},
        decision_state=summary,
        registry={"select_dft_candidates": {"handler": forbidden}},
        agent_client=propose,
        invocation_id="new",
        propose=propose,
        revise=forbidden,
        prepare_debug=forbidden,
        is_verified_second=lambda *a: False,
        mc_block=forbidden,
        model_failed=lambda action: True,
        record_id_factory=forbidden,
    )
    assert len(calls) == 1 and result["status"] == "not_configured"
    assert state["qbc_candidates"] == original_pool


def test_valid_existing_audit_remains_binding_during_revision():
    from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence

    state, summary, config = fixture()
    state["dft_audit_candidate_ids"] = ["new-3"]
    refresh_dft_evidence(state, summary, config)
    assert state["dft_audit_candidate_ids"] == ["new-3"]
    assert summary["dft_audit_candidate_ids"] == ["new-3"]
    assert summary["decision_context"]["independent_audit_candidate_ids"] == ["new-3"]


def test_frozen_dft_pool_retains_success_failed_and_different_frames():
    from phase_agent.tools.workflows.initial_tool_proposal import _post_dft_candidate_pool

    candidates = [
        {"candidate_id": "done", "structure_sha256": "done-hash"},
        {"candidate_id": "failed", "structure_sha256": "failed-hash"},
        {"candidate_id": "same-id-new-frame", "structure_sha256": "new-hash"},
        {"candidate_id": "new", "structure_sha256": "new2-hash"},
    ]
    state = {"tasks": [
        {"structure_id": "done", "stage": "dft_single_point", "status": "completed"},
        {"structure_id": "failed", "stage": "dft_single_point", "status": "failed"},
        {"structure_id": "same-id-new-frame", "stage": "dft_single_point", "status": "completed", "input_structure_sha256": "old-hash"},
    ]}
    assert _post_dft_candidate_pool(candidates, state, frozen_dft_action=True) == candidates
    assert [x["candidate_id"] for x in _post_dft_candidate_pool(candidates, state)] == ["new"]
    assert candidates[0]["structure_sha256"] == "done-hash"


def test_post_dft_workflow_keeps_all_frozen_frames_before_model_validation(monkeypatch):
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
    from phase_agent.evidence import post_dft_assessment
    from phase_agent.science.qbc import post_mc_candidates

    state, summary, config = fixture()
    state["tasks"] = [{"structure_id": "new-0", "stage": "dft_single_point", "status": "completed"}]
    expected = deepcopy(state["qbc_candidates"])
    monkeypatch.setattr(post_dft_assessment, "post_dft_assessment", lambda *a: {"status": "evaluated"})
    monkeypatch.setattr(post_mc_candidates, "post_mc_candidates", lambda *a: deepcopy(expected))
    class FrozenClient:
        _frozen_dft_action = {"tool": "select_dft_candidates"}
        def __call__(self, *args, **kwargs):
            return {}
    calls = []
    def propose(evidence, **kwargs):
        calls.append(evidence)
        assert evidence["qbc_candidates"] == expected
        return {"tool": "pause_search", "fallback_reason": "stop after evidence inspection"}
    def forbidden(*args, **kwargs):
        raise AssertionError("must not execute")
    result = select_initial_proposal(current=state, mode="interactive", config=config,
        context={"unified_dialogue": True}, decision_state=summary,
        registry={"select_dft_candidates": {"handler": forbidden}}, agent_client=FrozenClient(),
        invocation_id="new", propose=propose, revise=forbidden, prepare_debug=forbidden,
        is_verified_second=lambda *a: False, mc_block=forbidden,
        model_failed=lambda action: True, record_id_factory=forbidden)
    assert len(calls) == 1 and result["status"] == "not_configured"


def test_frozen_dft_action_survives_model_client_adapter():
    from phase_agent.models.capabilities import copy_model_capabilities

    class FrozenClient:
        _frozen_dft_action = {"tool": "select_dft_candidates", "target_ids": ["S-one"]}

    def wrapped(_payload):
        return None

    copy_model_capabilities(FrozenClient(), wrapped)
    assert wrapped._frozen_dft_action == FrozenClient._frozen_dft_action


def test_partial_pool_filter_preserves_surviving_independent_draw():
    from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence

    state, summary, config = fixture()
    state["dft_audit_candidate_ids"] = ["old", "new-3", "new-1"]
    refresh_dft_evidence(state, summary, config)
    assert state["dft_audit_candidate_ids"] == ["new-3", "new-1"]
    assert summary["decision_context"]["independent_audit_candidate_ids"] == ["new-3", "new-1"]

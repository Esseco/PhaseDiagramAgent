"""Pending DFT revisions self-correct once through a fake model, never execute."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.tools.workflows.tool_proposal_revision import revise_pending_proposal


def action(ids, key):
    return {
        "tool": "select_dft_candidates",
        "task_key": key,
        "decision_source": "llm_agent",
        "target_ids": ids,
        "parameters": {
            "decisions": [
                {
                    "candidate_id": value,
                    "action": "DFT_SINGLE_POINT",
                    "reason": "synthetic hull evidence",
                }
                for value in ids
            ],
        },
        "reason": "synthetic selection",
        "budget": 30.0 * len(ids),
    }


def revised(ids, key="human-revision"):
    return {
        "revision_status": "revised",
        "analysis": "fake model revised selection",
        "action": action(ids, key),
    }


def run_revision(monkeypatch, responses):
    import phase_agent.tools.workflows.tool_proposal_revision as module

    monkeypatch.setattr(module, "update_state_snapshot", lambda state, **kwargs: deepcopy(state))
    config = {
        "qbc": default_dft_decision_config(),
        "dft": {"selection": {"max_tasks_per_round": 100}},
        "agent": {"allowed_tools": ["select_dft_candidates"]},
    }
    original = {
        "recommended_action": "select_dft_candidates",
        "raw_action": action(["audit"], "original"),
        "reason": "saved pending selection",
    }
    stored = {
        "record_id": "p",
        "revision": 3,
        "agent_proposal": original,
        "feedback_history": [{"comment": "saved comment"}],
    }
    current = {
        "confirmed_config_version": "c",
        "active_model_version": "m",
        "tasks": [],
        "pending_execution_policies": {"p": stored},
        "action_records": [
            {"record_id": "p", "status": "awaiting_approval", "agent_proposal": deepcopy(original)}
        ],
        "dft_audit_candidate_ids": ["audit"],
        "qbc_candidates": [
            {"candidate_id": value, "atom_count": 40, "phase": "O3", "predicted_Ehull": 0.01}
            for value in ("near", "audit")
        ],
    }
    before = deepcopy(current)
    client = Mock(side_effect=responses)
    handler = Mock(side_effect=AssertionError("no handler may execute during revision"))

    def fake_revise(proposal, opinion, *, agent_client, state, allowed_tools, **kwargs):
        return agent_client(
            {
                "proposal": deepcopy(proposal),
                "opinion": opinion,
                "state": deepcopy(state),
                "allowed_tools": list(allowed_tools),
            }
        )

    result = revise_pending_proposal(
        current=current,
        mode="interactive",
        config=config,
        context={"effective_config": config},
        proposal=deepcopy(original),
        stored=deepcopy(stored),
        opinion="Revise from current evidence",
        decision_state={"decision_context": {"dft_audit_candidate_ids": ["audit"]}},
        record_id="p",
        pending_key="p",
        invocation_id="p",
        agent_client=client,
        registry={"select_dft_candidates": {"owner": "decision_layer", "handler": handler}},
        revise=fake_revise,
        prepare_debug_relax=lambda value, *args, **kwargs: deepcopy(value),
        apply_generation_defaults=Mock(side_effect=AssertionError("not a Branch revision")),
    )
    handler.assert_not_called()
    assert result["state"]["tasks"] == before["tasks"]
    return result, before, client


def test_missing_audit_candidate_is_revised_once_by_fake_agent(monkeypatch):
    result, before, client = run_revision(
        monkeypatch,
        [revised(["near"]), revised(["near", "audit"], "model-correction")],
    )
    assert result["status"] == "awaiting_approval"
    assert result["revision"] == 4
    assert client.call_count == 2
    correction = client.call_args_list[1].args[0]
    feedback = correction["state"]["decision_context"]["dft_validation_feedback"]
    assert "independent_dft_audit_required" in feedback["error"]
    assert correction["allowed_tools"] == ["select_dft_candidates"]
    pending = result["state"]["pending_execution_policies"]["p"]
    saved = pending["agent_proposal"]["raw_action"]
    assert saved["task_key"] == "model-correction"
    assert [row["candidate_id"] for row in saved["parameters"]["decisions"]] == ["near", "audit"]
    assert saved["parameters"]["dft_input_preview"]["task_count"] == 2
    assert (
        pending["feedback_history"][-1]["prior_proposal"]
        == before["pending_execution_policies"]["p"]["agent_proposal"]
    )
    assert result["execution"] is None and result["final_action"] is None


@pytest.mark.parametrize(
    "second",
    [
        revised(["near"], "still-missing-audit"),
        {"revision_status": "revision_failed", "analysis": "fake model unavailable"},
    ],
)
def test_failed_bounded_correction_keeps_original_pending(monkeypatch, second):
    result, before, client = run_revision(monkeypatch, [revised(["near"]), second])
    assert result["status"] == "rejected"
    assert "independent_dft_audit_required" in result["reason"]
    assert client.call_count == 2
    assert result["state"]["pending_execution_policies"] == before["pending_execution_policies"]
    assert result["state"]["action_records"] == before["action_records"]


def test_valid_revision_does_not_request_an_extra_fake_model_call(monkeypatch):
    result, before, client = run_revision(monkeypatch, [revised(["audit"], "valid-first")])
    assert result["status"] == "awaiting_approval"
    client.assert_called_once()
    assert result["action"]["task_key"] == "valid-first"
    assert result["execution"] is None

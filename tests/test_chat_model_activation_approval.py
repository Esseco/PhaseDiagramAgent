"""Chat approval of a displayed saved model switch; no live projects or models."""

from copy import deepcopy
import json
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.graphs.dialogue.support import (
    bind_presented_proposal,
    review_presented_proposal,
    _load_binding,
)
from phase_agent.runtime.chat_review import review_pending
from phase_agent.runtime.plan_queries import pending_plan_reply
from phase_agent.runtime.training_reply import concise_training_reply


@pytest.fixture
def approval(tmp_path, monkeypatch):
    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "direction_proposal": {"direction": "activate", "reason": "训练有效，切换后另审刷新"},
    }
    state = {
        "active_model_version": "old",
        "active_model": {"version": "old"},
        "confirmed_config_version": "c1",
        "tasks": [],
        "remote_finetune_jobs": {"job": {"training_handoff": handoff}},
        "candidate_models": {
            "new": {
                "status": "validated_candidate",
                "old_model_version": "old",
                "agent_review": {"choice": "activate", "reason": "训练已验证"},
                "model": {"version": "new"},
            }
        },
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    executor = Mock(return_value={"status": "activated", "reason": "切换已批准，刷新另行审阅"})
    monkeypatch.setattr("phase_agent.runtime.review_requests.review_additional", executor)
    handler = SimpleNamespace(
        state_path=path, conversation_id="chat-a", lock=RLock(), workflow_kwargs={}
    )

    def review(plan, decision, **kwargs):
        return review_pending(handler, plan, decision, **kwargs)

    handler.review_pending = Mock(side_effect=review)
    return state, handler, executor


@pytest.mark.parametrize("message", ["同意", "approve"])
def test_displayed_switch_uses_original_versioned_owner_without_token(approval, message):
    state, handler, execute = approval
    bind_presented_proposal(handler, state)
    binding = deepcopy(_load_binding(handler))
    before = handler.state_path.read_bytes()
    reply = review_presented_proposal(handler, state, message)
    assert "切换已批准" in reply
    assert "/phase/approval" not in reply
    handler.review_pending.assert_called_once_with(
        binding["plan_id"],
        "confirm_sensitive",
        expected_state_version=binding["state_version"],
        expected_proposal_hash=binding["proposal_hash"],
        comment=message,
    )
    execute.assert_called_once()
    assert execute.call_args.args[3] == "confirm_sensitive"
    assert _load_binding(handler) is None
    assert handler.state_path.read_bytes() == before


def test_first_approval_without_display_only_presents_saved_switch(approval):
    state, handler, execute = approval
    reply = review_presented_proposal(handler, state, "同意")
    assert "已保存方案" in reply and "核对" in reply
    assert "/phase/approval" in reply
    handler.review_pending.assert_not_called()
    execute.assert_not_called()


def test_semantic_approval_uses_the_same_saved_switch(approval):
    state, handler, execute = approval
    bind_presented_proposal(handler, state)
    plan_id = _load_binding(handler)["plan_id"]
    reply = review_presented_proposal(
        handler,
        state,
        "直接批准这个模型切换",
        requested_review={"plan_id": plan_id, "decision": "approve"},
    )
    assert "切换已批准" in reply
    assert execute.call_args.args[3] == "confirm_sensitive"


@pytest.mark.parametrize("change", ["proposal", "state"])
def test_stale_switch_is_rejected_by_existing_hash_or_state_guard(approval, change):
    state, handler, execute = approval
    bind_presented_proposal(handler, state)
    revised = deepcopy(state)
    if change == "proposal":
        revised["remote_finetune_jobs"]["job"]["training_handoff"]["direction_proposal"][
            "reason"
        ] = "changed"
    else:
        revised["tasks"] = [{"task_id": "new-task", "stage": "deep_search", "status": "pending"}]
    handler.state_path.write_text(json.dumps(revised), encoding="utf-8")
    assert "未执行" in review_presented_proposal(handler, revised, "同意")
    execute.assert_not_called()
    assert _load_binding(handler) is None


def test_switch_rejection_stays_rejection(approval):
    state, handler, execute = approval
    bind_presented_proposal(handler, state)
    review_presented_proposal(handler, state, "拒绝")
    assert execute.call_args.args[3] == "reject"


def test_other_sensitive_reviews_still_require_their_confirmation(approval, monkeypatch):
    state, handler, execute = approval
    row = {
        "plan_id": "recovery",
        "state_version": "v",
        "proposal_hash": "h",
        "review_kind": "execution_recovery",
        "review_card": {"sensitive": True},
    }
    monkeypatch.setattr("phase_agent.graphs.dialogue.support._review_rows", lambda *a: [row])
    bind_presented_proposal(handler, state)
    assert "/phase/approval" in review_presented_proposal(handler, state, "同意")
    execute.assert_not_called()


def test_other_conversation_cannot_use_display_binding(approval):
    state, handler, execute = approval
    bind_presented_proposal(handler, state)
    handler.conversation_id = "chat-b"
    assert "核对以上方案" in review_presented_proposal(handler, state, "同意")
    execute.assert_not_called()


def test_multiple_plans_cannot_be_approved_as_one(approval):
    state, handler, execute = approval
    state["pending_execution_policies"] = {
        "second": {"agent_proposal": {"recommended_action": "generate_branches"}}
    }
    assert "没有唯一" in review_presented_proposal(handler, state, "同意")
    execute.assert_not_called()


def test_switch_prompts_offer_chat_approval_and_explain_scope(approval):
    state, _, _ = approval
    handoff = state["remote_finetune_jobs"]["job"]["training_handoff"]
    saved_reply = pending_plan_reply(state)
    training_reply = concise_training_reply(handoff, state)
    for reply in (saved_reply, training_reply):
        assert "同意" in reply and "拒绝" in reply
        assert "刷新" in reply
    assert "/phase/approval" in saved_reply
    assert "/phase/approval" not in training_reply

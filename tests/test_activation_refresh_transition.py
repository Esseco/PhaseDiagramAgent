"""Synthetic model activation -> separate refresh approval -> portable inputs."""

from copy import deepcopy
import json
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from phase_agent.graphs.dialogue.nodes import present
from phase_agent.graphs.dialogue.support import _load_binding, review_presented_proposal
from phase_agent.runtime.chat_review import review_pending
from phase_agent.runtime.review_requests import pending_reviews
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
from tests.test_model_refresh_workflow import setup


def fixture(tmp_path, monkeypatch, *, fail=False):
    refreshed, session, context, registry = setup(tmp_path)
    state = deepcopy(refreshed)
    state.pop("model_refresh")
    state["active_model_version"] = "old"
    state["active_model"] = {"version": "old"}
    state["remote_finetune_jobs"] = {
        "job": {
            "training_handoff": {
                "stage": "awaiting_activation_approval",
                "candidate_model_version": "new",
                "direction_proposal": {"direction": "activate", "reason": "训练已审阅"},
            }
        }
    }
    state["candidate_models"] = {
        "new": {
            "status": "validated_candidate",
            "old_model_version": "old",
            "agent_review": {"choice": "activate"},
            "model": {"version": "new"},
        }
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    handler = SimpleNamespace(
        state_path=path, conversation_id="a", lock=RLock(), workflow_kwargs={}
    )

    def switch(message, facts, **kwargs):
        if message.startswith("拒绝"):
            return {"state": facts, "reason": "已拒绝"}
        current = deepcopy(refreshed)
        return {"state": current, "reason": "已激活"}

    monkeypatch.setattr(
        "phase_agent.tools.local.review_candidate_command.review_candidate_command", switch
    )

    def workflow(invocation, feedback, message):
        assert json.loads(path.read_text(encoding="utf-8"))["active_model_version"] == "new"
        if fail:
            raise RuntimeError("preview unavailable")
        current = json.loads(path.read_text(encoding="utf-8"))
        result = run_model_refresh_gate(
            current, session, registry=registry, context=context, human_feedback=feedback
        )
        path.write_text(json.dumps(result["state"]), encoding="utf-8")
        return result

    handler._run = Mock(side_effect=workflow)
    handler.review_pending = lambda *args, **kwargs: review_pending(
        handler, *args, is_sensitive=lambda p: False, **kwargs
    )
    return state, handler


def display(state, handler):
    runtime = SimpleNamespace(context=SimpleNamespace(handler=handler))
    return present(
        {
            "result": {
                "status": "training_handoff",
                "state": state,
                "training_handoffs": [],
                "reason": "是否切换新模型？同意/拒绝",
            }
        },
        runtime,
    )


def test_displayed_training_handoff_binds_switch(tmp_path, monkeypatch):
    state, handler = fixture(tmp_path, monkeypatch)
    display(state, handler)
    assert _load_binding(handler)["plan_id"] == "review:model_activation:job"
    handler._run.assert_not_called()


def test_first_chat_approval_switches_and_presents_refresh_without_executing(tmp_path, monkeypatch):
    state, handler = fixture(tmp_path, monkeypatch)
    display(state, handler)
    reply = review_presented_proposal(handler, state, "同意")
    assert "新模型已切换" in reply
    assert "是否批准生成刷新文件" in reply
    assert "继续" not in reply
    assert len(handler._run.call_args_list) == 1
    assert handler._run.call_args.args[1] is None
    saved = json.loads(handler.state_path.read_text(encoding="utf-8"))
    assert saved["active_model_version"] == "new"
    assert saved["tasks"] == []
    assert not (tmp_path / "submissions").exists()
    assert _load_binding(handler)["plan_id"] == pending_reviews(saved)[0]["plan_id"]
    result = review_presented_proposal(handler, saved, "同意")
    assert result
    saved = json.loads(handler.state_path.read_text(encoding="utf-8"))
    assert len(saved["tasks"]) == 1
    assert saved["model_refresh"]["status"] == "waiting_results"
    assert handler._run.call_args.args[1]["decision"] == "approve"
    assert (tmp_path / "submissions").exists()


def test_control_approval_also_presents_refresh(tmp_path, monkeypatch):
    state, handler = fixture(tmp_path, monkeypatch)
    row = pending_reviews(state)[0]
    outcome = handler.review_pending(
        row["plan_id"],
        "confirm_sensitive",
        expected_state_version=row["state_version"],
        expected_proposal_hash=row["proposal_hash"],
    )
    assert outcome["status"] == "awaiting_approval"
    assert "是否批准生成刷新文件" in outcome["reason"]
    assert handler._run.call_args.args[1] is None
    assert not (tmp_path / "submissions").exists()


def test_rejection_does_not_propose_refresh(tmp_path, monkeypatch):
    state, handler = fixture(tmp_path, monkeypatch)
    display(state, handler)
    review_presented_proposal(handler, state, "拒绝")
    handler._run.assert_not_called()


def test_refresh_failure_reports_successful_activation_without_replay(tmp_path, monkeypatch):
    state, handler = fixture(tmp_path, monkeypatch, fail=True)
    display(state, handler)
    reply = review_presented_proposal(handler, state, "同意")
    assert "新模型已切换" in reply and "preview unavailable" in reply
    assert (
        json.loads(handler.state_path.read_text(encoding="utf-8"))["active_model_version"] == "new"
    )
    assert handler._run.call_count == 1
    assert not (tmp_path / "submissions").exists()


def test_explicit_switch_command_also_proposes_refresh(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.maintenance.inspection import handle

    state, handler = fixture(tmp_path, monkeypatch)
    handler.config_delegate = None
    reply = handle(handler, "激活候选 new 原因：已审阅", state, [], "a")
    assert "是否批准生成刷新文件" in reply
    assert handler._run.call_args.args[1] is None


def test_refresh_card_preserves_scope_and_budget(tmp_path, monkeypatch):
    from phase_agent.runtime.concise_dialogue import approval_reply
    from phase_agent.runtime.review_presentation import review_card

    state, session, context, registry = setup(tmp_path)
    response = run_model_refresh_gate(state, session, registry=registry, context=context)
    proposal = response["agent_proposal"]
    reply = approval_reply(proposal, response["state"])
    card = review_card("p", {"agent_proposal": proposal})
    assert "生成新模型刷新文件" in card["title"]
    assert "是否批准生成刷新文件" in reply
    assert "弛豫 0" in reply and "单点 1" in reply
    assert "成本" in reply and "提交" in reply and "branch" in reply


def test_studio_and_short_reply_keep_refresh_approval(tmp_path, monkeypatch):
    from phase_agent.runtime.studio_reply_presentation import format_turn_reply
    from phase_agent.runtime.interaction_reports import short_reply
    from phase_agent.runtime.concise_dialogue import concise_body

    state, handler = fixture(tmp_path, monkeypatch)
    display(state, handler)
    reply = review_presented_proposal(handler, state, "同意")
    saved = json.loads(handler.state_path.read_text(encoding="utf-8"))
    visible = short_reply(concise_body(format_turn_reply(reply, saved), saved))
    assert "是否批准生成刷新文件" in visible
    assert "同意" in visible and "拒绝" in visible


@pytest.mark.parametrize(
    "active,status",
    [("new", "approval_required"), ("old", "approval_required"), ("new", "waiting_results")],
)
def test_no_replay_for_unchanged_model_or_existing_refresh(tmp_path, active, status):
    from phase_agent.runtime.model_activation_followup import propose_refresh_after_activation

    handler = SimpleNamespace(_run=Mock())
    result = {
        "state": {
            "active_model_version": active,
            "model_refresh": {"status": status, "new_model_version": "new"},
        }
    }
    previous = active if status == "approval_required" else "old"
    assert propose_refresh_after_activation(handler, result, previous) is None
    handler._run.assert_not_called()

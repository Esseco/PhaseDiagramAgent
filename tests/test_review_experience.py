import json
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import pytest

from phase_agent.runtime.review_presentation import proposal_changes, review_card
from phase_agent.runtime.chat_review import review_pending
from phase_agent.runtime.scientific_progress import scientific_progress
from phase_agent.tools.policy.file_approval import proposal_hash
from phase_agent.tools.step_runner.build_status_summary import build_status_summary


def proposal(budget=1):
    return {
        "recommended_action": "generate_branches",
        "reason": "增加覆盖",
        "expected_purpose": "生成结构",
        "raw_action": {
            "tool": "generate_branches",
            "parameters": {"total_quota": 300},
            "target_ids": [],
            "budget": budget,
        },
    }


def test_card_and_diff_preserve_source_and_hash():
    record = {
        "agent_proposal": proposal(2),
        "revision": 1,
        "feedback_history": [{"prior_proposal": proposal(1)}],
    }
    before = deepcopy(record)
    card = review_card("p1", record)
    assert card["changes"] == [{"field": "budget", "before": 1, "after": 2}]
    assert card["proposal_hash"] == proposal_hash(record["agent_proposal"])
    assert record == before
    assert "后续新动作" in card["approval_boundary"]
    assert proposal_changes({"agent_proposal": proposal()}) == []


def test_modify_request_validates_identity_and_is_never_approval(tmp_path):
    path = tmp_path / "state.json"
    state = {"pending_execution_policies": {"p1": {"agent_proposal": proposal()}}}
    path.write_text(json.dumps(state))
    calls = []
    handler = SimpleNamespace(
        state_path=path,
        lock=RLock(),
        _run=lambda *args: calls.append(args) or {"status": "awaiting_approval"},
    )
    args = {
        "expected_state_version": build_status_summary(state, config_version=None)["summary_id"],
        "expected_proposal_hash": proposal_hash(proposal()),
        "is_sensitive": lambda p: False,
    }
    result = review_pending(handler, "p1", "modify", comment="改为200个", **args)
    assert result["status"] == "awaiting_approval"
    assert calls[0][1] == {"decision": "comment", "comment": "改为200个"}
    for change, error in [
        ({"comment": ""}, "修改要求"),
        ({"expected_proposal_hash": "stale"}, "proposal_hash_mismatch"),
    ]:
        with pytest.raises(ValueError, match=error):
            review_pending(handler, "p1", "modify", **{**args, "comment": "改方案", **change})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "status,kind",
    [
        ("pending", "external_results"),
        ("submitted", "external_results"),
        ("running", "task_running"),
        ("completed", "ready"),
    ],
)
def test_wait_status_uses_task_facts(tmp_path, status, kind):
    config = tmp_path / "agent_runtime.json"
    config.write_text('{"state_path":"state.json"}')
    (tmp_path / "state.json").write_text(
        json.dumps({"tasks": [{"task_id": "t1", "status": status}]})
    )
    progress = scientific_progress(config)
    assert progress["waiting_state"]["kind"] == kind
    assert "聊天" not in progress["summary"]


def test_pending_card_has_human_wait_and_chat_shows_revision(tmp_path):
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    config = tmp_path / "agent_runtime.json"
    config.write_text('{"state_path":"state.json"}')
    record = {
        "revision": 1,
        "agent_proposal": proposal(2),
        "feedback_history": [{"prior_proposal": proposal(1)}],
    }
    state = {"pending_execution_policies": {"p1": record}}
    (tmp_path / "state.json").write_text(json.dumps(state))
    progress = scientific_progress(config)
    assert progress["waiting_state"]["kind"] == "human_review"
    assert progress["plan_cards"][0]["revision"] == 1
    text = format_workflow_reply(
        {"status": "awaiting_approval", "agent_proposal": record["agent_proposal"], "state": state},
        tmp_path / "state.json",
    )
    assert "修订 1" in text and "budget：1 → 2" in text
    assert "旧版批准不适用于新版方案" in text

"""Approval resumes without retaining redundant full project snapshots."""

import gc
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import patch

from phase_agent.runtime.chat_review import review_pending
from phase_agent.policies.file_approval import proposal_hash


def test_review_releases_both_validation_snapshots_before_graph():
    class State(dict):
        pass

    refs = []
    proposal = {"raw_action": {"tool": "pause_search", "decision_source": "codex"}}

    def read(*_args):
        state = State(pending_execution_policies={"p": {"agent_proposal": proposal}})
        refs.append(weakref.ref(state))
        return state

    def invoke(*_args):
        gc.collect()
        assert len(refs) == 2 and all(reference() is None for reference in refs)
        return {"status": "completed"}

    handler = SimpleNamespace(
        lock=threading.RLock(), state_path="unused", workflow_kwargs={}, _run=invoke
    )
    with (
        patch("phase_agent.runtime.chat_review.read_json", side_effect=read),
        patch(
            "phase_agent.runtime.chat_review.build_status_summary",
            new=lambda *a, **k: {"summary_id": "v"},
        ),
        patch(
            "phase_agent.evidence.project_authority.project_identity",
            new=lambda state: {"model_consistent": True},
        ),
        patch(
            "phase_agent.policies.recovery_transition.post_recovery_tools", new=lambda state: None
        ),
        patch(
            "phase_agent.policies.recovery_transition.legacy_post_recovery_plan_ids",
            new=lambda state: set(),
        ),
    ):
        result = review_pending(
            handler,
            "p",
            "approve",
            expected_state_version="v",
            expected_proposal_hash=proposal_hash(proposal),
            is_sensitive=lambda _: False,
        )
    assert result["status"] == "completed"

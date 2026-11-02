from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.runtime.codex.context import ensure_codex_post_dft_review
from phase_agent.runtime.chat_execution import run_workflow_turn


def _snapshot():
    return {
        "context_version": "V1",
        "context_digest": "D1",
        "allowed_tools": ["pause_search"],
        "decision_context": {},
        "state": {},
    }


def _control(run):
    temp = TemporaryDirectory()
    handler = SimpleNamespace(
        state_path=Path(temp.name) / "state.json",
        workflow_kwargs={},
        lock=__import__("threading").RLock(),
        _run=run,
    )
    control = LocalAgentControl(handler)
    control._test_tempdir = temp
    return control


def test_structured_codex_action_skips_summary_model(tmp_path):
    action = {"tool": "pause_search", "reason": "reuse current agent decision"}

    class StructuredCodex:
        _codex_structured = True
        _decision_source = "codex"

        def __call__(self, _payload):
            return action

    def workflow(**kwargs):
        return kwargs["agent_client"]({"allowed_tools": ["pause_search"]})

    handler = SimpleNamespace(
        workflow=workflow,
        workflow_kwargs={"agent_client": StructuredCodex()},
        unified_dialogue=True,
        decision_backend=None,
        state_path=tmp_path / "state.json",
    )
    handler.state_path.write_text("{}", encoding="utf-8")

    with patch(
        "phase_agent.evidence.requests.request_summary_first",
        side_effect=AssertionError("structured Codex must not call summary model"),
    ):
        result = run_workflow_turn(handler, "invocation", None, "继续")

    assert result == action


def test_codex_proposal_rejects_stale_context_before_run():
    called = []

    def run(*args):
        called.append(args)

    control = _control(run)
    with patch("phase_agent.runtime.codex.context.build_codex_context", return_value=_snapshot()):
        with pytest.raises(ValueError, match="codex_context_stale"):
            control.propose_codex(
                {"tool": "pause_search"}, context_version="OLD", context_digest="D1"
            )
    assert called == []


def test_codex_proposal_rejects_digest_mismatch_before_run():
    called = []

    def run(*args):
        called.append(args)

    control = _control(run)
    with patch("phase_agent.runtime.codex.context.build_codex_context", return_value=_snapshot()):
        with pytest.raises(ValueError, match="codex_context_digest_mismatch"):
            control.propose_codex(
                {"tool": "pause_search"}, context_version="V1", context_digest="OLD"
            )
    assert called == []


def test_codex_scientific_proposal_adds_required_round_budget_review():
    seen = []

    def run(*args):
        action = control_handler.workflow_kwargs["agent_client"]({})
        seen.append(action)
        return {"status": "awaiting_approval", "record_id": "p1"}

    control_handler = SimpleNamespace(
        state_path=Path(TemporaryDirectory().name) / "state.json",
        workflow_kwargs={},
        lock=__import__("threading").RLock(),
        _run=run,
    )
    control = LocalAgentControl(control_handler)
    snapshot = _snapshot()
    snapshot["allowed_tools"] = ["select_dft_candidates"]
    snapshot["decision_context"] = {"round_budget_evidence": {"required": True}}
    with patch("phase_agent.runtime.codex.context.build_codex_context", return_value=snapshot):
        control.propose_codex(
            {
                "tool": "select_dft_candidates",
                "parameters": {"decisions": []},
                "budget": 12,
            },
            context_version="V1",
            context_digest="D1",
        )

    review = seen[0]["round_budget_review"]
    assert review["choice"] == "supplement_dft"
    assert {row["path"] for row in review["alternatives"]} == {
        "supplement_dft",
        "new_search",
    }


def test_codex_scientific_proposal_adds_post_dft_review_without_model_repair():
    action = {"tool": "select_dft_candidates", "parameters": {"decisions": []}}
    result = ensure_codex_post_dft_review(
        action, {"round_budget_review_required": True}
    )
    review = result["post_dft_review"]
    assert review["choice"] == "supplement_dft"
    assert review["stop_status"] == "continue"
    assert review["finetune_recommendation"] in {"defer", "insufficient_evidence"}


def test_codex_context_reuses_snapshot_until_state_file_changes(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    handler = SimpleNamespace(state_path=path, lock=__import__("threading").RLock())
    control = LocalAgentControl(handler)
    snapshot = _snapshot()
    with patch("phase_agent.runtime.codex.context.build_codex_context", return_value=snapshot) as build:
        assert control.codex_context() == snapshot
        assert control.codex_context() == snapshot
    assert build.call_count == 1

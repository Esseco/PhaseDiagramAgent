from types import SimpleNamespace
from phase_agent.graphs.dialogue.support import _save_binding, _load_binding


def test_binding_survives_handler_recreation_and_isolates_threads(tmp_path):
    a = SimpleNamespace(state_path=tmp_path / "state.json", conversation_id="a")
    binding = {"plan_id": "p", "state_version": "v", "proposal_hash": "h"}
    _save_binding(a, binding)
    restored = SimpleNamespace(state_path=a.state_path, conversation_id="a")
    assert _load_binding(restored) == binding
    other = SimpleNamespace(state_path=a.state_path, conversation_id="b")
    assert _load_binding(other) is None
    _save_binding(restored, None)
    assert _load_binding(a) is None


def test_second_approval_calls_review_after_handler_recreation(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.support import review_presented_proposal
    from unittest.mock import Mock
    state = {"pending_execution_policies": {"p": {"agent_proposal": {"recommended_action": "generate_branches"}}}}
    from phase_agent.tools.step_runner.file_protocol import write_json
    write_json(tmp_path / "state.json", state)
    monkeypatch.setattr("phase_agent.runtime.chat_approval_rules.is_sensitive_proposal", lambda p: False)
    monkeypatch.setattr("phase_agent.runtime.plan_queries.pending_plan_reply", lambda state: "saved plan")
    monkeypatch.setattr("phase_agent.runtime.workflow_reply_presentation.format_workflow_reply", lambda *a: "executed once")
    first = SimpleNamespace(state_path=tmp_path / "state.json", conversation_id="a", review_pending=Mock())
    assert "saved plan" in review_presented_proposal(first, state, "同意")
    first.review_pending.assert_not_called()
    binding = _load_binding(first)
    second = SimpleNamespace(state_path=first.state_path, conversation_id="a", review_pending=Mock(return_value={"result": {}}))
    assert review_presented_proposal(second, state, "同意") == "executed once"
    second.review_pending.assert_called_once_with("p", "approve", expected_state_version=binding["state_version"], expected_proposal_hash=binding["proposal_hash"], comment="同意")
    assert _load_binding(first) is None


def test_post_recovery_dft_approval_rebinds_current_plan_once(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.support import review_presented_proposal
    from unittest.mock import Mock
    from phase_agent.tools.step_runner.file_protocol import write_json

    state = {
        "post_recovery_next_step_pending": True,
        "confirmed_config_version": "v",
        "active_model_version": "m",
        "confirmed_config": {},
        "pending_execution_policies": {
            "p": {
                "agent_proposal": {
                    "recommended_action": "select_dft_candidates",
                    "raw_action": {
                        "tool": "select_dft_candidates",
                        "parameters": {"prepare_inputs_after_approval": True},
                    },
                }
            }
        },
    }
    write_json(tmp_path / "state.json", state)
    monkeypatch.setattr("phase_agent.runtime.chat_approval_rules.is_sensitive_proposal", lambda p: False)
    handler = SimpleNamespace(
        state_path=tmp_path / "state.json",
        conversation_id="fresh-conversation",
        review_pending=Mock(return_value={"result": {}}),
    )

    review_presented_proposal(handler, state, "同意")

    handler.review_pending.assert_called_once()
    assert handler.review_pending.call_args.args[:2] == ("p", "approve")

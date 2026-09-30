import json
from unittest.mock import Mock, patch
from run.open_webui_api import RunWorkflowChatHandler
from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.policy.file_approval import proposal_hash


def test_reject_sensitive_plan_needs_no_sensitive_confirmation(tmp_path):
    proposal = {"recommended_action": "select_dft_candidates", "raw_action": {"tool": "select_dft_candidates"}}
    state = {"pending_execution_policies": {"p": {"agent_proposal": proposal}},
             "pending_mc_regeneration": {"directory": "old-mc"}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    handler = RunWorkflowChatHandler({"state_path": str(path)})
    handler._run = Mock(return_value={"status": "rejected_by_user"})
    with patch("run.open_webui_api._is_sensitive_proposal", return_value=True):
        handler.review_pending("p", "reject",
            expected_state_version=build_status_summary(state, config_version=None)["summary_id"],
            expected_proposal_hash=proposal_hash(proposal), comment="拒绝")
    assert handler._run.call_args.args[1]["decision"] == "reject"


def test_chat_reject_targets_dft_before_old_mc_plan(tmp_path):
    proposal = {"recommended_action": "select_dft_candidates", "raw_action": {"tool": "select_dft_candidates"}}
    state = {"pending_execution_policies": {"p": {"agent_proposal": proposal}},
             "pending_mc_regeneration": {"directory": "old-mc"}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    handler = RunWorkflowChatHandler({"state_path": str(path)})
    handler.review_pending = Mock(return_value={"result": {"status": "rejected_by_user"}})
    reply = handler([{"role": "user", "content": "拒绝"}])
    assert "取消" in reply
    assert handler.review_pending.call_args.args[:2] == ("p", "reject")
    assert json.loads(path.read_text())["pending_mc_regeneration"]

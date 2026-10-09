from analysis_layer.cost.estimate_proposal_runtime import estimate_proposal_runtime
from execution_layer.policy.execution_policy import build_agent_proposal
from run.workflow_reply_presentation import format_workflow_reply


def test_dft_runtime_uses_real_candidate_and_history():
    action = {"tool": "select_dft_candidates", "parameters": {"decisions": [
        {"candidate_id": "a", "action": "DFT_SINGLE_POINT"}]}, "reason": "test"}
    state = {"qbc_candidates": [{"candidate_id": "a", "atom_count": 40}],
             "cost_history": [{"stage": "dft_single_point", "status": "completed", "atom_count": 40,
                 "runtime_observation": {"elapsed_seconds": 3600, "backend": "vasp",
                                         "hardware": "cpu", "cpu_count": 32}}]}
    report = estimate_proposal_runtime(action, state)
    assert report["serial_seconds"] == 3600
    proposal = build_agent_proposal(action, {}, runtime_state=state)
    proposal["action_parameters"]["dft_input_preview"] = {"task_count": 1,
        "selected_structures": [{"candidate_id": "a"}]}
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": proposal}, "state.json")
    assert "串行累计约 1 小时" in text and "不含排队" in text


def test_missing_sample_never_turns_cost_into_hours():
    action = {"tool": "allocate_mc_bohb", "budget": 5000, "parameters": {}, "reason": "test"}
    report = estimate_proposal_runtime(action, {})
    assert report["serial_seconds"] is None
    proposal = build_agent_proposal(action, {})
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": proposal}, "state.json")
    assert "暂无法估算" in text


def test_partial_estimation_does_not_report_complete_batch_time():
    action = {"tool": "select_dft_candidates", "parameters": {"decisions": [
        {"candidate_id": "unknown", "action": "DFT_RELAX"}]}}
    assert estimate_proposal_runtime(action, {})["status"] == "unavailable"

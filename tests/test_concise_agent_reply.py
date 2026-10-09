from run.agent_api import format_workflow_reply


def test_wait_omits_empty_diagnostics_and_task_list():
    result = {"status": "awaiting_manual_submission", "manual_wait": {
        "waiting_by_stage": {"deep_search": 119}, "waiting_task_count": 119,
        "upload_root": "MC-round-0002", "task_directories": ["task1", "task2"],
        "result_collection": {"considered_count": 119, "missing_result_count": 119}}}
    text = format_workflow_reply(result, "state.json")
    assert "MC 119 个" in text
    assert "扫描" not in text and "task1" not in text
    assert "results" in text and "继续" in text


def test_approval_preserves_budget_and_confirmation():
    result = {"status": "awaiting_approval", "events": [{"approval_files": {"directory": "approval-record"}}], "agent_proposal": {
        "recommended_action": "allocate_mc_bohb", "expected_purpose": "第二轮 MC，58 个 branch",
        "estimated_cost": {"estimated_total_cost": 300},
        "action_parameters": {"mc_budget": 5000, "budget_preview": {"requested_steps": 1740}}}}
    text = format_workflow_reply(result, "state.json")
    assert "300" in text and "1740" in text and "同意" in text
    assert "完整参数与依据" not in text
    assert "完整参数与依据" in format_workflow_reply(result, "state.json", verbose=True)


def test_failure_preserves_reason():
    assert format_workflow_reply({"status": "failed", "reason": "结构文件不存在"}, "state.json") == "本轮失败：结构文件不存在"

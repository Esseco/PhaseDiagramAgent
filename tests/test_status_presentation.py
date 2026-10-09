from copy import deepcopy
from orchestration.studio_chat_graph import build_studio_graph
from run.status_presentation import format_progress
from run.response_preferences import response_detail, detailed_response


def test_epoch_action_and_waived_results_are_explicit():
    state = {"active_model_version": "mace-test",
        "upload_layout": {"model_rounds": {"mace-test": 1}},
        "pending_execution_policies": {"action-1": {"agent_proposal": {
            "raw_action": {"tool": "update_mlip"}}}},
        "tasks": [{"stage": "deep_search", "status": "completed"},
                  {"stage": "dft_single_point", "status": "pending", "recovery_wait_waived": True}],
        "budget_remaining": 12314.742357}
    before = deepcopy(state)
    text = format_progress(state)
    assert "epoch0（mace-test）" in text
    assert "微调训练输入方案待确认" in text
    assert "生成微调训练输入" in text
    assert "下一步" in text and "同意" in text
    assert "仍需回收 0；未回传但不再等待 1" in text
    assert "12314.7" in text
    assert state == before


def test_unknown_epoch_is_not_guessed():
    assert "模型轮次未记录" in format_progress({"active_model_version": "test"})


def test_prepared_training_reports_submission_not_another_proposal():
    text = format_progress({"active_model_version": "test", "remote_finetune_jobs": {
        "job": {"original_model_version": "test", "status": "inputs_prepared",
                "directory": "synthetic-training", "activated": False}}})
    assert "微调输入已准备" in text
    assert "在inputs内提交 GPU.sh" in text
    assert "synthetic-training" in text
    assert "由 Agent 检查已有结果" not in text


def test_studio_context_changes_display_and_does_not_leak(tmp_path):
    state = {"active_model_version": "test", "confirmed_config_version": "config-test",
             "phase_diagrams": {"mlip": {"version": "h1", "csv_path": "synthetic.csv"}}}
    graph = build_studio_graph(lambda *_: format_progress(state), receipt_path=tmp_path / "gateway.json")
    config = {"configurable": {"thread_id": "display-test"}}
    def invoke(identity, context):
        return graph.invoke({"messages": [{"role": "user", "id": identity, "content": "现在什么进度"}]},
                            config, context=context)["messages"][-1].content
    brief = invoke("brief", {"response_detail": "brief"})
    detailed = invoke("detailed", {"response_detail": "detailed"})
    assert "config-test" not in brief and "config-test" in detailed
    assert "synthetic.csv" in detailed and "synthetic.csv" not in brief
    assert not detailed_response()
    assert "config-test" not in invoke("default", None)


def test_display_context_resets_after_failure():
    try:
        with response_detail("detailed"):
            assert detailed_response()
            raise RuntimeError("synthetic")
    except RuntimeError:
        pass
    assert not detailed_response()

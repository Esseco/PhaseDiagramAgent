from run.status_presentation import format_progress


def test_recovered_results_take_priority_over_training_proposal():
    state = {"active_model_version": "old", "remote_finetune_jobs": {
        "job": {"status": "results_received", "activated": False,
                "returned_results": {"issues": ["com_1 缺少有效模型SHA256"],
                                     "kfold_metrics": {"energy_MAE": 1}}}},
        "pending_execution_policies": {"proposal": {"agent_proposal": {
            "raw_action": {"tool": "update_mlip"}}}}}
    result = format_progress(state)
    assert "交叉验证指标已登记" in result
    assert "com_1 缺少有效模型SHA256" in result
    assert "无需重复生成训练输入" in result
    assert "回复“同意”" not in result
    assert "生成微调训练输入。" not in result
    assert state["pending_execution_policies"]

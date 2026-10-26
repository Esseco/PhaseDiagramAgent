from phase_agent.runtime.workflow_reply_presentation import _current_completed_mc_recovery_line
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def test_next_relax_mc_plan_explains_previous_batch_was_recovered():
    state = {
        "active_model_version": "model-new",
        "relax_mc_batches": {
            "batch": {
                "model_version": "model-new",
                "status": "completed",
                "import_checksum": "sha256",
                "result_summary": {"task_count": 2, "failed_task_ids": []},
            }
        },
        "tasks": [
            {"relax_mc_bundle_id": "batch", "status": "completed"},
            {"relax_mc_bundle_id": "batch", "status": "completed"},
        ],
    }

    assert "上一批 Relax+MC 已回收并核验 2/2 个任务" in (
        _current_completed_mc_recovery_line(state) or ""
    )


def test_relax_mc_approval_explains_new_branch_reason_and_difference():
    state = {
        "active_model_version": "model-new",
        "relax_mc_batches": {
            "batch": {
                "model_version": "model-new",
                "status": "completed",
                "import_checksum": "sha256",
                "branch_ids": ["B-old"],
                "result_summary": {"task_count": 1, "failed_task_ids": []},
            }
        },
        "tasks": [{"relax_mc_bundle_id": "batch", "status": "completed"}],
    }
    params = {
        "mode": "relax_mc_inputs",
        "relax_mc_preview": {"workload": {"relax_structure_tasks": 1, "mc_task_upper": [1, 1, 1], "gpu_jobs_upper_by_wave": [1, 1, 1], "max_concurrent_jobs": 1}},
    }
    proposal = {
        "recommended_action": "prepare_local_batch_files",
        "expected_purpose": "new batch",
        "estimated_cost": {"estimated_total_cost": 1},
        "raw_action": {"tool": "prepare_local_batch_files", "parameters": params, "target_ids": ["B-new"]},
        "decision_summary": {
            "rationale": "旧 branch 已有证据，新 branch 尚无当前模型 Relax/MC 证据",
            "expected_outcome": "比较新 branch 的 Ehull",
        },
    }
    text = format_workflow_reply({"status": "awaiting_approval", "state": state, "agent_proposal": proposal}, None)
    assert "上一批 Relax+MC 已回收并核验 1/1 个任务" in text
    assert "本批选用这些新增 branch 的原因" in text
    assert "本方案只准备已登记 branch 的 Relax+MC 输入，不生成新 branch" in text
    assert "本批预期" in text

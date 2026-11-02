"""An approved prepared input batch is a delivery receipt, not another proposal."""

from copy import deepcopy

import pytest

from phase_agent.runtime.chat_state_presentation import brief_chat_state, format_epoch_reply
from phase_agent.runtime.concise_dialogue import concise_body
from phase_agent.runtime.interaction_reports import short_reply
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from tests.test_dft_recovery_reply_metrics import next_proposal
from tests.test_partial_recovery_choice import _dft_state
from tests.test_post_dft_review import review


def prepared_state(tmp_path, status="inputs_prepared"):
    state = _dft_state()
    # Keep a completed DFT batch with failures as earlier scientific evidence.
    state["remote_finetune_jobs"] = {
        "training-current": {
            "status": status,
            "directory": str(tmp_path / "training-current"),
            "original_model_version": "m1",
            "activated": False,
            "handoff": {
                "folder": str(tmp_path / "Finetune_Batch01_input"),
                "return_directory": str(tmp_path / "Finetune_Batch01_output"),
                "submit_script": str(tmp_path / "Finetune_Batch01_input/submit_all.sh"),
                "command": "sbatch submit_all.sh",
                "layout_version": 4,
            },
        }
    }
    return state


@pytest.mark.parametrize("status", ["inputs_prepared", "submitted", "running"])
def test_training_delivery_survives_the_final_chat_pipeline(tmp_path, status):
    state = prepared_state(tmp_path, status)
    proposal = next_proposal("update_mlip")
    proposal["raw_action"]["post_dft_review"] = review()
    proposal["raw_action"]["post_dft_review"]["choice"] = "finetune"
    result = {
        "status": "awaiting_remote_training",
        "state": state,
        "events": [{"agent_proposal": proposal, "final_action": proposal["raw_action"]}],
    }
    before = deepcopy(state)
    reply = format_workflow_reply(result, tmp_path / "state.json")
    visible = short_reply(
        format_epoch_reply(format_turn_reply(concise_body(reply, state), state), state)
    )
    transfer = state["remote_finetune_jobs"]["training-current"]["handoff"]

    assert "微调" in brief_chat_state(state) and "DFT" not in brief_chat_state(state)
    assert transfer["folder"] in visible and transfer["return_directory"] in visible
    assert "本轮总结：" not in visible and "首选建议：" not in visible
    if status == "inputs_prepared":
        assert "结果：微调输入已生成" in visible
        assert "上传" in visible and "提交" in visible and "sbatch submit_all.sh" in visible
        assert "等待当前任务完成" not in visible
    else:
        assert "已提交" in visible or "运行中" in visible
        assert "回传" in visible and "再次提交" in visible
    assert state == before


def test_status_query_prefers_current_training_delivery_over_old_dft_failures(tmp_path):
    from phase_agent.runtime.status_presentation import format_progress

    state = prepared_state(tmp_path)
    reply = format_progress(state)
    transfer = state["remote_finetune_jobs"]["training-current"]["handoff"]
    assert "微调输入已生成" in reply
    assert transfer["folder"] in reply and transfer["return_directory"] in reply
    assert "上传" in reply and "sbatch submit_all.sh" in reply
    assert "回复“继续”使用已回收子集" not in reply


def test_actual_approval_prepares_training_files_and_repeat_does_not_rebuild(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    import zipfile

    from phase_agent.configuration.session.apply_config_revision import apply_config_revision
    from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json
    from tests.test_dft_comparison_csv import add_result
    from tests.test_three_scientific_decisions import review as scientific_review
    from tests.test_unified_react_dialogue import make_handler

    calls = []
    action = {
        "tool": "update_mlip",
        "task_key": "approve-training-inputs",
        "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
        "reason": "Synthetic error evidence supports preparing training inputs",
        "expected_purpose": "Prepare training inputs only",
        "post_dft_review": review(),
        "scientific_decision": scientific_review("round_direction"),
    }
    action["post_dft_review"].update(
        choice="finetune",
        finetune_recommendation="now",
        finetune_assessment="Synthetic qualified data and errors support finetuning",
        search_assessment="Defer search until this model has been improved",
    )

    def model(request):
        calls.append(
            {
                "mode": request.get("mode"),
                "user_instruction": request.get("user_instruction"),
                "keys": sorted(request),
                "feedback": (request.get("decision_context") or {}).get(
                    "proposal_validation_feedback"
                ),
            }
        )
        return deepcopy(action)

    def forbidden(*args, **kwargs):
        pytest.fail("Preparing approved inputs must not run training or submit jobs")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    handler = make_handler(tmp_path, model)
    session = confirm_config_snapshot(
        apply_config_revision(
            create_config_draft(
                handler.workflow_kwargs["config_session"]["confirmed_snapshot"]["config"]
            ),
            {
                "mlip.name": "m1",
                "mlip.version": "m1",
                "mlip.model_path": "/remote/models/m1.model",
                "mlip_finetune.labels.include_stress": False,
                "python_environments.remote_mlip": "mace",
            },
        ),
        user_confirmed=True,
    )
    handler.workflow_kwargs["config_session"] = session
    handler.workflow_kwargs["run_config"]["upload_batches_directory"] = str(tmp_path / "uploads")
    snapshot = session["confirmed_snapshot"]
    action["scientific_decision"].update(
        config_version=snapshot["config_version"], model_version="m1"
    )
    state = {
        "active_model_version": "m1",
        "confirmed_config_version": snapshot["config_version"],
        "confirmed_config": snapshot["config"],
        "scientific_decision_policy": {"schema_version": 1},
        "last_trained_dataset_version": "previous-epoch",
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": ["dft_single_point"]},
    }
    for index in range(12):
        add_result(state, task_id=f"dft-{index}")
    for index in range(3):
        task = deepcopy(state["tasks"][0])
        task.update(task_id=f"failed-{index}", status="failed")
        state["tasks"].append(task)
    for task in state["tasks"]:
        task["config_version"] = snapshot["config_version"]
    write_json(handler.state_path, state)

    proposed = handler([{"role": "user", "content": "继续"}], conversation_id="approval-delivery")
    assert "是否同意" in short_reply(proposed)
    count_before_approval = len(calls)
    approved = short_reply(
        handler([{"role": "user", "content": "同意"}], conversation_id="approval-delivery")
    )
    saved = read_json(handler.state_path, {})
    jobs = saved.get("remote_finetune_jobs") or {}
    assert len(jobs) == 1, approved
    job = next(iter(jobs.values()))
    assert job["status"] == "inputs_prepared" and not job["submitted"] and not job["activated"]
    transfer = job["handoff"]
    folder = Path(transfer["folder"])
    archive = Path(transfer["upload_archive"])
    assert folder.is_dir() and Path(transfer["submit_script"]).is_file()
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert "inputs/training_plan.json" in package.namelist()
    assert "结果：微调输入已生成" in approved
    assert str(folder) in approved and transfer["return_directory"] in approved
    assert "上传" in approved and "提交" in approved and "sbatch submit_all.sh" in approved
    assert "9 个独立单 GPU 训练" in approved and "最多 4 个同时运行" in approved
    assert "DFT 本轮已回收" not in approved
    assert len(calls) == count_before_approval, calls
    original = (archive.stat().st_mtime_ns, hashlib.sha256(archive.read_bytes()).hexdigest())

    # Legacy false recovery approvals left this marker even though the input
    # generation had completed and no new result was imported.
    saved["post_recovery_next_step_pending"] = True
    saved["recovery_acknowledgement"] = {"source_stages": []}
    write_json(handler.state_path, saved)

    for message in ("同意", "继续"):
        repeated = handler(
            [{"role": "user", "content": message}], conversation_id="approval-delivery"
        )
        assert str(folder) in repeated and transfer["return_directory"] in repeated
        assert "若已提交" in repeated and "请勿重复提交" in repeated
    assert len(read_json(handler.state_path, {})["remote_finetune_jobs"]) == 1
    assert (
        archive.stat().st_mtime_ns,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    ) == original
    assert len(calls) == count_before_approval

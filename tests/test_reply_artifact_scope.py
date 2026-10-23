"""Files in a stage reply belong to this turn's action, not the latest history."""

from copy import deepcopy
from pathlib import Path
import json

import pytest

from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def history():
    return {
        "qbc_selection_history": [
            {"task_key": "current", "artifacts": {"json": "/private/current-dft.json"}},
            {"task_key": "old", "artifacts": {"json": "/private/old-dft.json"}},
        ],
        "generation_history": [
            {"task_key": "branch", "artifacts": {"json": "/private/branch.json"}}
        ],
    }


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("status", ["training_handoff", "awaiting_remote_training", "paused"])
def test_unrelated_stages_do_not_append_old_scientific_files(tmp_path, verbose, status):
    state = history()
    saved = deepcopy(state)
    reply = format_workflow_reply(
        {"status": status, "state": state, "reason": "微调结果已回收，等待模型激活。"},
        tmp_path / "state.json",
        verbose=verbose,
    )
    assert "old-dft.json" not in reply and "branch.json" not in reply
    assert state == saved
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("nested", [True, False])
def test_current_executed_dft_keeps_only_matching_plan(tmp_path, nested):
    action = {"tool": "select_dft_candidates", "task_key": "current"}
    result = {"status": "completed", "state": history()}
    if nested:
        result["events"] = [{"final_action": action}]
    else:
        result["final_action"] = action
    reply = format_workflow_reply(result, tmp_path / "state.json")
    assert "current-dft.json" in reply and "old-dft.json" not in reply
    assert "branch.json" not in reply


def test_current_branch_keeps_branch_file_without_dft(tmp_path):
    result = {
        "status": "completed",
        "state": history(),
        "events": [{"final_action": {"tool": "generate_branches", "task_key": "branch"}}],
    }
    reply = format_workflow_reply(result, tmp_path / "state.json")
    assert "branch.json" in reply and "dft.json" not in reply


def test_missing_action_identity_does_not_guess_latest_file(tmp_path):
    result = {
        "status": "completed",
        "state": history(),
        "final_action": {"tool": "select_dft_candidates"},
    }
    assert "old-dft.json" not in format_workflow_reply(result, tmp_path / "state.json")


def test_activation_handoff_keeps_current_approval_and_hides_old_dft(tmp_path):
    state = history()
    state.update(
        active_model_version="m1",
        candidate_models={
            "new": {
                "agent_review": {"choice": "activate", "reason": "训练结果已回收，等待人工切换。"}
            }
        },
        remote_finetune_jobs={
            "job": {
                "status": "results_received",
                "activated": False,
                "returned_results": {"status": "completed"},
            }
        },
    )
    saved = deepcopy(state)
    result = {
        "status": "training_handoff",
        "state": state,
        "training_handoffs": [
            {"stage": "awaiting_activation_approval", "candidate_model_version": "new"}
        ],
    }
    from phase_agent.runtime.chat_state_presentation import format_epoch_reply

    reply = format_epoch_reply(format_workflow_reply(result, tmp_path / "state.json"), state)
    assert "微调完成" in reply and "换用新模型" in reply
    assert "同意" in reply and "/phase/approval" not in reply
    assert "DFT 选点及输入任务" not in reply and "old-dft.json" not in reply
    assert state == saved and not list(tmp_path.iterdir())


@pytest.mark.parametrize("project", ["O3-1ele-NaCoO2", "O3-1ele-NaCrO2", "O3-2ele-NaFeMn"])
@pytest.mark.real_project
def test_actual_project_history_does_not_leak_into_training_reply(tmp_path, project):
    path = Path("E:/0-PhaseDiagram") / project / "workflow_state/state.json"
    raw = path.read_bytes()
    state = json.loads(raw.decode("utf-8"))
    before = json.dumps(state, sort_keys=True)
    reply = format_workflow_reply(
        {"status": "training_handoff", "state": state, "reason": "等待模型激活审批。"},
        tmp_path / "state.json",
    )
    assert "DFT 选点及输入任务详情" not in reply
    assert "Branch 及结构路径详情" not in reply
    assert json.dumps(state, sort_keys=True) == before
    assert path.read_bytes() == raw

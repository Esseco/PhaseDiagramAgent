from copy import deepcopy
import pytest
from phase_agent.runtime.response_preferences import response_detail


@pytest.fixture(autouse=True)
def detailed_existing_checks():
    with response_detail("detailed"):
        yield


from phase_agent.decisions.agent.dialogue_contract import dialogue_output_contract
from phase_agent.runtime.configuration_chat import CONFIG_AGENT_SYSTEM_PROMPT
from phase_agent.runtime.status_presentation import format_progress
from phase_agent.runtime.waiting_files_presentation import waiting_file_lines
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def state(tmp_path):
    directory = (
        tmp_path / "epoch1_remote-test" / "Model-refresh-0001_Single-point" / "inputs" / "batch-1"
    )
    directory.mkdir(parents=True)
    script = directory / "GPU.sh"
    script.write_text("synthetic")
    archive = directory.with_name(directory.name + ".upload.zip")
    archive.write_text("synthetic")
    results = directory.parent.parent / "results"
    return {
        "active_model_version": "remote-test",
        "model_refresh": {"status": "waiting_results", "wave": 0, "task_ids": ["refresh-1"]},
        "tasks": [{"task_id": "refresh-1", "stage": "relax_and_feature", "status": "pending"}],
        "slurm_batches": [
            {
                "task_ids": ["refresh-1"],
                "upload_directory": str(directory),
                "script_path": str(script),
                "results_directory": str(results),
                "status": "prepared",
            }
        ],
    }


def test_progress_and_workflow_show_actual_files_and_return_location(tmp_path):
    data = state(tmp_path)
    before = deepcopy(data)
    for text in [
        format_progress(data),
        format_workflow_reply(
            {"status": "tasks_in_progress", "state": data}, "synthetic-state.json"
        ),
    ]:
        assert "重新计算已有结构" in text
        assert "batch-1.upload.zip" in text
        assert data["slurm_batches"][0]["results_directory"] in text
        assert "sbatch GPU.sh" in text
        assert "输入已生成" in text
        assert "保留任务子目录" in text
    assert data == before


def test_only_current_refresh_batches_are_shown(tmp_path):
    data = state(tmp_path)
    data["tasks"].append({"task_id": "old", "stage": "deep_search", "status": "pending"})
    data["slurm_batches"].append({"task_ids": ["old"], "upload_directory": "old-unrelated"})
    assert "old-unrelated" not in "\n".join(waiting_file_lines(data))


def test_missing_archive_uses_recorded_directory_and_missing_paths_are_explicit(tmp_path):
    data = state(tmp_path)
    batch = data["slurm_batches"][0]
    from pathlib import Path

    Path(batch["upload_directory"] + ".upload.zip").unlink()
    assert "提交路径（任务输入目录）" in "\n".join(waiting_file_lines(data))
    batch.pop("script_path")
    batch.pop("results_directory")
    text = "\n".join(waiting_file_lines(data))
    assert "提交脚本未记录" in text and "结果回传位置未记录" in text
    assert "sbatch" not in text


def test_known_handoff_uses_total_script_not_sbatch(tmp_path):
    data = state(tmp_path)
    data["slurm_batches"][0]["handoff"] = {
        "upload_archive": "verified/inputs.zip",
        "submit_script": "verified/submit_all.sh",
        "results_directory": "verified/results",
    }
    text = "\n".join(waiting_file_lines(data))
    assert "sbatch submit_all.sh" in text
    assert "results/transfer/all.results.zip" in text
    assert "verified/results" in text


def test_ordinary_wait_and_completed_tasks(tmp_path):
    data = state(tmp_path)
    data.pop("model_refresh")
    assert "batch-1.upload.zip" in "\n".join(waiting_file_lines(data))
    data["tasks"][0]["status"] = "completed"
    assert waiting_file_lines(data) == []


def test_dialogue_and_configuration_share_actionable_rule():
    from phase_agent.runtime.response_preferences import ACTIONABLE_REPLY_RULE

    rule = dialogue_output_contract()["reply_guidance"]
    assert rule == ACTIONABLE_REPLY_RULE
    assert "操作需要时给一个上传目录、必要命令及回传目录" in rule
    assert rule in CONFIG_AGENT_SYSTEM_PROMPT


def test_brief_mode_has_one_round_folder_and_no_batch_paths(tmp_path):
    data = state(tmp_path)
    second = deepcopy(data["slurm_batches"][0])
    second["upload_directory"] = second["upload_directory"].replace("Single-point", "Relax")
    data["slurm_batches"].append(second)
    folder = tmp_path / "archives/01-upload/epoch1_remote-test_current"
    folder.mkdir(parents=True)
    (folder / "inputs.zip").write_text("synthetic archive")
    (folder / "submit_all.sh").write_text("synthetic entry")
    (folder / "dispatch_tasks.py").write_text("synthetic dispatcher")
    for batch in data["slurm_batches"]:
        batch["handoff"] = {
            "folder": str(folder),
            "upload_archive": str(folder / "inputs.zip"),
            "return_directory": str(tmp_path / "return-results"),
        }
    with response_detail("brief"):
        text = "\n".join(waiting_file_lines(data))
        reply = format_workflow_reply(
            {"status": "tasks_in_progress", "state": data}, "synthetic-state.json"
        )
    for output in [text, reply]:
        assert output.count("提交路径：") == 1
        assert "[打开" not in output and "](" not in output
        assert "epoch1_remote-test" in output
        assert "batch-1" not in output and ".upload.zip" not in output
        assert "results" in output and "继续" in output
        assert len(output.splitlines()) <= 4


def test_brief_missing_handoff_does_not_advertise_internal_epoch(tmp_path):
    data = state(tmp_path)
    with response_detail("brief"):
        text = "\n".join(waiting_file_lines(data))
    assert "提交路径：" not in text
    assert "epoch1_remote-test" not in text
    assert "尚未准备完整" in text


def test_failed_refresh_does_not_hide_other_packages_or_restore_waived_wait(tmp_path):
    data = state(tmp_path)
    data["model_refresh"]["task_ids"] = ["failed-refresh"]
    data["tasks"].append(
        {"task_id": "failed-refresh", "status": "failed", "stage": "relax_and_feature"}
    )
    data["tasks"][0]["stage"] = "dft_single_point"
    folder = tmp_path / "archives/01-upload/current"
    folder.mkdir(parents=True)
    for name in ("inputs.zip", "submit_all.sh", "dispatch_tasks.py"):
        (folder / name).write_text("synthetic")
    data["slurm_batches"][0]["handoff"] = {
        "folder": str(folder),
        "upload_archive": str(folder / "inputs.zip"),
        "return_directory": str(tmp_path / "return-results"),
    }
    with response_detail("brief"):
        assert "提交路径：" in "\n".join(waiting_file_lines(data))
        data["tasks"][0]["recovery_wait_waived"] = True
        assert waiting_file_lines(data) == []

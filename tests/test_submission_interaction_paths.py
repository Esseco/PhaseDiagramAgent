import json
from pathlib import Path

from phase_agent.runtime.concise_dialogue import approval_reply
from phase_agent.runtime.handoff_presentation import handoff_lines, planned_handoff_lines
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.tools.remote.submission_layout import write_submission_index


def test_handoff_reply_names_submission_script_command_and_return_paths():
    handoff = {
        "folder": r"E:\project\submissions\epoch1\transfers\DFTSP_Batch01_input",
        "submit_script": r"E:\project\submissions\epoch1\transfers\DFTSP_Batch01_input\submit_all.sh",
        "command": "sbatch submit_all.sh",
        "return_directory": r"E:\project\submissions\epoch1\transfers\DFTSP_Batch01_output",
        "layout_version": 4,
    }
    text = "\n".join(handoff_lines([handoff]))
    assert "提交路径：" in text and handoff["folder"] in text
    assert "提交脚本：" in text and handoff["submit_script"] in text
    assert "sbatch submit_all.sh" in text
    assert "回收路径：" in text and handoff["return_directory"] in text


def test_handoff_command_uses_the_registered_script_name():
    handoff = {
        "folder": r"E:\project\submissions\epoch1\transfers\RelaxBatch_input",
        "submit_script": r"E:\project\submissions\epoch1\transfers\RelaxBatch_input\GPU.sh",
        "return_directory": r"E:\project\submissions\epoch1\transfers\RelaxBatch_output",
    }
    text = "\n".join(handoff_lines([handoff]))
    assert "提交命令：`sbatch GPU.sh`" in text
    assert "sbatch submit_all.sh" not in text


def test_approval_preview_shows_paths_for_relax_mc_dft_and_finetuning(tmp_path):
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 2}},
        "relax_mc_batches": {},
    }
    state_path = tmp_path / "workflow_state" / "state.json"
    root = tmp_path / "submissions" / "epoch1"
    cases = [
        (
            {
                "tool": "generate_branches",
                "parameters": {"module_execution_scope": {"submit_authorized": False}},
            },
            root / "delivery/relax_mc" / "RelaxMC_Batch01_input",
            root / "delivery/relax_mc" / "RelaxMC_Batch01_output",
        ),
        (
            {"tool": "allocate_mc_bohb", "parameters": {}},
                root / "delivery/transfers" / "<MC批次>_input",
                root / "delivery/transfers" / "<MC批次>_output",
        ),
        (
            {
                "tool": "prepare_local_batch_files",
                "parameters": {
                    "mode": "relax_mc_inputs",
                    "relax_mc_preview": {"model_version": "m1"},
                },
            },
            root / "delivery/relax_mc" / "RelaxMC_Batch01_input",
            root / "delivery/relax_mc" / "RelaxMC_Batch01_output",
        ),
        (
            {
                "tool": "select_dft_candidates",
                "parameters": {"dft_input_preview": {"task_count": 2, "model_version": "m1"}},
            },
                root / "delivery/transfers" / "<DFT批次>_input",
                root / "delivery/transfers" / "<DFT批次>_output",
        ),
        (
            {
                "tool": "update_mlip",
                "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
            },
            root / "delivery/transfers" / "<微调批次>_input",
            root / "delivery/transfers" / "<微调批次>_output",
        ),
    ]
    for action, upload, returned in cases:
        text = "\n".join(planned_handoff_lines({"raw_action": action}, state, state_path))
        assert str(upload) in text
        assert str(returned) in text
        assert "sbatch submit_all.sh" in text
        assert "不提交超算" in text


def test_approval_reply_includes_relax_mc_path_preview(tmp_path):
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "relax_mc_batches": {},
    }
    state_path = tmp_path / "workflow_state" / "state.json"
    proposal = {
        "expected_purpose": "为已选分支准备 Relax+MC 输入",
        "raw_action": {
            "tool": "prepare_local_batch_files",
            "parameters": {
                "mode": "relax_mc_inputs",
                "relax_mc_preview": {
                    "model_version": "m1",
                    "workload": {
                        "relax_structure_tasks": 1,
                        "mc_task_upper": [1],
                        "gpu_jobs_upper_by_wave": [1],
                        "max_concurrent_jobs": 1,
                    },
                    "diagnosis": {"warnings": []},
                },
            },
        },
    }
    text = approval_reply(proposal, state, state_path=state_path)
    assert str(tmp_path / "submissions/epoch0/delivery/relax_mc/RelaxMC_Batch01_input") in text
    assert str(tmp_path / "submissions/epoch0/delivery/relax_mc/RelaxMC_Batch01_output") in text
    assert "提交脚本" in text and "sbatch submit_all.sh" in text
    assert "先补齐 branch 生成概况和范围核对，再审批" in text
    detailed = format_workflow_reply(
        {"status": "awaiting_approval", "agent_proposal": proposal, "state": state},
        state_path,
        verbose=True,
    )
    assert "先补齐 branch 生成概况和范围核对，再审批" in detailed
    assert "预计回收路径" in detailed
    assert "回复“同意”执行" not in detailed


def test_approval_preview_uses_configured_submission_root(tmp_path):
    action = {"tool": "update_mlip", "parameters": {"action": "RETRAIN_MLIP"}}
    root = tmp_path / "custom-submissions"
    text = "\n".join(
        planned_handoff_lines(
            {"raw_action": action},
            {"active_model_version": "m1"},
            tmp_path / "workflow_state/state.json",
            config={"upload_batches_directory": str(root)},
        )
    )
    assert str(root / "epoch0/delivery/transfers") in text
    assert "具体批次文件夹在输入准备完成后登记" in text


def test_submission_index_is_machine_readable_and_records_exact_handoff_paths(tmp_path):
    root = tmp_path / "project" / "submissions"
    operation = root / "epoch0" / "G0001_DFT-0001_DFT-single-point"
    batch = operation / "inputs" / "batch-001"
    handoff_folder = root / "epoch0" / "delivery/transfers" / "DFTSP_Batch01_input"
    return_folder = root / "epoch0" / "delivery/transfers" / "DFTSP_Batch01_output"
    for directory in (batch, operation / "results", handoff_folder, return_folder):
        directory.mkdir(parents=True, exist_ok=True)
    script = handoff_folder / "submit_all.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    handoff = {
        "folder": str(handoff_folder),
        "upload_archive": str(handoff_folder / "inputs.zip"),
        "submit_script": str(script),
        "command": "sbatch submit_all.sh",
        "return_directory": str(return_folder),
        "job_count": 1,
        "layout_version": 4,
    }
    state = {
        "active_model_version": "m1",
        "upload_layout": {"model_rounds": {"m1": 1}},
        "tasks": [{"task_id": "t1", "status": "completed"}],
        "slurm_batches": [
            {
                "batch_id": "b1",
                "task_ids": ["t1"],
                "stage": "dft_single_point",
                "status": "prepared",
                "upload_directory": str(batch),
                "script_path": str(batch / "GPU.sh"),
                "results_directory": str(operation / "results"),
                "handoff": handoff,
            }
        ],
        "remote_finetune_jobs": {},
        "relax_mc_batches": {},
    }
    write_submission_index(root, state)
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    row = index["handoffs"][0]
    assert row["submission_path"]["path"] == str(handoff_folder)
    assert row["submit_script"]["path"] == str(script)
    assert row["submit_command"] == "sbatch submit_all.sh"
    assert row["return_path"]["path"] == str(return_folder)
    assert row["return_path"]["relative_path"] == "epoch0/delivery/transfers/DFTSP_Batch01_output"
    assert row["submit_script"]["exists"] is True
    assert row["status"] == ["prepared"]
    assert row["task_status_counts"] == {"completed": 1}
    epoch_readme = (root / "epoch0/README.md").read_text(encoding="utf-8")
    assert "`../submission_index.json`" in epoch_readme
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "submission_index.json" in readme and "epochN/delivery/transfers" in readme


def test_empty_submission_index_is_valid_for_unstarted_project(tmp_path):
    root = tmp_path / "new-project" / "submissions"
    root.mkdir(parents=True)
    write_submission_index(root, {})
    index = json.loads((root / "submission_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert index["handoffs"] == [] and index["operations"] == []


def test_relax_mc_preparation_publishes_exact_paired_paths(tmp_path):
    from tests.test_relax_mc_batch import package

    _, _, config, _, _, result = package(tmp_path, count=2)
    index_path = Path(config["upload_batches_directory"]) / "submission_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    row = index["relax_mc_batches"][0]
    assert row["submission_path"]["path"] == result["upload_folder"]
    assert row["submit_script"]["path"] == result["script_path"]
    assert row["return_path"]["path"] == str(Path(result["expected_results_archive"]).parent)
    assert result["submitted"] is False

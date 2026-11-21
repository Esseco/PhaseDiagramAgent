import json
from pathlib import Path

from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner


def test_mc_threshold_uses_packaged_worker(tmp_path):
    structure = tmp_path / "structure.vasp"
    structure.write_text("synthetic", encoding="utf8")
    task = {
        "task_id": "MC-1",
        "task_key": "mc:1",
        "branch_id": "B-1",
        "stage": "deep_search",
        "status": "pending",
        "model_version": "m1",
        "parameters": {},
    }
    state = {
        "confirmed_config_version": "v1",
        "tasks": [task],
        "pending_tasks": [task.copy()],
        "budget_reservations": {"mc:1": {"status": "reserved", "reserved_cost": 1}},
    }

    def prepare(row):
        return {
            **row,
            "worker_job": {
                "structure_path": str(structure),
                "parameters": {"feature_difference_threshold": 3.0},
            },
        }

    runner = ManualUploadBatchRunner(
        tmp_path / "upload",
        worker_command=["python3", "--executor", "module:execute"],
        stage_batch_sizes={"deep_search": 20},
        task_preparer=prepare,
    )
    result = runner.prepare(state)
    directory = Path(result["batch"]["upload_directory"])
    entry = json.loads((directory / "manifest.json").read_text(encoding="utf8"))[0]
    task_path = directory / entry["input_path"]
    saved = json.loads(task_path.read_text(encoding="utf8"))
    assert saved["worker_job"]["parameters"]["feature_difference_threshold"] == 3.0
    assert "_handoff_refresh_executor:execute_mlip_task" in (
        directory / "run_mlip_batch.py"
    ).read_text(encoding="utf8")
    assert (task_path.parent / "_handoff_refresh_worker.py").is_file()

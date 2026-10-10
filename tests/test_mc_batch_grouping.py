from pathlib import Path

from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner


def _tasks(count):
    return [{"task_id": f"MC-{index:03d}", "task_key": f"mc:{index}",
             "branch_id": f"B-{index:03d}", "stage": "deep_search",
             "status": "pending", "model_version": "mace-mh-1",
             "parameters": {"seed": index, "max_mc_steps": 30 if index % 2 else 100}}
            for index in range(count)]


def _state(tasks):
    return {"confirmed_config_version": "v1", "tasks": tasks,
            "pending_tasks": list(tasks), "budget_reservations": {
                row["task_key"]: {"status": "reserved", "reserved_cost": 1.0}
                for row in tasks}}


def test_119_branches_make_12_manual_gpu_jobs_with_distinct_task_results(tmp_path):
    tasks = _tasks(119)
    runner = ManualUploadBatchRunner(tmp_path / "upload",
        worker_command=["python3", "--executor", "module:execute"],
        stage_batch_sizes={"deep_search": 20}, task_preparer=lambda task: task)
    assert runner.batch_sizes["deep_search"] == 10  # Old confirmed configs are capped.
    state = _state(tasks)
    sizes = []
    while True:
        result = runner.prepare(state)
        state = result["state"]
        if result["status"] == "no_tasks":
            break
        batch = result["batch"]
        sizes.append(len(batch["task_ids"]))
        assert (Path(batch["upload_directory"]) / "GPU.sh").is_file()
        assert len(batch["task_directories"]) == len(batch["task_ids"])
    assert sizes == [10] * 11 + [9]
    assert len({row["result_path"] for row in state["tasks"]}) == 119
    assert all(row.get("slurm_batch_id") for row in state["tasks"])


def test_mc_parameters_do_not_split_jobs_but_model_and_resources_do(tmp_path):
    tasks = _tasks(4)
    tasks[2]["model_version"] = "another-model"
    tasks[3]["resource_profile"] = "large"
    state = _state(tasks)
    runner = SlurmBatchRunner(tmp_path / "jobs", worker_command=["worker"],
                              stage_batch_sizes={"deep_search": 20})
    assert runner.stage_batch_sizes["deep_search"] == 10
    selected = runner._select_tasks(state, tasks)
    assert [row["task_id"] for row in selected] == ["MC-000", "MC-001"]

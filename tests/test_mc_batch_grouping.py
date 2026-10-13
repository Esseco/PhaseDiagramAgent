from pathlib import Path

from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner


def _tasks(count):
    return [
        {
            "task_id": f"MC-{index:03d}",
            "task_key": f"mc:{index}",
            "branch_id": f"B-{index:03d}",
            "stage": "deep_search",
            "status": "pending",
            "model_version": "mace-mh-1",
            "parameters": {"seed": index, "max_mc_steps": 30 if index % 2 else 100},
        }
        for index in range(count)
    ]


def _state(tasks):
    return {
        "confirmed_config_version": "v1",
        "tasks": tasks,
        "pending_tasks": list(tasks),
        "budget_reservations": {
            row["task_key"]: {"status": "reserved", "reserved_cost": 1.0} for row in tasks
        },
    }


def test_119_branches_make_6_manual_gpu_jobs_with_distinct_task_results(tmp_path):
    tasks = _tasks(119)
    runner = ManualUploadBatchRunner(
        tmp_path / "upload",
        worker_command=["python3", "--executor", "module:execute"],
        stage_batch_sizes={"deep_search": 20},
        task_preparer=lambda task: task,
    )
    assert runner.batch_sizes["deep_search"] == 20
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
    assert sizes == [20] * 5 + [19]
    assert len({row["result_path"] for row in state["tasks"]}) == 119
    assert all(row.get("slurm_batch_id") for row in state["tasks"])


def test_mc_parameters_do_not_split_jobs_but_model_and_resources_do(tmp_path):
    tasks = _tasks(4)
    tasks[2]["model_version"] = "another-model"
    tasks[3]["resource_profile"] = "large"
    state = _state(tasks)
    runner = SlurmBatchRunner(
        tmp_path / "jobs", worker_command=["worker"], stage_batch_sizes={"deep_search": 20}
    )
    assert runner.stage_batch_sizes["deep_search"] == 20
    selected = runner._select_tasks(state, tasks)
    assert [row["task_id"] for row in selected] == ["MC-000", "MC-001"]


def test_configured_mc_capacity_has_no_hidden_ten_task_cap(tmp_path):
    from phase_agent.tools.mc_batch_policy import mc_batch_limit

    assert mc_batch_limit(20) == 20
    assert mc_batch_limit(40) == 40
    runner = SlurmBatchRunner(
        tmp_path, worker_command=["worker"], stage_batch_sizes={"deep_search": 20}
    )
    script = runner._script("batch", tmp_path / "manifest.json", 20, "deep_search")
    assert "#SBATCH --array" not in script
    assert "seq 0 19" in script
    assert "--array-index" in script


def test_relax_groups_use_one_allocation_for_100_tasks(tmp_path):
    runner = ManualUploadBatchRunner(
        tmp_path,
        worker_command=["python3", "--executor", "module:execute"],
        stage_batch_sizes={"relax_and_feature": 100},
        task_preparer=lambda task: task,
    )
    tasks = _tasks(205)
    for task in tasks:
        task["stage"] = "relax_and_feature"
        task["parameters"] = {"fmax": 0.05}
    state = _state(tasks)
    sizes = []
    while True:
        result = runner.prepare(state)
        state = result["state"]
        if result["status"] == "no_tasks":
            break
        sizes.append(len(result["batch"]["task_ids"]))
    assert sizes == [100, 100, 5]


def test_remote_mlip_jobs_are_serial_gpu_groups(tmp_path):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner

    runner = RemoteBatchRunner(tmp_path, worker_command=["worker"])
    for stage, count in (("relax_and_feature", 100), ("deep_search", 20)):
        script = runner._script("batch", count, stage)
        assert "#SBATCH --array" not in script
        assert f"seq 0 {count - 1}" in script

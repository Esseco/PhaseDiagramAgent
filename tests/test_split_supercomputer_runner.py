import json

from phase_agent.tools.slurm.slurm_batch_runner import SlurmBatchRunner
from phase_agent.tools.step_runner.advise_next_actions import advise_next_actions
from phase_agent.tools.step_runner.confirm_action_plan import confirm_action_plan
from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.tools.step_runner.prepare_confirmed_plan import prepare_confirmed_plan
from phase_agent.tools.step_runner.recover_results import recover_results
from phase_agent.tools.step_runner.scheduler_adapter import MockSchedulerAdapter
from phase_agent.tools.step_runner.submit_prepared_jobs import submit_prepared_jobs


def _task(index, stage, model="m1"):
    return {
        "task_id": f"T{index:04d}", "task_key": f"K{index:04d}",
        "object_id": f"S{index:04d}", "structure_id": f"S{index:04d}",
        "stage": stage, "status": "pending", "model_version": model,
        "parameters": {"fixed": True},
    }


def _state(tasks):
    return {
        "confirmed_config_version": "config-1", "tasks": tasks,
        "pending_tasks": list(tasks),
        "budget_reservations": {
            row["task_key"]: {"status": "reserved", "reserved_cost": 1.0,
                              "stage": row["stage"]} for row in tasks
        },
        "reserved_relative_cost": float(len(tasks)),
    }


def test_stage_batch_sizes_and_compatibility(tmp_path):
    tasks = (
        [_task(i, "relax_and_feature") for i in range(1, 206)]
        + [_task(250, "relax_and_feature", model="m2")]
        + [_task(i, "deep_search") for i in range(300, 341)]
        + [_task(i, "dft_single_point") for i in range(500, 503)]
    )
    state = _state(tasks)
    runner = SlurmBatchRunner(
        tmp_path / "jobs", worker_command=["worker"],
        stage_batch_sizes={"relax_and_feature": 100, "deep_search": 20,
                           "dft_single_point": 1},
        dispatcher=lambda task: {"status": "pending", "outputs": {"generator": "atomate"}},
        stage_profiles={"dft": {"stages": ["dft_single_point"], "vasp_command": "vasp_std"}},
    )
    sizes = []
    while True:
        result = runner.prepare(state); state = result["state"]
        if result["status"] == "no_tasks":
            break
        sizes.append(len(result["batch"]["task_ids"]))
    assert sizes == [100, 100, 5, 1, 10, 10, 10, 10, 1, 1, 1, 1]


def test_plan_prepare_submit_are_idempotent(tmp_path):
    summary_path = tmp_path / "summary.json"
    write_json(summary_path, {"summary_id": "summary-1", "config_version": "config-1"})
    calls = []
    def client(payload):
        calls.append(payload)
        return {"actions": [{"tool": "run_calculation_stage"}], "reason": "test"}
    first = advise_next_actions(summary_path, tmp_path / "plans", agent_client=client,
                                config_version="config-1", node_role="login")
    second = advise_next_actions(summary_path, tmp_path / "plans", agent_client=client,
                                 config_version="config-1", node_role="login")
    assert first["plan"]["plan_id"] == second["plan"]["plan_id"]
    assert len(calls) == 1
    confirm_action_plan(first["plan_path"], user_confirmed=True, node_role="login")
    state_path = tmp_path / "state.json"; write_json(state_path, _state([_task(1, "deep_search")]))
    runner = SlurmBatchRunner(tmp_path / "jobs", worker_command=["worker"],
                              stage_batch_sizes={"deep_search": 20})
    prepared = prepare_confirmed_plan(state_path, first["plan_path"], batch_runner=runner,
                                      expected_config_version="config-1", node_role="compute")
    repeated = prepare_confirmed_plan(state_path, first["plan_path"], batch_runner=runner,
                                      expected_config_version="config-1", node_role="compute")
    assert len(prepared["batches"]) == 1
    assert repeated["status"] == "already_prepared_or_no_tasks"
    scheduler = MockSchedulerAdapter()
    submitted = submit_prepared_jobs(state_path, scheduler=scheduler, node_role="login")
    repeated_submit = submit_prepared_jobs(state_path, scheduler=scheduler, node_role="login")
    assert len(submitted["submitted"]) == 1
    assert repeated_submit["status"] == "nothing_to_submit"
    assert len(scheduler.submissions) == 1


def test_partial_and_duplicate_recovery_settle_once(tmp_path):
    state_path = tmp_path / "state.json"; summary_path = tmp_path / "summary.json"
    runner = SlurmBatchRunner(tmp_path / "jobs", worker_command=["worker"],
                              stage_batch_sizes={"deep_search": 20})
    state = runner.prepare(_state([_task(1, "deep_search"), _task(2, "deep_search")]))["state"]
    write_json(state_path, state)
    task = state["tasks"][0]
    write_json(task["result_path"], {"task_id": task["task_id"], "task_key": task["task_key"],
                                     "status": "completed", "actual_cost": 0.7})
    write_json((tmp_path / "jobs/slurm-000001/00000-T0001/task.finished.json"),
               {"status": "completed"})
    first = recover_results(state_path, summary_path, result_collector=runner,
                            config_version="config-1", node_role="compute")
    second = recover_results(state_path, summary_path, result_collector=runner,
                             config_version="config-1", node_role="compute")
    assert first["results_found"] == 1
    assert first["state"]["budget_usage"]["total_relative_cost"] == 0.7
    assert second["results_found"] == 0
    assert second["state"]["budget_usage"]["total_relative_cost"] == 0.7

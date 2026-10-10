from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner


def test_409_structures_produce_five_python_scripts(tmp_path):
    tasks = [{"task_id": f"S{i}", "task_key": f"relax:{i}", "stage": "relax_and_feature",
              "branch_id": f"B{i // 3}", "status": "pending", "model_version": "mh1"}
             for i in range(409)]
    state = {"tasks": tasks, "budget_reservations": {
        row["task_key"]: {"status": "reserved"} for row in tasks}}
    runner = ManualUploadBatchRunner(tmp_path, worker_command=["python3", "--executor", "module:execute"],
                                    task_preparer=lambda row: row)
    counts = []
    for _ in range(5):
        result = runner.prepare(state)
        assert result["status"] == "prepared"
        counts.append(len(result["batch"]["task_ids"]))
        state = result["state"]
    assert counts == [100, 100, 100, 100, 9]
    assert len(list(tmp_path.rglob("*.py"))) == 5
    assert len(list(tmp_path.rglob("GPU.sh"))) == 5
    assert runner.prepare(state)["status"] == "no_tasks"

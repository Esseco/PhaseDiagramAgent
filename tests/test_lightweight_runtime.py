import json
from execution_layer.remote.run_single_task import run_single_task
from execution_layer.remote.integrity import file_checksum
from analysis_layer.cost.estimate_task_cost import estimate_task_cost


def test_execution_adds_timing_before_checksum(tmp_path):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"task_id": "t", "task_key": "k", "model_version": "mace"}))
    result = run_single_task(task, executor=lambda t: {"status": "completed"})
    assert result["runtime_observation"]["elapsed_seconds"] >= 0
    marker = json.loads((tmp_path / "task.finished.json").read_text())
    assert marker["result_checksum"] == file_checksum(tmp_path / "result.json")


def test_history_dominates_time_and_uses_same_patience():
    history = [{"stage": "deep_search", "status": "completed", "atom_count": 40,
        "runtime_observation": {"elapsed_seconds": 100, "backend": "mace", "hardware": "v100",
        "gpu_count": 1, "cpu_count": 2, "patience": 20, "max_mc_steps": 100, "actual_mc_steps": 50}}]
    report = estimate_task_cost("deep_search", atom_count=40, backend="mace", hardware="v100",
                               patience=20, max_mc_steps=100, state={"cost_history": history})
    assert report["quality"] == "historical_runtime_estimate"
    assert report["scenarios"][0]["scenario"] == "historical_typical_steps"
    assert report["scenarios"][0]["runtime_estimate"]["elapsed_seconds"] == 100
    assert report["scenarios"][-1]["runtime_estimate"]["elapsed_seconds"] == 200


def test_unknown_hardware_does_not_invent_time():
    report = estimate_task_cost("relax_and_feature", atom_count=40, state={"cost_history": []})
    assert report["scenarios"][0]["runtime_estimate"]["status"] == "insufficient_samples"

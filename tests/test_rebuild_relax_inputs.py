import json
import pytest
from phase_agent.tools.local.rebuild_relax_inputs import (
    is_relax_rebuild_request, plan_relax_rebuild, rebuild_relax_inputs)
from phase_agent.tools.policy.execution_policy import apply_execution_policy


def fixture_state(tmp_path):
    batch = tmp_path / "remote-000001"
    directory = batch / "00000-R1"
    directory.mkdir(parents=True)
    (directory / "task.json").write_text("{}")
    (directory / "initial.vasp").write_text("generated copy")
    (batch / "manifest.json").write_text(json.dumps([{"task_id": "R1"}]))
    return {"tasks": [{"stage": "relax_and_feature", "task_id": "R1", "task_key": "k1",
                       "status": "pending", "batch_id": batch.name, "input_path": str(directory / "task.json")}],
            "budget_reservations": {"k1": {"status": "reserved", "reserved_cost": 4}},
            "slurm_batches": [{"batch_id": batch.name, "task_ids": ["R1"]}]}, batch


def test_plan_does_not_delete_then_approved_cleanup_preserves_ids_budget(tmp_path):
    state, batch = fixture_state(tmp_path)
    plan = plan_relax_rebuild(state, tmp_path)
    assert batch.exists()
    restored = rebuild_relax_inputs(state, tmp_path, plan)
    assert not batch.exists()
    assert restored["tasks"][0]["task_id"] == "R1"
    assert restored["budget_reservations"] == state["budget_reservations"]


def test_results_and_unknown_files_protect_entire_batch(tmp_path):
    state, batch = fixture_state(tmp_path)
    (batch / "result.json").write_text("{}")
    with pytest.raises(ValueError):
        plan_relax_rebuild(state, tmp_path)
    assert batch.exists()


def test_changed_plan_rejected(tmp_path):
    state, batch = fixture_state(tmp_path)
    with pytest.raises(ValueError):
        rebuild_relax_inputs(state, tmp_path, {"directories": [], "task_ids": []})
    assert batch.exists()


def test_running_tasks_not_deleted(tmp_path):
    state, batch = fixture_state(tmp_path)
    state["tasks"][0]["status"] = "running"
    with pytest.raises(ValueError):
        plan_relax_rebuild(state, tmp_path)
    assert batch.exists()


def test_explicit_request_and_autonomous_approval_guard():
    assert is_relax_rebuild_request("重新生成mlip输入")
    assert not is_relax_rebuild_request("继续")
    result = apply_execution_policy({"raw_action": {"tool": "prepare_local_batch_files",
        "parameters": {"rebuild_inputs": True}}}, execution_mode="autonomous")
    assert result["status"] == "awaiting_approval" and not result["execute"]

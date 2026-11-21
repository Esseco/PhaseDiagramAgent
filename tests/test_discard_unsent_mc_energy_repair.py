from pathlib import Path
import pytest
from phase_agent.tools.state.discard_unsent_mc_energy_repair import discard_unsent_mc_energy_repair


def fixture(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    preview = {"checksum": "r", "frames": [{"source_task_id": "source"}]}
    identity = {"invocation_id": "d"}
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_mc_energy_repair",
        "target_ids": ["source"],
        "parameters": {"mode": "mc_energy_repair_inputs", "energy_repair_preview": preview},
    }
    task = {
        "task_id": "t",
        "task_key": "k",
        "stage": "relax_and_feature",
        "status": "pending",
        "config_version": "c",
        "model_version": "m",
        "parent_decision_id": "d",
        "energy_basis_repair_id": "r",
        "slurm_batch_id": "b",
    }
    state = {
        "tasks": [task],
        "pending_tasks": [task.copy()],
        "budget_reservations": {"k": {"status": "reserved", "reserved_cost": 1.5}},
        "mc_energy_repairs": {
            "r": {
                "preview": preview,
                "status": "prepared",
                "task_ids": ["t"],
                "approval_record_id": "d",
            }
        },
        "slurm_batches": [
            {
                "batch_id": "b",
                "status": "prepared",
                "task_ids": ["t"],
                "handoff": {"folder": str(folder)},
            }
        ],
    }
    refs = [
        {
            k: task[k]
            for k in (
                "task_id",
                "task_key",
                "stage",
                "status",
                "config_version",
                "model_version",
                "parent_decision_id",
            )
        }
    ]
    events = [
        {"event": "claimed", "payload": {"identity": identity, "action": action}},
        {
            "event": "tool_returned",
            "payload": {
                "identity": identity,
                "outcome": "completed",
                "evidence": {"result_status": "prepared", "returned_task_refs": refs},
            },
        },
    ]
    return state, events, folder


def test_discards_only_receipt_bound_local_tasks(tmp_path):
    state, events, folder = fixture(tmp_path)
    result, count = discard_unsent_mc_energy_repair(
        state,
        events,
        decision_id="d",
        repair_id="r",
        user_message="discard",
        workspace_root=tmp_path,
    )
    assert count == 1 and result["tasks"][0]["status"] == "cancelled"
    assert result["budget_reservations"]["k"]["status"] == "released"
    assert result["slurm_batches"][0]["status"] == "cancelled"
    assert result["mc_energy_repairs"]["r"]["status"] == "cancelled"
    assert not result["pending_tasks"] and folder.exists()
    assert state["tasks"][0]["status"] == "pending"


@pytest.mark.parametrize(
    "case", ["submitted", "result", "receipt", "target", "missing_event", "budget", "other_batch"]
)
def test_discard_fails_closed_without_mutation(tmp_path, case):
    state, events, folder = fixture(tmp_path)
    if case == "submitted":
        state["slurm_batches"][0]["job_id"] = "123"
    if case == "result":
        result = folder / "result.json"
        result.write_text("{}")
        state["tasks"][0]["result_path"] = str(result)
    if case == "receipt":
        (folder / ".submitted_jobs.json").write_text("{}")
    if case == "target":
        events[0]["payload"]["action"]["target_ids"] = ["other"]
    if case == "missing_event":
        events.pop()
    if case == "budget":
        state["budget_reservations"]["k"]["status"] = "settled"
    if case == "other_batch":
        state["slurm_batches"][0]["task_ids"].append("other")
    with pytest.raises(ValueError):
        discard_unsent_mc_energy_repair(
            state,
            events,
            decision_id="d",
            repair_id="r",
            user_message="discard",
            workspace_root=tmp_path,
        )
    assert state["tasks"][0]["status"] == "pending"

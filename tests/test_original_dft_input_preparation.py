"""Reconstruct approved Co/Cr selections privately; never submit or run DFT."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import zipfile

import pytest

from tests.test_legacy_delivery_recovery import setup_original
from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches
from phase_agent.graphs.approval_graph import durable_approval


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("p")  # Keep nested portable paths below Windows MAX_PATH.


@pytest.mark.parametrize("project,count", [("O3-1ele-NaCoO2", 35), ("O3-1ele-NaCrO2", 15)])
@pytest.mark.real_project
def test_original_selected_tasks_approval_to_current_input_without_new_decision(
    tmp_path, monkeypatch, project, count
):
    saved, handoff = setup_original(tmp_path, project)
    source_tasks = saved["tasks"]
    keys = (
        "task_id",
        "task_key",
        "stage",
        "status",
        "model_version",
        "config_version",
        "structure_id",
        "object_id",
        "branch_id",
        "parameters",
        "planned_relative_cost",
    )
    tasks = [{k: deepcopy(t[k]) for k in keys if k in t} for t in source_tasks]
    state = {
        "tasks": tasks,
        "confirmed_config_version": source_tasks[0]["config_version"],
        "budget_reservations": {
            t["task_key"]: {
                "status": "reserved",
                "stage": t["stage"],
                "reserved_cost": t.get("planned_relative_cost", 0),
            }
            for t in tasks
        },
    }
    before = deepcopy(state)
    geometries = {}
    with zipfile.ZipFile(handoff["upload_archive"]) as archive:
        for task in source_tasks:
            relative = Path(task["input_path"]).relative_to(handoff["import_directory"])
            geometries[task["task_id"]] = archive.read(
                relative.with_name("initial_structure.vasp").as_posix()
            )
    calls = []

    def generate(task, **kwargs):
        folder = Path(task["work_directory"])
        for name in ("POSCAR", "initial_structure.vasp"):
            (folder / name).write_bytes(geometries[task["task_id"]])
        for name in ("INCAR", "KPOINTS", "POTCAR", "workflow.py"):
            (folder / name).write_text("fake scientific input; never execute")
        calls.append(task["task_id"])
        return {"outputs": {"generator": "atomate"}}  # Simulated external adapter response.

    monkeypatch.setattr(
        "phase_agent.tools.local.prepare_dft_upload_batches.prepare_pycode_relax", generate
    )
    monkeypatch.setattr("phase_agent.persistence.shared_dft.reusable_results", lambda *a, **k: [])
    envelope = {
        "invocation_id": "original-selection",
        "proposal_hash": "frozen",
        "config_version": state["confirmed_config_version"],
        "revision": 0,
    }
    approval_path = tmp_path / "approval-copy/state.json"
    approval_path.parent.mkdir()
    assert durable_approval(approval_path, envelope)["status"] == "awaiting_approval"
    assert calls == []
    assert durable_approval(approval_path, envelope, "approve")["status"] == "decision_delivered"
    config = {
        "upload_batches_directory": str(tmp_path / "s"),
        "mlip": {"version": source_tasks[0]["model_version"], "model_path": "/HPC/original.model"},
        "python_environments": {"remote_mlip": "current"},
    }
    context = {
        "event_state": state,
        "effective_config": config,
        "manager": SimpleNamespace(data={}),
    }
    result = prepare_dft_upload_batches(
        action={"target_ids": [t["task_id"] for t in tasks]}, context=context
    )
    assert result["status"] == "prepared", result
    assert set(calls) == {t["task_id"] for t in source_tasks} and len(calls) == count
    assert len(result["state"]["tasks"]) == count and state == before
    deliveries = result["handoffs"]
    assert len(deliveries) == 1
    folder = Path(deliveries[0]["folder"])
    assert folder.name.endswith("_input")
    assert {p.name for p in folder.iterdir()} == {
        "inputs.zip",
        "submit_all.sh",
        "dispatch_tasks.py",
    }
    with zipfile.ZipFile(folder / "inputs.zip") as archive:
        payloads = [
            json.loads(archive.read(n)) for n in archive.namelist() if n.endswith("/task.json")
        ]
    assert {t["task_id"] for t in payloads} == set(calls)
    context["event_state"] = result["state"]
    again = prepare_dft_upload_batches(action={"target_ids": calls}, context=context)
    assert again["status"] == "already_prepared", again
    assert len(calls) == count and len(again["state"]["tasks"]) == count

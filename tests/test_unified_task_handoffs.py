from pathlib import Path
from types import SimpleNamespace

from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs
from phase_agent.tools.local.prepare_dft_upload_batches import prepare_dft_upload_batches


def test_packaging_failure_keeps_tasks_and_reports_blocker(monkeypatch):
    state = {
        "tasks": [{"task_id": "MC-1", "status": "pending"}],
        "budget_reservations": {"k": {"status": "reserved"}},
    }

    def fail(_state):
        raise OSError("script missing")

    monkeypatch.setattr("phase_agent.tools.remote.submission_handoff.ensure_waiting_handoffs", fail)
    result = finalize_prepared_handoffs(state)
    assert result["status"] == "not_configured"
    assert result["state"] is state
    assert "script missing" in result["reason"]
    assert state["tasks"][0]["task_id"] == "MC-1"
    assert state["budget_reservations"]["k"]["status"] == "reserved"


def test_dft_preparation_finishes_with_one_unified_package(tmp_path, monkeypatch):
    monkeypatch.setattr("phase_agent.persistence.shared_dft.reusable_results", lambda *a, **k: [])

    def inputs(task, **kwargs):
        folder = Path(task["work_directory"])
        for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "workflow.py"):
            text = (
                "Na\n1\n1 0 0\n0 1 0\n0 0 1\nNa\n1\nDirect\n0 0 0\n"
                if name == "POSCAR" else "print('fake workflow')\n" if name == "workflow.py"
                else "fake input"
            )
            (folder / name).write_text(text)
        return {"outputs": {"generator": "atomate"}}

    monkeypatch.setattr(
        "phase_agent.tools.local.prepare_dft_upload_batches.prepare_pycode_relax", inputs
    )
    state = {
        "confirmed_config_version": "v1",
        "tasks": [
            {
                "task_id": "D1",
                "task_key": "dft:1",
                "stage": "dft_single_point",
                "status": "pending",
                "model_version": "m1",
            }
        ],
        "budget_reservations": {"dft:1": {"status": "reserved"}},
    }
    config = {
        "upload_batches_directory": str(tmp_path / "submissions"),
        "mlip": {"version": "m1", "model_path": "/HPC/model.model"},
        "python_environments": {"remote_mlip": "current"},
    }
    result = prepare_dft_upload_batches(
        action={"target_ids": ["D1"]},
        context={
            "event_state": state,
            "effective_config": config,
            "manager": SimpleNamespace(data={}),
        },
    )
    assert result["status"] == "prepared"
    assert len(result["handoffs"]) == 1
    assert result["handoffs"][0]["job_count"] == 1
    folder = Path(result["handoffs"][0]["folder"])
    assert folder.parent == tmp_path / "submissions/epoch0/transfers"
    assert all((folder / n).is_file() for n in ("inputs.zip", "submit_all.sh", "dispatch_tasks.py"))
    assert result["state"]["tasks"][0]["task_id"] == "D1"
    assert state["tasks"][0].get("slurm_batch_id") is None

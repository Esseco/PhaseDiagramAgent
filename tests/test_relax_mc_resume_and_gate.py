from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import zipfile

import pytest

from phase_agent.tools.relax_mc_batch.files import write, read, rebase
from phase_agent.tools.relax_mc_batch.resume import resume_batch
from phase_agent.tools.relax_mc_batch.completion_gate import (
    require_complete_archive,
    require_complete_imports,
)
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.transfer_bundle import create_bundle
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


class Scheduler:
    def __init__(self):
        self.calls = []
        self.active_ids = set()
        self.failed_state = "FAILED"
        self.uncertain = False

    def active(self, ids):
        return set(ids) & self.active_ids

    def evidence(self, ids):
        return {
            i: {
                "state": "COMPLETED" if i == "10" else self.failed_state,
                "exit_code": "0:53",
                "reason": "None",
                "elapsed_seconds": 0,
            }
            for i in ids
        }

    def submit(self, path, *, dependency=None, exclude_nodes=()):
        self.calls.append((str(path), dependency, exclude_nodes))
        if self.uncertain:
            raise RuntimeError("lost sbatch receipt")
        return str(100 + len(self.calls))


def fixture(root):
    root.mkdir(exist_ok=True)
    tasks, batches = [], []
    for index, count in enumerate([1, 2]):
        batch = "b" + str(index)
        worker = root / "jobs" / batch
        worker.mkdir(parents=True)
        script = worker / "GPU.sh"
        script.write_text("#!/bin/bash\ntrue\n")
        ids = []
        for n in range(count):
            tid = batch + "-" + str(n)
            d = worker / tid
            d.mkdir()
            task = {
                "task_id": tid,
                "task_key": "key-" + tid,
                "batch_id": batch,
                "task_checksum": "sum-" + tid,
                "model_version": "m1",
                "config_version": "c1",
                "relax_mc_bundle_id": "bundle",
                "stage": "deep_search",
                "status": "completed" if index == 0 else "pending",
                "input_path": str(d / "task.json"),
                "result_path": str(d / "result.json"),
            }
            write(d / "task.json", task)
            if index == 0:
                write(d / "result.json", {**task, "outputs": {"energy": -1}})
                write(
                    d / "task.finished.json",
                    {**task, "result_checksum": file_checksum(d / "result.json")},
                )
            tasks.append(task)
            ids.append(tid)
        batches.append(
            {
                "batch_id": batch,
                "job_id": str(10 + index),
                "script_path": str(script),
                "task_ids": ids,
                "submission_state": "submitted",
            }
        )
    state = {
        "tasks": tasks,
        "slurm_batches": batches,
        "relax_mc_remote": {
            "phase": 3,
            "status": "paused",
            "completed_mc_waves": [0, 1],
            "phase_controllers": {
                "3": {"batch_ids": ["b0", "b1"], "job_id": "12", "submission_state": "submitted"}
            },
        },
    }
    manifest = {
        "bundle_id": "bundle",
        "stage_profiles": {
            "relax_mc_cpu": {"slurm_options": {"partition": "p1", "ntasks": 1, "mem": "8G"}}
        },
    }
    write(root / "batch.json", manifest)
    write(root / "workflow_state.json", rebase(state, root, portable=True))
    manager = PhaseDataManager({"P": ["O3"], "H": {"max_det": 2}, "TM_ratio": {"Co": 1}})
    write(root / "manager_checkpoint.json", manager.data)
    write(
        root / "transfer_manifest.json",
        {"files": [{"path": "batch.json", "sha256": file_checksum(root / "batch.json")}]},
    )
    return state


def test_preview_submit_preserves_completed_and_uses_original_dependent_controller(tmp_path):
    before = fixture(tmp_path)
    io = Scheduler()
    signed = Path(before["tasks"][0]["result_path"]).read_bytes()
    preview = resume_batch(tmp_path, io=io, exclude_nodes=["bad-node"])
    assert preview["retry_task_count"] == 2 and not io.calls
    with pytest.raises(ValueError, match="plan_changed"):
        resume_batch(
            tmp_path, io=io, submit=True, expected_plan="stale", exclude_nodes=["bad-node"]
        )
    result = resume_batch(
        tmp_path, io=io, submit=True, expected_plan=preview["plan_id"], exclude_nodes=["bad-node"]
    )
    assert len(io.calls) == 2
    assert io.calls[0][0] == before["slurm_batches"][1]["script_path"]
    assert io.calls[0][2] == ("bad-node",)
    assert io.calls[1][1] == "afterany:10:101"
    assert io.calls[1][0].endswith("analyze_phase_3.sh")
    assert Path(before["tasks"][0]["result_path"]).read_bytes() == signed
    state = rebase(read(tmp_path / "workflow_state.json"), tmp_path)
    assert state["tasks"] == before["tasks"]
    assert state["slurm_batches"][0]["job_id"] == "10"
    assert state["slurm_batches"][1]["attempt_history"][0]["job_id"] == "11"
    assert state["relax_mc_remote"]["status"] == "running"
    assert result["analysis_job_id"] == "102"
    resume_batch(
        tmp_path, io=io, submit=True, expected_plan=preview["plan_id"], exclude_nodes=["bad-node"]
    )
    assert len(io.calls) == 2


@pytest.mark.parametrize("state", ["UNKNOWN", "COMPLETED"])
def test_unconfirmed_failure_never_submits(tmp_path, state):
    fixture(tmp_path)
    io = Scheduler()
    io.failed_state = state
    with pytest.raises(ValueError, match="startup_failure_not_confirmed"):
        resume_batch(tmp_path, io=io)
    assert not io.calls


def test_active_or_existing_result_blocks_retry(tmp_path):
    state = fixture(tmp_path)
    io = Scheduler()
    io.active_ids.add("11")
    with pytest.raises(ValueError, match="still_active"):
        resume_batch(tmp_path, io=io)
    io.active_ids.clear()
    Path(state["tasks"][1]["result_path"]).write_text("partial")
    with pytest.raises(ValueError, match="existing_worker_result"):
        resume_batch(tmp_path, io=io)
    assert not io.calls


def test_uncertain_receipt_is_durable_and_never_resubmitted(tmp_path):
    fixture(tmp_path)
    io = Scheduler()
    preview = resume_batch(tmp_path, io=io)
    io.uncertain = True
    with pytest.raises(RuntimeError):
        resume_batch(tmp_path, io=io, submit=True, expected_plan=preview["plan_id"])
    with pytest.raises(ValueError, match="submission_uncertain"):
        resume_batch(tmp_path, io=io, submit=True, expected_plan=preview["plan_id"])
    assert len(io.calls) == 1


def archive(root, status="paused", pending=True):
    root.mkdir(exist_ok=True)
    tasks = [
        {"task_id": "a", "status": "completed"},
        {"task_id": "b", "status": "pending" if pending else "completed"},
    ]
    summary = {
        "status": status,
        "task_count": 2,
        "bundle_id": "b",
        "plan_checksum": "p",
        "model_version": "m1",
        "config_version": "c1",
    }
    write(root / "workflow_state.json", {"tasks": tasks})
    write(root / "result_summary.json", summary)
    write(root / "candidate_pool.json", {})
    return Path(
        create_bundle(root, list(root.glob("*.json")), root / "results.zip", kind="results")
    )


@pytest.mark.parametrize(
    "status,pending", [("paused", True), ("partial", False), ("failed", True), ("completed", True)]
)
@pytest.mark.parametrize("previous_checksum", ["same", None, "older-checksum"])
def test_formal_gate_blocks_incomplete_before_any_extraction(
    tmp_path, status, pending, previous_checksum
):
    from phase_agent.tools.relax_mc_batch.results import import_results
    from phase_agent.tools.remote.transfer_bundle import checksum

    path = archive(tmp_path, status, pending)
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "config_version": "c1",
                "expected_results_archive": str(path),
                "import_checksum": checksum(path)
                if previous_checksum == "same"
                else previous_checksum,
            }
        },
    }
    before = deepcopy(state)
    with pytest.raises(ValueError, match="formal_recovery_blocked"):
        import_results(
            action={"parameters": {"bundle_id": "b"}},
            context={
                "event_state": state,
                "effective_config": {"mlip": {}},
                "config_version": "c1",
            },
        )
    assert state == before
    assert not (tmp_path / "verified_results").exists()


def test_completed_gate_and_publication_gate(tmp_path):
    require_complete_archive(archive(tmp_path, "completed", False))
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "status": "paused",
                "import_checksum": "old",
                "result_summary": {"status": "paused"},
            }
        },
    }
    with pytest.raises(ValueError, match="publication_blocked"):
        require_complete_imports(state)
    from phase_agent.runtime.phase_diagram_chat import _current_snapshot

    with pytest.raises(ValueError, match="publication_blocked"):
        _current_snapshot(state, "mlip")


def test_public_review_quarantines_without_changing_tasks_or_raw_results(tmp_path):
    from phase_agent.runtime.relax_mc_return_review import review_return
    from phase_agent.tools.remote.transfer_bundle import checksum

    path = archive(tmp_path)
    state_path = tmp_path / "workflow_state/state.json"
    state_path.parent.mkdir()
    state = {
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
        "tasks": [{"task_id": "a", "status": "completed"}],
        "phase_records": [{"energy": -1}],
        "phase_diagrams": {"mlip": {"entries": [1], "model_version": "m1"}},
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "config_version": "c1",
                "plan_checksum": "p",
                "status": "paused",
                "import_checksum": checksum(path),
                "expected_results_archive": str(path),
            }
        },
    }
    write(state_path, state)
    control = SimpleNamespace(
        state_path=state_path, chat_handler=SimpleNamespace(lock=threading.RLock())
    )
    raw = path.read_bytes()
    outcome = review_return(control, "b", quarantine=True)
    after = read(state_path)
    assert outcome["status"] == "quarantined_incomplete"
    assert after["tasks"] == state["tasks"] and after["phase_records"] == state["phase_records"]
    assert after["phase_diagrams"]["mlip"]["validity"] == "quarantined_incomplete_batch"
    assert path.read_bytes() == raw


def test_resume_rejoins_original_final_analysis_and_automatic_packaging(
    tmp_path_factory, monkeypatch
):
    from tests.test_relax_mc_phase_dependencies import setup, run_batch
    from phase_agent.tools.relax_mc_batch.results import inspect_archive

    root, io, calls = setup(tmp_path_factory.mktemp("s"), monkeypatch)
    run_batch(root, submit=True, io=io)
    for phase in (0, 1):
        assert run_batch(root, submit=True, io=io, phase=phase)["status"] == "queued"
    io.missing = True
    assert run_batch(root, submit=True, io=io, phase=2)["status"] == "queued"
    assert run_batch(root, submit=True, io=io, phase=3)["status"] == "paused"
    prior_calls = len(calls)
    prior = read(root / "workflow_state.json")
    completed = {
        t["task_id"]: t["outputs"]["energy"] for t in prior["tasks"] if t["status"] == "completed"
    }
    io.evidence = lambda ids: {
        job: {"state": "FAILED", "exit_code": "0:53", "elapsed_seconds": 0} for job in ids
    }
    plan = resume_batch(root, io=io)
    assert plan["retry_task_count"] > 0
    io.missing = False
    assert (
        resume_batch(root, io=io, submit=True, expected_plan=plan["plan_id"])["status"] == "queued"
    )
    outcome = run_batch(root, submit=True, io=io, phase=3)
    assert outcome["status"] == "completed"
    assert len(calls) - prior_calls == plan["retry_task_count"]
    after = read(root / "workflow_state.json")
    assert {
        t["task_id"]: t["outputs"]["energy"] for t in after["tasks"] if t["task_id"] in completed
    } == completed
    assert after["relax_mc_remote"]["completed_mc_waves"] == [0, 1, 2]
    assert inspect_archive(outcome["archive"])["summary"]["status"] == "completed"
    require_complete_archive(outcome["archive"])


@pytest.mark.parametrize(
    "row,eligible",
    [
        ({"state": "FAILED", "exit_code": "0:53", "elapsed_seconds": 0}, True),
        ({"state": "FAILED", "exit_code": "0:53", "elapsed_seconds": 30}, False),
        ({"state": "FAILED", "exit_code": "0:1", "elapsed_seconds": 0}, False),
        ({"state": "UNKNOWN", "exit_code": "0:53", "elapsed_seconds": 0}, False),
        ({"state": "NODE_FAIL", "elapsed_seconds": 0}, True),
        ({"state": "FAILED", "exit_code": "0:53"}, False),
    ],
)
def test_scheduler_classifier_never_infers_noncompute_from_failure_alone(row, eligible):
    from phase_agent.tools.relax_mc_batch.hpc_failure import classify_scheduler_failure

    assert classify_scheduler_failure(row).startswith("hpc_") is eligible


def test_agent_and_mcp_recovery_scope_keeps_completed_and_requires_remote_preflight():
    from phase_agent.tools.relax_mc_batch.hpc_failure import (
        startup_recovery_options,
        prepare_resume_handoff,
    )
    from phase_agent.tools.dispatch.scientific_operations import bind_operation

    state = {
        "active_model_version": "m1",
        "tasks": [
            {
                "task_id": "done",
                "status": "completed",
                "relax_mc_bundle_id": "b",
                "calculation_result_path": "/remote/work/jobs/r/result.json",
            },
            {"task_id": "missing", "status": "pending", "relax_mc_bundle_id": "b"},
        ],
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "result_summary": {
                    "status": "paused",
                    "diagnostics": {
                        "jobs": [{"job_id": "1", "state": "FAILED", "exit_code": "0:53"}],
                        "issue_groups": [{"job_id": "1", "task_ids": ["done", "missing"]}],
                    },
                },
            }
        },
    }
    facts = startup_recovery_options(state)[0]
    assert facts["candidate_task_ids"] == ["missing"]
    assert facts["requires_remote_preflight"] is True
    assert facts["remote_preview_command"] == "python3 run_batch.py resume --root /remote/work"
    action = bind_operation(
        {
            "operation": "prepare_relax_mc_resume",
            "tool": "prepare_local_batch_files",
            "parameters": {"mode": "relax_mc_resume", "bundle_id": "b"},
        }
    )
    result = prepare_resume_handoff(action=action, context={"event_state": state})
    assert result["status"] == "prepared" and result["submitted"] is False
    assert result["recovery_plan"]["candidate_task_count"] == 1


def test_quarantined_partial_plot_is_not_reported_as_last_complete(tmp_path):
    from phase_agent.analysis.feedback.versioned_products import publication_status

    folder = tmp_path / "latest/phase_diagram"
    folder.mkdir(parents=True)
    write(
        folder / "manifest.json",
        {"version": "partial-version", "identity": {"model_version": "m1", "epoch": "epoch0"}},
    )
    result = publication_status(
        tmp_path,
        "phase_diagram",
        "m1",
        "epoch0",
        status="not_ready",
        reason="relax_mc_formal_recovery_blocked",
    )
    assert "last_complete" not in result
    assert result["quarantined_latest"]["version"] == "partial-version"


def test_bounded_input_validation_checks_every_file_and_rejects_escape(tmp_path):
    from phase_agent.tools.relax_mc_batch.run import validate_frozen_inputs

    a = tmp_path / "inputs/a.txt"
    a.parent.mkdir()
    a.write_text("required")
    write(
        tmp_path / "transfer_manifest.json",
        {"files": [{"path": "inputs/a.txt", "sha256": file_checksum(a)}]},
    )
    validate_frozen_inputs(tmp_path)
    a.write_text("changed")
    with pytest.raises(ValueError, match="integrity_failed"):
        validate_frozen_inputs(tmp_path)
    write(
        tmp_path / "transfer_manifest.json", {"files": [{"path": "../outside.txt", "sha256": "x"}]}
    )
    with pytest.raises(ValueError, match="integrity_failed"):
        validate_frozen_inputs(tmp_path)

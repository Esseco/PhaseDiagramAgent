from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.tools.relax_mc_batch.recovery import build_recovery_return
from phase_agent.tools.remote.transfer_bundle import create_bundle, validate_bundle


def fixture(tmp_path):
    base = tmp_path / "original"
    base.mkdir()
    task = {
        "task_id": "t1",
        "task_key": "k1",
        "structure_id": "s1",
        "branch_id": "b1",
        "stage": "deep_search",
        "status": "pending",
        "task_checksum": "frozen-checksum",
        "config_version": "c",
        "model_version": "m",
        "batch_id": "gpu1",
        "result_path": "jobs/results/00000-t1/result.json",
        "input_path": "jobs/inputs/gpu1/00000-t1/task.json",
    }
    state = {"tasks": [task], "active_model_version": "m", "confirmed_config_version": "c"}
    summary = {
        "bundle_id": "b",
        "plan_checksum": "approved",
        "model_version": "m",
        "config_version": "c",
        "status": "failed",
        "reason": "old squeue error",
        "task_count": 1,
    }
    for name, data in (
        ("workflow_state.json", state),
        ("result_summary.json", summary),
        ("candidate_pool.json", {"branch_candidates": [], "mc_pool": []}),
    ):
        (base / name).write_text(json.dumps(data))
    archive = create_bundle(base, list(base.glob("*.json")), tmp_path / "old.zip", kind="results")
    evidence = tmp_path / "evidence"
    directory = evidence / "jobs/results/00000-t1"
    directory.mkdir(parents=True)
    structure = directory / "final.vasp"
    structure.write_text("original structure bytes")
    result = {
        **task,
        "status": "completed",
        "outputs": {
            "structure_path": "/cluster/work/inputs/final.vasp",
            "structure_checksum": hashlib.sha256(structure.read_bytes()).hexdigest(),
            "energy": -1.0,
            "composition": {"Na": 1},
        },
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result))
    marker = {
        k: result[k]
        for k in (
            "task_id",
            "task_key",
            "batch_id",
            "task_checksum",
            "config_version",
            "model_version",
            "status",
        )
    }
    marker["result_checksum"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (directory / "task.finished.json").write_text(json.dumps(marker))
    (evidence / "workflow_state.json").write_text(
        json.dumps(
            {
                "relax_mc_remote": {"phase": 3},
                "slurm_batches": [{"job_id": "123", "task_ids": ["t1"]}],
            }
        )
    )
    job = {"job_id": "123", "task_ids": ["t1"], "state": "COMPLETED", "exit_code": "0:0"}
    return archive, evidence, job


def build(tmp_path, archive, evidence, job):
    return build_recovery_return(
        archive, evidence, tmp_path / "built", job_evidence=job, remote_root="/cluster/work"
    )


def test_completed_gpu_with_failed_controller_is_collect_only_and_idempotent(tmp_path):
    archive, evidence, job = fixture(tmp_path)
    original = Path(archive).read_bytes()
    result = build(tmp_path, archive, evidence, job)
    assert result["status"] == "completed" and result["submitted"] is False
    assert result["recovery"]["recovered_task_ids"] == ["t1"]
    validate_bundle(result["archive"])
    again = build(tmp_path, archive, evidence, job)
    assert again["idempotent_replay"] and again["archive"] == result["archive"]
    assert Path(archive).read_bytes() == original


@pytest.mark.parametrize(
    "fault", ["wrong_job", "wrong_scope", "unknown_scheduler", "bad_checksum", "missing_marker"]
)
def test_recovery_never_invents_success_or_submits_on_bad_evidence(tmp_path, fault):
    archive, evidence, job = fixture(tmp_path)
    if fault == "wrong_job":
        job["job_id"] = "999"
    elif fault == "wrong_scope":
        job["task_ids"] = ["other"]
    elif fault == "unknown_scheduler":
        job["state"] = "UNKNOWN"
    elif fault == "bad_checksum":
        (evidence / "jobs/results/00000-t1/result.json").write_text("{}")
    else:
        (evidence / "jobs/results/00000-t1/task.finished.json").unlink()
    if fault == "missing_marker":
        result = build(tmp_path, archive, evidence, job)
        assert result["status"] == "partial" and result["recovery"]["unverified_task_ids"] == ["t1"]
        assert result["submitted"] is False
    else:
        with pytest.raises(ValueError):
            build(tmp_path, archive, evidence, job)


def test_collecting_early_phase_does_not_invent_later_mc_completion(tmp_path):
    archive, evidence, job = fixture(tmp_path)
    path = evidence / "workflow_state.json"
    snapshot = json.loads(path.read_text())
    snapshot["relax_mc_remote"]["phase"] = 1
    path.write_text(json.dumps(snapshot))
    result = build(tmp_path, archive, evidence, job)
    assert result["status"] == "partial"
    assert (
        result["recovery"]["next_phase_not_submitted"] and not result["recovery"]["batch_complete"]
    )
    assert result["submitted"] is False


def test_interrupted_package_build_preserves_evidence_and_can_restart(tmp_path, monkeypatch):
    from phase_agent.tools.relax_mc_batch import recovery

    archive, evidence, job = fixture(tmp_path)
    write = recovery.write
    fail = [True]

    def interrupted(path, value, **kwargs):
        if Path(path).name == "result_summary.json" and fail[0]:
            fail[0] = False
            raise OSError("interrupted packaging")
        return write(path, value, **kwargs)

    monkeypatch.setattr(recovery, "write", interrupted)
    with pytest.raises(OSError):
        build(tmp_path, archive, evidence, job)
    result = build(tmp_path, archive, evidence, job)
    assert result["status"] == "completed" and result["submitted"] is False
    assert list((tmp_path / "built").glob("interrupted-*"))


def test_verified_task_failure_is_not_reported_as_successful_batch(tmp_path):
    archive, evidence, job = fixture(tmp_path)
    path = evidence / "jobs/results/00000-t1/result.json"
    marker = path.with_name("task.finished.json")
    data = json.loads(path.read_text())
    data["status"] = "failed"
    data["error"] = "worker failure"
    path.write_text(json.dumps(data))
    m = json.loads(marker.read_text())
    m.update(status="failed", result_checksum=hashlib.sha256(path.read_bytes()).hexdigest())
    marker.write_text(json.dumps(m))
    result = build(tmp_path, archive, evidence, job)
    assert result["status"] == "partial" and result["recovery"]["verified_failed_task_ids"] == [
        "t1"
    ]
    assert not result["recovery"]["batch_complete"] and result["submitted"] is False


def test_marker_status_disagreement_is_rejected(tmp_path):
    archive, evidence, job = fixture(tmp_path)
    path = evidence / "jobs/results/00000-t1/result.json"
    marker = path.with_name("task.finished.json")
    data = json.loads(path.read_text())
    data["status"] = "failed"
    path.write_text(json.dumps(data))
    m = json.loads(marker.read_text())
    m["result_checksum"] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="marker_result_status_mismatch"):
        build(tmp_path, archive, evidence, job)

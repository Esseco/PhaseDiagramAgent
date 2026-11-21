import hashlib, json
from pathlib import Path
import pytest
from phase_agent.tools.remote.transfer_bundle import create_bundle, checksum
from phase_agent.tools.relax_mc_batch.transition_archive import (
    verify_transition,
    register_transition,
)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True), encoding="utf8")


def fixture(tmp_path):
    frozen = tmp_path / "frozen"
    raw = tmp_path / "raw"
    frozen.mkdir()
    raw.mkdir()
    task = {
        "task_id": "t",
        "task_key": "k",
        "stage": "relax_and_feature",
        "structure_id": "s",
        "branch_id": "b",
        "model_version": "m",
        "config_version": "c",
        "relax_mc_bundle_id": "batch",
        "structure_path": "inputs/a.vasp",
    }
    batch = {
        "bundle_id": "batch",
        "plan_checksum": "p",
        "model_version": "m",
        "config_version": "c",
        "initial_state": {"tasks": [task]},
    }
    write(frozen / "batch.json", batch)
    (frozen / "inputs").mkdir()
    from ase import Atoms
    from ase.io import write as write_atoms

    initial = Atoms("Na", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    write_atoms(frozen / "inputs/a.vasp", initial, format="vasp")
    from pymatgen.io.ase import AseAtomsAdaptor
    from Process_AL_MC.MC_sample import _relaxation_input_identity

    identity = _relaxation_input_identity(AseAtomsAdaptor.get_structure(initial))
    frozenzip = Path(
        create_bundle(
            frozen,
            [frozen / "batch.json", frozen / "inputs/a.vasp"],
            tmp_path / "inputs.zip",
            kind="inputs",
        )
    )
    worker = {
        **task,
        "task_checksum": "sum",
        "batch_id": "jobbatch",
        "protocol_version": 1,
        "input_file_version": "task-json-v1",
        "input_path": "jobs/input/task.json",
        "result_path": "jobs/result/result.json",
    }
    write(raw / "batch.json", batch)
    write(raw / "jobs/input/task.json", worker)
    review_obs = {
        "identity_scheme": "periodic_input_v1",
        "structure_id": identity,
        "monitor_complete": True,
        "extrapolation_failure": True,
    }
    out = {
        "optimizer_converged": True,
        "structure_checksum": hashlib.sha256(b"final").hexdigest(),
        "structure_path": "/remote/initial_relaxed.vasp",
        "extrapolation_monitor": {
            "extrapolation_failure": True,
            "final_feature_difference": 2.5,
            "failures": [{"trigger": "final", "score": 2.5}],
        },
        "relaxation_observation": review_obs,
    }
    result = {
        **worker,
        "status": "completed",
        "converged": True,
        "outputs": out,
        "checks_passed": False,
        "result_disposition": "excluded_extrapolation_failure",
        "relaxation_quality": {"status": "rejected", "reason": "extrapolation_failure"},
    }
    write(raw / "jobs/result/result.json", result)
    marker = {
        key: worker[key]
        for key in (
            "task_id",
            "task_key",
            "batch_id",
            "config_version",
            "model_version",
            "protocol_version",
            "input_file_version",
            "task_checksum",
        )
    }
    marker.update(status="completed", result_checksum=checksum(raw / "jobs/result/result.json"))
    write(raw / "jobs/result/task.finished.json", marker)
    (raw / "jobs/result/initial_relaxed.vasp").write_bytes(b"final")
    write(
        raw / "workflow_state.json",
        {"tasks": [worker], "slurm_batches": [{"job_id": "123", "task_ids": ["t"]}]},
    )
    paths = [p for p in raw.rglob("*") if p.is_file()]
    rawzip = Path(create_bundle(raw, paths, tmp_path / "raw.zip", kind="results"))
    receipt = tmp_path / "receipts.json"
    write(
        receipt,
        {
            "source_sha256": checksum(rawzip),
            "jobs": [{"job_evidence": {"job_id": "123", "state": "COMPLETED", "task_ids": ["t"]}}],
        },
    )
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "tasks": [task],
        "relax_mc_batches": {
            "batch": {
                "bundle_id": "batch",
                "status": "prepared",
                "model_version": "m",
                "config_version": "c",
                "plan_checksum": "p",
                "upload_archive": str(frozenzip),
            }
        },
    }
    return state, rawzip, receipt


def run(state, raw, receipt):
    return verify_transition(state, "batch", raw, checksum(raw), receipt, checksum(receipt))


def test_verify_and_register_only_review(tmp_path):
    state, raw, receipt = fixture(tmp_path)
    original = json.dumps(state, sort_keys=True)
    review = run(state, raw, receipt)
    assert (
        review["task_count"] == 1
        and review["original_failure_count"] == 1
        and review["effective_failure_count"] == 0
    )
    updated = register_transition(state, review)
    assert json.dumps(state, sort_keys=True) == original and updated["tasks"] == state["tasks"]
    assert updated["relax_mc_batches"]["batch"]["status"] == "prepared"
    assert (
        updated["relax_mc_batches"]["batch"]["threshold_transition"]["status"]
        == "verified_awaiting_continuation"
    )
    assert register_transition(updated, review) == updated


def test_reject_wrong_version_and_scheduler_evidence(tmp_path):
    state, raw, receipt = fixture(tmp_path)
    with pytest.raises(ValueError, match="current_version"):
        run({**state, "active_model_version": "other"}, raw, receipt)
    data = json.loads(receipt.read_text())
    data["jobs"][0]["job_evidence"]["state"] = "RUNNING"
    write(receipt, data)
    with pytest.raises(ValueError, match="scheduler_evidence"):
        run(state, raw, receipt)


def test_reject_unknown_or_changed_input(tmp_path):
    state, raw, receipt = fixture(tmp_path)
    with pytest.raises(ValueError, match="checksum_mismatch"):
        verify_transition(state, "batch", raw, "0" * 64, receipt, checksum(receipt))
    state["tasks"][0]["branch_id"] = "other"
    with pytest.raises(ValueError, match="task_binding_mismatch"):
        run(state, raw, receipt)


def test_reference_paths_are_portable_and_rebound(tmp_path):
    from phase_agent.tools.relax_mc_batch.files import rebase

    path = tmp_path / "references" / "phase-0.vasp"
    source = {"phase_references": {"P3": str(path)}, "other": "unchanged"}
    portable = rebase(source, tmp_path, portable=True)
    assert portable["phase_references"]["P3"] == "references/phase-0.vasp"
    restored = rebase(portable, tmp_path)
    assert restored["phase_references"]["P3"] == str(path.resolve())
    assert restored["other"] == "unchanged"


def test_remote_input_attestation_requires_exact_byte_scope(tmp_path):
    from phase_agent.tools.relax_mc_batch.transition_archive import validate_input_attestation

    review = {
        "frozen_archive_sha256": "a" * 64,
        "review_digest": "d",
        "identity_check": {
            "mismatches": [
                {"task_id": "t", "input_member": "inputs/00007.vasp", "initial_sha256": "b" * 64}
            ]
        },
    }
    evidence = {
        "original_archive_sha256": "a" * 64,
        "remote_input_fingerprint": "a" * 64,
        "method": "sha256sum",
        "host": "hpc.example",
        "files": [{"remote_path": "/work/inputs/00007.vasp", "sha256": "b" * 64}],
    }
    path = tmp_path / "attestation.json"
    write(path, evidence)
    assert validate_input_attestation(review, path, checksum(path))["task_ids"] == ["t"]
    evidence["files"][0]["sha256"] = "c" * 64
    write(path, evidence)
    with pytest.raises(ValueError, match="remote_input_hash_mismatch"):
        validate_input_attestation(review, path, checksum(path))


def test_materialize_preserves_signed_bytes_and_is_idempotent(tmp_path):
    from phase_agent.tools.relax_mc_batch.transition_materialize import materialize_verified_relax

    state, raw, receipt = fixture(tmp_path)
    review = run(state, raw, receipt)
    directory = tmp_path / "verified"
    first = materialize_verified_relax(raw, checksum(raw), directory, review)
    second = materialize_verified_relax(raw, checksum(raw), directory, review)
    assert len(first["results"]) == 1 and first == second
    result = first["results"][0]
    assert result["outputs"]["extrapolation_monitor"]["extrapolation_failure"] is False
    original = json.loads((directory / "jobs/result/result.json").read_text(encoding="utf8"))
    assert original["outputs"]["extrapolation_monitor"]["extrapolation_failure"] is True
    assert result["outputs"]["structure_path"] == str(
        (directory / "jobs/result/initial_relaxed.vasp").resolve()
    )
    with pytest.raises(ValueError, match="archive_changed"):
        materialize_verified_relax(raw, "0" * 64, directory, review)

import copy
import hashlib
import json
from pathlib import Path
import zipfile

from phase_agent.tools.relax_mc_batch.return_payload import result_files


def test_pending_worker_evidence_survives_without_marking_completed(tmp_path):
    out = tmp_path / "jobs/results/t"
    out.mkdir(parents=True)
    worker = tmp_path / "jobs/inputs/b/t"
    (worker / "pool").mkdir(parents=True)
    (worker / "pool/rank_0.vasp").write_text("candidate")
    (worker / "pool/unreferenced.vasp").write_text("not referenced")
    (out / "mc_occupations.bin").write_bytes(b"occupation")
    (out / "result.json").write_text(
        json.dumps({"outputs": {"pool_summary": [{"structure_file": "rank_0.vasp"}]}})
    )
    (out / "task.finished.json").write_text("{}")
    task = {
        "task_id": "t",
        "status": "pending",
        "stage": "deep_search",
        "result_path": str(out / "result.json"),
        "input_path": str(worker / "task.json"),
    }
    state = {"tasks": [task]}
    before = copy.deepcopy(state)
    selected = result_files(tmp_path, state, {})
    assert out / "result.json" in selected
    assert out / "task.finished.json" in selected
    assert out / "mc_occupations.bin" in selected
    assert worker / "pool/rank_0.vasp" in selected
    assert worker / "pool/unreferenced.vasp" not in selected
    assert state == before


def test_controller_exception_packages_uncollected_worker_results(tmp_path, monkeypatch):
    from tests.test_relax_mc_phase_dependencies import setup, run_batch

    root, io, _ = setup(tmp_path, monkeypatch)
    run_batch(root, submit=True, io=io)
    submitted = len(io.rows)

    def unavailable(ids):
        raise OSError("unexpected scheduler failure")

    io.active = unavailable
    result = run_batch(root, submit=True, io=io, phase=0)
    assert result["status"] == "failed"
    assert len(io.rows) == submitted
    state = json.loads((root / "workflow_state.json").read_text())
    assert all(t["status"] == "pending" for t in state["tasks"])
    with zipfile.ZipFile(root / "results.zip") as z:
        names = z.namelist()
        assert sum(n.endswith("/result.json") for n in names) == len(state["tasks"])
        assert sum(n.endswith("/task.finished.json") for n in names) == len(state["tasks"])


def test_new_package_records_actual_controller_sources_and_rejects_old_runtime(tmp_path):
    from tests.test_relax_mc_batch import package
    from phase_agent.tools.relax_mc_batch import prepare

    *_, result = package(tmp_path)
    root = Path(result["directory"])
    manifest = json.loads((root / "batch.json").read_text())
    with zipfile.ZipFile(root / "inputs.zip") as z:
        for name, digest in manifest["runtime_source_checksums"].items():
            assert (
                hashlib.sha256(
                    z.read("runtime/phase_agent/tools/relax_mc_batch/" + name)
                ).hexdigest()
                == digest
            )
    # Reuse validation runs before any manager/config or preparation access.
    manifest.pop("runtime_source_checksums")
    (root / "batch.json").write_text(json.dumps(manifest))
    bundle = manifest["bundle_id"]
    record = {
        "directory": str(root),
        "plan_checksum": manifest["plan_checksum"],
        "status": "prepared",
    }
    context = {
        "effective_config": {},
        "manager": None,
        "event_state": {"relax_mc_batches": {bundle: record}},
    }
    action = {
        "parameters": {
            "relax_mc_preview": {"bundle_id": bundle, "checksum": manifest["plan_checksum"]}
        }
    }
    outcome = prepare.prepare_batch(action=action, context=context)
    assert outcome["status"] == "rejected"
    assert outcome["reason"] == "relax_mc_runtime_changed_reprepare_required"
    assert outcome["submitted"] is False

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phase_agent.tools.relax_mc_batch.analysis import candidate_pool
from phase_agent.tools.relax_mc_batch.return_payload import result_files
from phase_agent.tools.remote.integrity import file_checksum


def pool_fixture(root, count=20):
    out = root / "jobs/results/t"
    worker = root / "jobs/inputs/b/t"
    out.mkdir(parents=True)
    (worker / "pool").mkdir(parents=True)
    (out / "final.vasp").write_text("final")
    (out / "mc_occupations.bin").write_bytes(b"occupations")
    (worker / "status.json.gz").write_bytes(b"checkpoint")
    summaries = []
    for n in range(count):
        name = f"unused_{n}.vasp"
        (worker / "pool" / name).write_text(f"unused geometry {n}")
        summaries.append({"structure_file": name, "energy_mean_per_atom": -1.0})
    outputs = {
        "structure_path": str(out / "final.vasp"),
        "checkpoint": str(worker / "status.json.gz"),
        "pool_summary": summaries,
    }
    raw = {"status": "completed", "outputs": outputs}
    (out / "result.json").write_text(json.dumps(raw))
    (out / "task.finished.json").write_text(
        json.dumps({"result_checksum": file_checksum(out / "result.json")})
    )
    task = {
        "task_id": "t",
        "structure_id": "s",
        "branch_id": "b",
        "model_version": "m1",
        "stage": "deep_search",
        "status": "completed",
        "input_path": str(worker / "task.json"),
        "result_path": str(out / "final.vasp"),
        "outputs": outputs,
    }
    return {"active_model_version": "m1", "tasks": [task]}, out, worker


def test_candidate_export_does_not_promote_all_historical_pool_rows(tmp_path):
    state, out, worker = pool_fixture(tmp_path)
    before = deepcopy(state)
    exported = candidate_pool(state)
    assert exported["mc_pool"] == []
    assert len(exported["branch_candidates"]) == 1
    assert exported["branch_candidates"][0]["structure_path"] == str(out / "final.vasp")
    assert state == before


@pytest.mark.parametrize("count", [1, 20])
def test_only_consumer_required_files_return_regardless_of_pool_size(tmp_path, count):
    state, out, worker = pool_fixture(tmp_path, count)
    raw = (out / "result.json").read_bytes()
    # Legacy mc_pool rows are historical metadata, not selected local candidates.
    candidates = {
        "branch_candidates": [{"structure_path": str(out / "final.vasp")}],
        "mc_pool": [
            {
                "source_task_id": "t",
                "structure_file": "unused_0.vasp",
                "structure_path": str(worker / "pool/unused_0.vasp"),
            }
        ],
    }
    selected = result_files(tmp_path, state, candidates)
    assert set(selected) == {
        out / "result.json",
        out / "task.finished.json",
        out / "final.vasp",
        out / "mc_occupations.bin",
        worker / "status.json.gz",
    }
    assert (out / "result.json").read_bytes() == raw
    assert len(json.loads(raw)["outputs"]["pool_summary"]) == count


def test_explicit_branch_candidate_geometry_is_required(tmp_path):
    state, out, worker = pool_fixture(tmp_path)
    chosen = worker / "pool/unused_0.vasp"
    candidates = {"branch_candidates": [{"structure_path": str(chosen)}]}
    assert chosen in result_files(tmp_path, state, candidates)
    chosen.unlink()
    with pytest.raises(FileNotFoundError, match="required_return_artifact"):
        result_files(tmp_path, state, candidates)


def test_uncollected_result_keeps_only_its_final_geometry(tmp_path):
    state, out, worker = pool_fixture(tmp_path)
    task = state["tasks"][0]
    task["status"] = "pending"
    task.pop("outputs")
    selected = result_files(tmp_path, state, {})
    assert out / "final.vasp" in selected
    assert not any(p.parent == worker / "pool" for p in selected)
    assert task["status"] == "pending"


def test_frozen_runtime_identity_covers_candidate_export_semantics():
    from phase_agent.tools.relax_mc_batch.prepare import _runtime_source_checksums

    assert "analysis.py" in _runtime_source_checksums()


def test_collect_only_maps_remote_checkpoint_without_rewriting_signed_json(tmp_path):
    state, out, worker = pool_fixture(tmp_path)
    raw = json.loads((out / "result.json").read_text())
    raw["checkpoint"] = "/cluster/work/jobs/inputs/b/t/status.json.gz"
    (out / "result.json").write_text(json.dumps(raw))
    original = (out / "result.json").read_bytes()
    files = result_files(tmp_path, state, {}, remote_root="/cluster/work")
    assert worker / "status.json.gz" in files
    assert (out / "result.json").read_bytes() == original

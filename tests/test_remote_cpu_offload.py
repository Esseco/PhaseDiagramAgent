"""Tests for offloading heavy CPU tasks (1/2/3) to HPC with strict CPU/GPU separation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.tools.relax_mc_batch.return_payload import result_files
from phase_agent.tools.remote.remote_branch_generation import (
    import_remote_branch_generation,
    prepare_remote_branch_generation_bundle,
    run_remote_branch_generation,
)
from phase_agent.tools.remote.remote_dft_materializer import (
    materialize_remote_dft_batch,
    prepare_remote_dft_manifest_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_remote_branch_generation_cpu_only_and_verified_import(tmp_path: Path):
    parent = tmp_path / "parent.vasp"
    parent.write_text("NaCoO2\n1.0\n", encoding="utf-8")

    prep = prepare_remote_branch_generation_bundle(
        {"P": ["P2"], "H": [1]},
        {"species": {"mobile": ["Na"]}},
        {"count": 2},
        [{"branch_id": "B0", "structure_path": str(parent)}],
        tmp_path / "prep",
    )
    assert prep["resource_type"] == "cpu"
    assert prep["requires_gpu"] is False
    slurm_text = Path(prep["script_path"]).read_text(encoding="utf-8")
    assert "--gres=gpu" not in slurm_text

    def fake_gen(manifest, struct_dir):
        out_vasp = struct_dir / "candidate.vasp"
        out_vasp.write_text("generated\n", encoding="utf-8")
        return {
            "branches": [{"branch_id": "B1", "P": "P2"}],
            "structures": [{"structure_id": "S1", "branch_id": "B1", "structure_path": str(out_vasp)}],
        }

    done = run_remote_branch_generation(
        prep["manifest_path"],
        tmp_path / "remote_out",
        generate_fn=fake_gen,
    )
    assert done["resource_type"] == "cpu"
    assert done["structure_count"] == 1

    imported = import_remote_branch_generation(done["archive"], tmp_path / "local_import")
    assert imported["status"] == "accepted"
    assert len(imported["structures"]) == 1
    assert Path(imported["structures"][0]["structure_path"]).is_file()


def test_remote_dft_materializer_cpu_prep_and_gpu_compute(tmp_path: Path):
    frame = tmp_path / "frame_0.vasp"
    frame.write_text("NaCoO2\n1.0\n", encoding="utf-8")
    sha = _sha256(frame)

    prep = prepare_remote_dft_manifest_bundle(
        [
            {
                "task_id": "DFT-1",
                "task_key": "k1",
                "stage": "dft_single_point",
                "structure_path": str(frame),
                "input_structure_sha256": sha,
            }
        ],
        tmp_path / "dft_prep",
        comparison_model={"version": "m1"},
    )
    assert prep["prep_resource_type"] == "cpu"
    assert prep["compute_resource_type"] == "gpu"
    assert "--gres=gpu" not in Path(prep["script_path"]).read_text(encoding="utf-8")

    seen = []
    mat = materialize_remote_dft_batch(
        prep["manifest_path"],
        tmp_path / "dft_jobs",
        preparer_fn=lambda task, comparison_model=None: seen.append((task["task_id"], comparison_model))
        or {"status": "prepared"},
    )
    assert mat["status"] == "materialized"
    assert mat["task_count"] == 1
    assert seen == [("DFT-1", {"version": "m1"})]

    # Tampered frame fails checksum check
    (tmp_path / "dft_prep" / "frames" / "DFT-1.vasp").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="remote_dft_materialize_checksum_mismatch"):
        materialize_remote_dft_batch(
            prep["manifest_path"], tmp_path / "dft_jobs_2", preparer_fn=lambda *a, **k: {}
        )


def test_return_payload_includes_remote_phase_quality_report(tmp_path: Path):
    root = tmp_path / "batch"
    root.mkdir()
    report_path = root / "remote_phase_quality_report.json"
    report_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    mat_path = root / "remote_dft_materialization.json"
    mat_path.write_text(json.dumps({"status": "materialized"}), encoding="utf-8")
    files = result_files(root, {"tasks": []}, {"branch_candidates": []})
    assert report_path.resolve() in files
    assert mat_path.resolve() in files

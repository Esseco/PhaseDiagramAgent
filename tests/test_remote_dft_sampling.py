"""Tests for remote trajectory retention, weighted kmeans++ DFT sampling, and local acceptance."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write
import numpy as np
import pytest
from pymatgen.io.ase import AseAtomsAdaptor

from phase_agent.agents.reliability.selection.recommend_post_mc_dft import recommend_post_mc_dft
from phase_agent.science.qbc.remote_dft_sampling import (
    EHULL_CUTOFF_EV_PER_ATOM,
    ENERGY_STD_CUTOFF_EV_PER_ATOM,
    FORCE_STD_CUTOFF_EV_PER_ANG,
    compute_sampling_weight,
    run_remote_dft_sampling,
    select_weighted_kmeans_pp,
    verify_remote_dft_sampling,
)


def _make_atoms(scale: float) -> Atoms:
    return Atoms(
        "NaCoO2",
        positions=[[0.0, 0.0, 0.0], [1.4 * scale, 1.4, 1.4], [0.7, 0.7, 2.1], [2.1, 2.1, 0.7]],
        cell=[3.0 * scale, 3.0, 4.0],
        pbc=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compute_sampling_weight_formula_and_cutoffs():
    res = compute_sampling_weight(0.010, 0.100, 0.050)
    expected_e = 1.0 - math.exp(-0.010 / ENERGY_STD_CUTOFF_EV_PER_ATOM)
    expected_f = 1.0 - math.exp(-0.100 / FORCE_STD_CUTOFF_EV_PER_ANG)
    expected_h = math.exp(-0.050 / EHULL_CUTOFF_EV_PER_ATOM)
    assert math.isclose(res["f_E"], expected_e, rel_tol=1e-12)
    assert math.isclose(res["f_F"], expected_f, rel_tol=1e-12)
    assert math.isclose(res["f_H"], expected_h, rel_tol=1e-12)
    assert math.isclose(res["weight"], expected_e * expected_f * expected_h, rel_tol=1e-12)

    with pytest.raises(ValueError):
        compute_sampling_weight(float("nan"), 0.1, 0.0)


def test_weighted_kmeans_pp_preserves_reference_dft_centers():
    refs = np.array([[0.0, 0.0]])
    cands = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    weights = np.array([1.0, 1.0, 1.0])
    chosen = select_weighted_kmeans_pp(cands, weights, reference_features=refs, n_select=2, seed=42)
    assert len(chosen) == 2
    assert 0 not in chosen
    assert set(chosen) == {1, 2}


def test_remote_sampling_gs_first_traj_kmeans_and_local_acceptance(tmp_path: Path):
    root = tmp_path / "batch"
    task_dir = root / "jobs" / "task_001"
    pool_dir = task_dir / "pool"
    ref_dir = root / "references" / "dft_refs"
    pool_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    # 1. Previous verified DFT reference
    ref_atoms = _make_atoms(1.0)
    ref_path = ref_dir / "dft_0000.vasp"
    ase_write(ref_path, ref_atoms, format="vasp", direct=True)
    ref_sha = _sha256(ref_path)

    # 2. Final phase diagram GS frame (not in previous DFT)
    gs_atoms = _make_atoms(1.1)
    gs_path = root / "gs_0.vasp"
    ase_write(gs_path, gs_atoms, format="vasp", direct=True)
    gs_sha = _sha256(gs_path)

    # 3. Trajectory frames saved in initial_relax_traj.json and pool traj gz
    init_traj = [
        {
            "structure": AseAtomsAdaptor.get_structure(_make_atoms(1.2 + 0.05 * i)).as_dict(),
            "energy_per_atom_std": 0.015,
            "f_std_max": 0.12,
            "ehull": 0.01 * i,
        }
        for i in range(4)
    ]
    (task_dir / "initial_relax_traj.json").write_text(json.dumps(init_traj), encoding="utf-8")

    mc_traj = [
        {
            "structure": AseAtomsAdaptor.get_structure(_make_atoms(1.5 + 0.05 * i)).as_dict(),
            "energy_per_atom_std": 0.020,
            "f_std_max": 0.15,
            "ehull": 0.02 * i,
        }
        for i in range(4)
    ]
    with gzip.open(pool_dir / "step_0_traj.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(mc_traj, stream)
    (pool_dir / "pool_summary.json").write_text(
        json.dumps([{"step": 0, "relax_traj_file": "step_0_traj.json"}]),
        encoding="utf-8",
    )

    snapshot = {
        "status": "completed",
        "entries": [
            {
                "structure_id": "GS-0",
                "structure_path": str(gs_path),
                "structure_sha256": gs_sha,
                "is_stable": True,
                "ehull": 0.0,
                "phase_identification_status": "identified",
                "composition": {"Na": 1, "Co": 1, "O": 2},
                "eform_per_O2": -1.0,
            }
        ],
    }
    state = {
        "phase_diagrams": {"mlip": snapshot},
        "dft_dataset_records": [
            {
                "task_id": "DFT-OLD",
                "status": "completed",
                "training_ready": True,
                "structure_sha256": ref_sha,
            }
        ],
        "tasks": [
            {
                "task_id": "MC-1",
                "stage": "deep_search",
                "status": "completed",
                "branch_id": "B1",
                "result_path": str(task_dir / "result.json"),
                "outputs": {},
            }
        ],
    }
    manifest = {
        "bundle_id": "bundle-1",
        "model_version": "mace-test",
        "dft_references": [
            {
                "task_id": "DFT-OLD",
                "path": "references/dft_refs/dft_0000.vasp",
                "sha256": ref_sha,
            }
        ],
    }

    def fake_extractor(atoms_list, **_kwargs):
        return np.array([[float(a.cell[0, 0]), float(len(a))] for a in atoms_list], dtype=float)

    result = run_remote_dft_sampling(
        root,
        state,
        manifest,
        target=5,
        seed=7,
        feature_extractor=fake_extractor,
    )
    assert result["gs_count"] == 1
    assert result["kmeans_count"] == 4
    assert len(result["selected"]) == 5
    assert result["selected"][0]["selection_mode"] == "phase_diagram_gs"
    assert result["selected"][0]["structure_sha256"] == gs_sha

    acceptance = verify_remote_dft_sampling(root, state, snapshot=snapshot)
    assert acceptance["status"] == "accepted"
    assert acceptance["selected_count"] == 5

    # Local recommendation reuses accepted remote sampling directly
    state["remote_dft_sampling"] = {"bundle-1": {**acceptance, "selected": result["selected"]}}
    rec = recommend_post_mc_dft([], {"dft": {"selection": {"max_tasks_per_round": 100}}}, state)
    assert len(rec["selected_candidates"]) == 5
    assert rec["summary"]["source"] == "remote_dft_sampling"

    # Tampered weight fails local acceptance
    tampered = json.loads((root / "dft_sampling_result.json").read_text(encoding="utf-8"))
    tampered["selected"][1]["weight"] = 999.0
    (root / "dft_sampling_result.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="remote_dft_sampling_weight_mismatch"):
        verify_remote_dft_sampling(root, state, snapshot=snapshot)


def test_outlier_hull_point_ignored_for_ehull_and_gs_but_retained_for_sampling(tmp_path: Path):
    root = tmp_path / "outlier_batch"
    task_dir = root / "jobs" / "task_001"
    (task_dir / "pool").mkdir(parents=True)

    # Outlier point on raw hull (e.g. extrapolation failure) + true normal GS point
    from phase_agent.science.qbc.remote_dft_sampling import _poscar_bytes

    outlier_atoms = _make_atoms(1.05)
    outlier_struct = AseAtomsAdaptor.get_structure(outlier_atoms)
    outlier_path = root / "outlier.vasp"
    outlier_path.write_bytes(_poscar_bytes(AseAtomsAdaptor.get_atoms(outlier_struct)))
    outlier_sha = _sha256(outlier_path)

    gs_atoms = _make_atoms(1.15)
    gs_path = root / "true_gs.vasp"
    ase_write(gs_path, gs_atoms, format="vasp", direct=True)
    gs_sha = _sha256(gs_path)

    # Put outlier in trajectory pool so it can still be sampled by kmeans++
    (task_dir / "initial_relax_traj.json").write_text(
        json.dumps(
            [
                {
                    "structure": outlier_struct.as_dict(),
                    "energy_per_atom_std": 0.05,
                    "f_std_max": 0.30,
                    "eform_per_O2": -5.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    snapshot = {
        "status": "completed",
        "entries": [
            {
                "structure_id": "OUTLIER-0",
                "structure_path": str(outlier_path),
                "structure_sha256": outlier_sha,
                "is_stable": True,
                "extrapolation_failure": True,
                "ehull": 0.0,
                "phase_identification_status": "identified",
                "composition": {"Na": 1, "Co": 1, "O": 2},
                "eform_per_O2": -5.0,
            },
            {
                "structure_id": "TRUE-GS",
                "structure_path": str(gs_path),
                "structure_sha256": gs_sha,
                "is_stable": False,
                "ehull": 0.5,
                "phase_identification_status": "identified",
                "composition": {"Na": 1, "Co": 1, "O": 2},
                "eform_per_O2": -1.0,
            },
        ],
    }
    state = {
        "phase_diagrams": {"mlip": snapshot},
        "dft_dataset_records": [],
        "tasks": [
            {
                "task_id": "MC-1",
                "stage": "deep_search",
                "status": "completed",
                "branch_id": "B1",
                "result_path": str(task_dir / "result.json"),
                "outputs": {},
            }
        ],
    }
    manifest = {"bundle_id": "b-out", "model_version": "mace-test", "dft_references": []}

    def fake_extractor(atoms_list, **_kwargs):
        return np.array([[float(a.cell[0, 0])] for a in atoms_list], dtype=float)

    result = run_remote_dft_sampling(
        root, state, manifest, target=5, seed=0, feature_extractor=fake_extractor
    )
    assert result["excluded_hull_outlier_count"] == 1
    assert result["gs_count"] == 1
    assert result["selected"][0]["structure_sha256"] == gs_sha
    assert result["kmeans_count"] == 1
    assert result["selected"][1]["structure_sha256"] == outlier_sha
    assert result["selected"][1]["selection_mode"] == "weighted_kmeans_pp"
    assert verify_remote_dft_sampling(root, state, snapshot=snapshot)["status"] == "accepted"


def test_parent_first_subsequent_extrapolation_uses_kmeans_pp(tmp_path: Path, monkeypatch):
    import sys
    import types
    from phase_agent.science.mlip.parent_first_sampling import run_parent_first_sampling

    parent_path = tmp_path / "parent.vasp"
    ref_path = tmp_path / "ref.vasp"
    model_path = tmp_path / "model.model"
    model_path.write_bytes(b"fake-model")
    ase_write(parent_path, _make_atoms(1.0), format="vasp", direct=True)
    ase_write(ref_path, _make_atoms(0.95), format="vasp", direct=True)

    def fake_perturb(final, out_dir, *, frames_per_parent, **_kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        struct_file = out_dir / "perturbed.extxyz"
        frames = [_make_atoms(1.2 + 0.1 * i) for i in range(frames_per_parent)]
        ase_write(struct_file, frames, format="extxyz")
        manifest = {
            "identities": [
                {"parent_record_id": "P1", "perturbation_index": i}
                for i in range(frames_per_parent)
            ]
        }
        mpath = out_dir / "manifest.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        return {"structures": str(struct_file), "manifest": str(mpath), "n_frames": frames_per_parent}

    fake_pkg = types.ModuleType("Process_MLIP")
    fake_sub = types.ModuleType("Process_MLIP.nepkit_sampling")
    fake_sub.perturb_layered_oxide_structure = fake_perturb
    monkeypatch.setitem(sys.modules, "Process_MLIP", fake_pkg)
    monkeypatch.setitem(sys.modules, "Process_MLIP.nepkit_sampling", fake_sub)

    def fake_relax(job):
        out = Path(job["output_directory"])
        out.mkdir(parents=True, exist_ok=True)
        relaxed = out / "initial_relaxed.vasp"
        ase_write(relaxed, _make_atoms(1.0), format="vasp", direct=True)
        return {
            "status": "completed",
            "optimizer_converged": True,
            "structure_path": str(relaxed),
            "structure_checksum": _sha256(relaxed),
            "extrapolation_monitor": {
                "final_feature_difference": 0.1,
                "extrapolation_failure": False,
                "failures": [],
            },
        }

    def fake_extract(atoms_list, **_kwargs):
        return np.array([[float(a.cell[0, 0])] for a in atoms_list], dtype=float)

    res = run_parent_first_sampling(
        [{"record_id": "P1", "path": str(parent_path), "sha256": _sha256(parent_path)}],
        [{"task_id": "DFT-0", "path": str(ref_path), "sha256": _sha256(ref_path)}],
        tmp_path / "out",
        model_path=str(model_path),
        model_sha256=_sha256(model_path),
        target=2,
        frames_per_parent=4,
        sampling_method="kmeans_pp",
        relax_fn=fake_relax,
        extract_fn=fake_extract,
    )
    assert res["status"] == "completed"
    assert res["selected_count"] == 2
    saved = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert saved["direct_report"]["method"] == "weighted_kmeans_pp"
    assert saved["direct_report"]["reference_count"] == 1


def test_post_dft_assessment_splits_gs_and_round_sample_errors():
    from phase_agent.evidence.post_dft_assessment import post_dft_assessment
    from phase_agent.runtime.post_dft_presentation import post_dft_lines
    from tests.test_dft_comparison_csv import add_result

    state = {}
    add_result(state, task_id="DFT-GS-1", energy_error=0.04, force_error=0.01)
    add_result(state, task_id="DFT-KM-1", energy_error=0.20, force_error=0.05)
    state["tasks"][0]["selection_mode"] = "phase_diagram_gs"
    state["tasks"][0]["is_ground_state"] = True
    state["tasks"][1]["selection_mode"] = "weighted_kmeans_pp"

    assessment = post_dft_assessment(state, {"mlip": {"version": "m1"}})
    assert assessment is not None
    assert assessment["gs_paired_structures"] == 1
    assert assessment["sample_paired_structures"] == 1
    assert math.isclose(assessment["gs_metrics"]["energy_per_atom"]["mae"], 0.020, rel_tol=1e-9)
    assert math.isclose(assessment["sample_metrics"]["energy_per_atom"]["mae"], 0.100, rel_tol=1e-9)

    text = post_dft_lines(state, assessment=assessment)
    assert "本轮 GS 帧（1 对）" in text
    assert "本轮采样数据（1 对）" in text

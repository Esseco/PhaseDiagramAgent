"""Actual three-project structures; real POSCAR/ASE IO, simulated calculator only."""

from pathlib import Path
import json
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Structure
from phase_agent.science.mlip import run_mace_subprocess as adapter
from phase_agent.science.mlip.predict_structure import predict_structure


@pytest.mark.parametrize("project", ["O3-1ele-NaCoO2", "O3-1ele-NaCrO2", "O3-2ele-NaFeMn"])
@pytest.mark.real_project
def test_actual_project_structure_roundtrip_with_explicit_format(tmp_path, monkeypatch, project):
    root = Path("E:/0-PhaseDiagram") / project
    state = json.loads((root / "workflow_state/state.json").read_text(encoding="utf-8"))
    active = state["active_model"]
    structure = None
    if "NaFeMn" not in project:
        returned = (
            root
            / "submissions/archives/01-upload/epoch0_DFTSP_Batch01_Recovery03_output/recovered.results.zip"
        )
        with zipfile.ZipFile(returned) as z:
            for name in z.namelist():
                if name.endswith("/result.json"):
                    row = json.loads(z.read(name))
                    if row.get("status") == "completed":
                        structure = Structure.from_dict(row["outputs"]["structure"])
                        break
    else:
        archives = {
            b.get("handoff", {}).get("upload_archive") for b in state.get("slurm_batches", [])
        }
        for archive in sorted(str(p) for p in archives if p):
            with zipfile.ZipFile(archive) as z:
                names = [n for n in z.namelist() if n.endswith("initial_structure.vasp")]
                if names:
                    structure = Structure.from_str(z.read(names[0]).decode(), fmt="poscar")
                    break
    assert structure is not None, project
    original_to = Structure.to

    def strict_write(self, *args, **kwargs):
        assert kwargs.get("fmt") == "poscar", "Remote pymatgen cannot infer input.vasp"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Structure, "to", strict_write)

    class FakeMACE:
        def __init__(self, **kwargs):
            assert kwargs["head"] == active.get("mace_head", "omat_pbe")

        def get_potential_energy(self, atoms, **kwargs):
            return -float(len(atoms))

        def get_forces(self, atoms):
            return np.zeros((len(atoms), 3))

    monkeypatch.setitem(sys.modules, "mace.calculators", SimpleNamespace(MACECalculator=FakeMACE))
    model = tmp_path / "original.model"
    model.write_bytes(b"fake weights; no real inference")
    calls = []

    def fake_subprocess(command, **kwargs):
        assert command[0] == sys.executable
        assert Path(command[-3]).name == "mace_worker.py"
        job = json.loads(Path(command[-2]).read_text())
        assert job["operation"] == "predict"
        calls.append(job)
        result = predict_structure(job)
        Path(command[-1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_subprocess)

    def forbidden(*a, **k):
        raise AssertionError("No actual subprocesses")

    monkeypatch.setattr(adapter.subprocess, "Popen", forbidden)
    result = adapter.run_mace_with_py_mace(
        structure,
        model_path=str(model),
        operation="predict",
        work_directory=tmp_path / "work",
        environment="current",
        parameters={"mace_head": active.get("mace_head", "omat_pbe")},
    )
    assert len(calls) == 1 and result["status"] == "completed"
    assert len(result["structure"]) == len(structure)
    assert np.allclose(result["structure"].lattice.matrix, structure.lattice.matrix)
    assert np.allclose(result["structure"].frac_coords, structure.frac_coords)
    assert result["energy_unit"] == "eV" and result["forces_unit"] == "eV/angstrom"
    assert result["energy"] == -len(structure)
    assert np.asarray(result["forces"]).shape == (len(structure), 3)
    print(
        project,
        "actual_structure_atoms",
        len(structure),
        "model",
        active.get("version"),
        flush=True,
    )

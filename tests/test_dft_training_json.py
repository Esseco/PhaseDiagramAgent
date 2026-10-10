import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read
from pymatgen.core import Lattice, Structure

from phase_agent.science.dft.vasp_training_labels import final_training_labels
from phase_agent.science.training.prepare_mace_finetune import prepare_mace_finetune
from phase_agent.analysis.feedback.dft_result_products import record_dft_products


def test_json_recovery_to_training_xyz(tmp_path):
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    labels = final_training_labels(SimpleNamespace(converged=True, ionic_steps=[{
        "structure": structure, "e_0_energy": -8.,
        "forces": [[.1, .2, .3], [-.1, -.2, -.3]], "stress": np.eye(3) * 10,
    }]))
    result = json.loads(json.dumps({"task_id": "T1", "structure_id": "S1",
        "stage": "dft_single_point", "status": "completed", "converged": True,
        "outputs": {"energy": -9., **labels}}))
    state = {}
    record_dft_products(state, result)
    assert state["new_dft_records"][0]["energy"] == -8.
    report = prepare_mace_finetune(state["new_dft_records"], tmp_path, config={
        "labels": {"include_stress": True}, "split": {"train": 1, "valid": 0, "test": 0}})
    assert report["accepted"] == 1
    atoms = read(tmp_path / "_shared_data" / "train.xyz")
    assert atoms.info["REF_energy"] == -8.
    np.testing.assert_allclose(atoms.arrays["REF_forces"], labels["forces"])
    np.testing.assert_allclose(np.asarray(atoms.info["REF_stress"]).reshape(3, 3),
                               -np.eye(3) * .006241509074460763)
    record_dft_products(state, result)
    assert len(state["new_dft_records"]) == 1


def test_missing_or_nonfinite_forces_rejected():
    structure = Structure(Lattice.cubic(4), ["Na"], [[0, 0, 0]])
    for forces in ([], [[float("nan"), 0, 0]]):
        with pytest.raises(ValueError):
            final_training_labels(SimpleNamespace(ionic_steps=[{
                "structure": structure, "e_0_energy": -1., "forces": forces}]))

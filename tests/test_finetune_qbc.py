from pathlib import Path

import numpy as np
from ase import Atoms

from scientific_layer.qbc.generate_qbc_file import generate_qbc_file
from scientific_layer.training.prepare_mace_finetune import prepare_mace_finetune


def test_group_split_and_bootstrap(tmp_path):
    records = []
    for index in range(10):
        records.append({"structure_id": f"S{index}", "branch_id": f"B{index // 2}", "status": "completed", "converged": True, "structure": Atoms("H", positions=[[0, 0, 0]]), "energy": float(index), "forces": [[0, 0, 0]]})
    report = prepare_mace_finetune(records, tmp_path, config={"seed": 1, "split": {"train": 0.6, "valid": 0.2, "test": 0.2}, "training": {"foundation_model": "mh-1.model"}, "committee": [{"seed": 1, "bootstrap_seed": 11}, {"seed": 2, "bootstrap_seed": 12}]})
    groups = [set(report["split_groups"][key]) for key in ("train", "valid", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert Path(report["committees"][0]["parameters"]["train_file"]).name == "train_bootstrap.xyz"


def test_qbc_file_contains_per_model_predictions(tmp_path):
    structure = Atoms("H", positions=[[0, 0, 0]])
    members = [{"model_id": f"m{i}", "model": i} for i in range(3)]
    committee = {"committee_id": "c1", "loaded_members": members}
    result = generate_qbc_file([{"structure_id": "S1", "branch_id": "B1", "structure": structure}], committee, tmp_path / "qbc.json", predictor=lambda structure, model, member: {"energy": float(model), "forces": np.full((1, 3), model)})
    assert result["status"] == "completed"
    assert len(result["results"][0]["qbc"]["predictions"]) == 3

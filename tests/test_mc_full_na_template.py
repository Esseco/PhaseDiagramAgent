from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from scientific_layer.mlip.slurm_executor import create_mlip_task_preparer
from scientific_layer.structures.build_mc_full_na_template import build_mc_full_na_template


H = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
BOUNDARY = {"P": ["O3", "O1"], "H": {"O3": [H], "O1": [H]},
            "TM_ratio": {"Fe": 1}}


def _parent(with_na):
    symbols = ["Na", "Fe", "O", "O"] if with_na else ["Fe", "O", "O"]
    coords = [[0, 0, 0], [0, 0, .5], [.25, .25, .25], [.75, .75, .75]]
    if not with_na:
        coords = coords[1:]
    return Structure(Lattice.cubic(4), symbols, coords)


def test_mc_template_uses_parent_h_and_t_not_relaxed_structure(tmp_path):
    parent = tmp_path / "O3.vasp"
    _parent(True).to(filename=str(parent), fmt="poscar")
    relaxed = tmp_path / "best_relaxed.vasp"
    relaxed.write_text("not the Na-site template", encoding="utf-8")
    branch = {"branch_id": "B1", "P": "O3", "H": H, "x": .5, "T": ["Fe", "Fe"]}
    manager = SimpleNamespace(boundary=BOUNDARY, data={"branches": {"B1": branch},
        "structures": {"S1": {"structure_id": "S1", "branch_id": "B1",
                              "source_path": str(tmp_path / "initial.vasp")}}})
    prepare = create_mlip_task_preparer(manager, {"O3": str(parent)},
                                       {"mlip": {"model_path": "remote.model"}})
    task = prepare({"task_id": "MC1", "stage": "deep_search", "structure_id": "S1",
                    "structure_path": str(relaxed), "incremental_budget": 20})
    template = Structure.from_file(task["worker_job"]["parameters"]["full_na_structure"])
    assert task["worker_job"]["structure_path"] == str(relaxed)
    assert template.composition.get("Na") == 2
    assert template.composition.get("Fe") == 2
    assert template.composition.get("O") == 4


def test_na_free_parent_requires_explicit_na_site_template(tmp_path):
    branch = {"branch_id": "B2", "P": "O1", "H": H, "x": .5, "T": ["Fe", "Fe"]}
    parent = tmp_path / "O1.vasp"
    template = tmp_path / "O1_Na_sites.vasp"
    _parent(False).to(filename=str(parent), fmt="poscar")
    _parent(True).to(filename=str(template), fmt="poscar")
    with pytest.raises(ValueError, match="没有 Na 位点"):
        build_mc_full_na_template(branch, BOUNDARY, {"O1": str(parent)})
    full = build_mc_full_na_template(
        branch, BOUNDARY, {"O1": str(parent)},
        config={"system": {"mc_full_na_templates": {"O1": str(template)}}})
    assert full.composition.get("Na") == 2

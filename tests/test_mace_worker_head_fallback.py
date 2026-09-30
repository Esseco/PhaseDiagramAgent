import sys
from types import ModuleType

from scientific_layer.mlip.mace_worker import run_mace_worker


def test_old_mh1_task_without_head_uses_omat_pbe(tmp_path, monkeypatch):
    captured = {}
    module = ModuleType("Process_AL_MC.relax")
    def relax(structure, model, **kwargs):
        captured.update(kwargs)
        from pymatgen.core import Lattice, Structure
        Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]]).to(
            filename=kwargs["output_path"], fmt="poscar")
        return {"status": "completed"}
    module.relax_structure_mace = relax
    monkeypatch.setitem(sys.modules, "Process_AL_MC.relax", module)
    result = run_mace_worker({"operation": "relax", "structure_path": "initial.vasp",
        "model_path": "/models/mace-mh-1.model", "model_version": "mace-mh-1",
        "output_directory": str(tmp_path), "parameters": {}})
    assert captured["head"] == "omat_pbe"
    assert result["status"] == "completed"

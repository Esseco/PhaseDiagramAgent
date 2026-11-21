import sys
from types import ModuleType

from phase_agent.science.mlip.mace_worker import run_mace_worker


def test_old_mh1_task_without_head_uses_omat_pbe(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io import write

    initial = tmp_path / "initial.vasp"
    write(initial, Atoms("NaO", positions=[[0, 0, 0], [2, 2, 2]], cell=[4, 4, 4], pbc=True))
    monkeypatch.setattr("phase_agent.science.mlip.mace_worker._extrapolation_monitor_factory", lambda: lambda atoms: object())
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
    result = run_mace_worker({"operation": "relax", "structure_path": str(initial),
        "model_path": "/models/mace-mh-1.model", "model_version": "mace-mh-1",
        "output_directory": str(tmp_path), "parameters": {}})
    assert captured["head"] == "omat_pbe"
    assert result["status"] == "completed"

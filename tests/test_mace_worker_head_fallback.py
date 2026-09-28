import sys
from types import ModuleType

from scientific_layer.mlip.mace_worker import run_mace_worker


def test_old_mh1_task_without_head_uses_omat_pbe(tmp_path, monkeypatch):
    captured = {}
    module = ModuleType("Process_AL_MC")

    class Runner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return {"status": "completed"}

    module.LayeredOxide_MCOrderingClass = Runner
    monkeypatch.setitem(sys.modules, "Process_AL_MC", module)
    result = run_mace_worker({"operation": "relax", "structure_path": "initial.vasp",
        "model_path": "/models/mace-mh-1.model", "model_version": "mace-mh-1",
        "output_directory": str(tmp_path), "parameters": {}})
    assert captured["mace_head"] == "omat_pbe"
    assert result["status"] == "failed"  # the mocked runner did not produce a pool structure

import sys
from types import SimpleNamespace
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes

from phase_agent.science.mlip.predict_structure import predict_structure


def test_main_prediction_not_committee_average_and_geometry_cache_preserved(tmp_path, monkeypatch):
    class FakeCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        def __init__(self, model_paths, **kwargs):
            super().__init__()
            self.value = float(model_paths)
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": self.value, "forces": np.full((len(atoms), 3), self.value)}
    monkeypatch.setitem(sys.modules, "mace", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mace.calculators", SimpleNamespace(MACECalculator=FakeCalculator))
    path = tmp_path / "initial.vasp"
    write(path, Atoms("Na", positions=[[0,0,0]], cell=[5,5,5], pbc=True), format="vasp")
    output = tmp_path / "output"
    output.mkdir()
    result = predict_structure({"structure_path": str(path), "output_directory": str(output),
        "model_paths": ["1", "3"], "model_version": "new",
        "parameters": {"model_refresh_operation": "predict", "main_model_index": 1}})
    assert result["energy"] == 3
    assert result["forces"] == [[3,3,3]]
    assert result["qbc"]["member_count"] == 2
    assert (output / "evaluated.vasp").read_bytes() == path.read_bytes()
    assert result["single_point_completed"] is True
    assert "converged" not in result

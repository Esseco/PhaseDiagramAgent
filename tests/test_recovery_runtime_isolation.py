"""Generated recovery packages execute in new interpreters without the checkout.

Only VASP XML parsing and the expensive calculator are replaced. POSCAR IO,
archive generation, bundled worker subprocess, receipts and return import run
through production code on small synthetic data.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.tools.remote import dft_reassessment_handoff as handoff
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.legacy_recovery import recovery_plan
from phase_agent.tools.remote.legacy_recovery_runner import terminal
from phase_agent.tools.remote.dft_mlip_pair import load_pair
from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
from phase_agent.tools.remote.transfer_bundle import create_bundle, import_bundle


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def snapshot(root):
    return {p.relative_to(root).as_posix(): file_checksum(p) for p in root.rglob("*") if p.is_file()}


def fixture(root):
    source = root / "epoch0_DFTSP_Batch01"
    worker = source / "DFT/inputs/batch/00000-DFT-test"
    worker.mkdir(parents=True)
    weights = root / "original.model"
    weights.write_bytes(b"fake model; no inference")
    model = {"version": "m1", "model_path": str(weights), "environment": "current",
             "mace_head": "omat_pbe", "model_sha256": file_checksum(weights)}
    structure = Structure(Lattice.cubic(5), ["Na", "Cr", "O", "O"],
                          [[0, 0, 0], [.5, .5, .5], [.25, .25, .25], [.75, .75, .75]])
    structure.to(filename=worker / "initial.vasp", fmt="poscar")
    task = {"task_id": "DFT-test", "task_key": "key-test", "batch_id": "batch",
            "config_version": "c1", "model_version": "m1", "stage": "dft_single_point",
            "task_checksum": "a" * 64, "protocol_version": 1, "input_file_version": 1}
    save(worker / "task.json", task)
    save(worker / "workflow.json", {"calculation": "static", "structure_file": "initial.vasp"})
    save(worker / "comparison_model.json", model)
    (worker / "GPU.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    save(source / "submission_plan.json", {"scripts": ["DFT/inputs/batch/00000-DFT-test/GPU.sh"]})
    create_bundle(source, [p for p in source.rglob("*") if p.is_file()], source / "inputs.zip", kind="inputs")
    original = {"folder": str(source), "upload_archive": str(source / "inputs.zip"),
                "import_directory": str(root / "imported"), "return_directory": str(root / "returned")}
    state = {
        "tasks": [{**task, "input_path": str(worker / "task.json")}],
        "slurm_batches": [{"batch_id": "batch", "task_ids": [task["task_id"]], "handoff": original}],
    }
    plan = recovery_plan(state, original)
    outputs = {"structure": structure.as_dict(), "composition": structure.composition.as_dict(),
               "atom_count": len(structure), "energy": -8.0, "training_energy": -8.0,
               "energy_unit": "eV", "forces": [[0., 0., 0.]] * len(structure),
               "forces_unit": "eV/angstrom", "training_ready": True,
               "training_frame_index": 0, "final_frame_index": 0, "final_frame_valid": True,
               "xml_complete": True, "electronic_converged": True}
    result = {**task, "status": "failed", "converged": False, "outputs": outputs,
              "error": "static unexpectedly changed the structure"}
    destination = source / plan["tasks"][0]["result_directory"]
    save(destination / "result.json", result)
    save(destination / "task.finished.json", {**plan["tasks"][0]["expected"], "status": "failed",
                                              "result_checksum": file_checksum(destination / "result.json")})
    returned = root / "old.results.zip"
    create_bundle(source, [destination / "result.json", destination / "task.finished.json"], returned, kind="results")
    save(worker / "workflow_state.json", {"stages": {"static": {
        "error": "static unexpectedly changed the structure", "directory": "runs/static/attempt_001"}}})
    final = worker / "runs/static/attempt_001/final.vasp"
    final.parent.mkdir(parents=True)
    structure.to(filename=final, fmt="poscar")
    return source, state, original, returned, plan


DRIVER = r'''
import json, os, runpy, site, subprocess, sys
from pathlib import Path

root, fake = map(Path, sys.argv[1:])
assert not any('Process_PhaseDiagram' in p or 'Py-Code' in p for p in sys.path)
# py1 has third-party scientific wheels in the user's site-packages. Adding
# that dependency directory does not add either source checkout or execute .pth.
sys.path.append(site.getusersitepackages())
ns = runpy.run_path(str(root / 'recover_results.py'))
work, plan, transfer = ns['unpack'](root, extra_payloads=ns['EXTRA_PAYLOADS'])
runtime = work / 'runtime'
sys.path.insert(0, str(runtime))
os.environ['PYTHONPATH'] = os.pathsep.join([str(runtime), str(fake)])
from Process_Vasp.workflows import atomate_runner
from pymatgen.core import Structure
atomate_runner._validate = lambda path, stage: Structure.from_str((path / 'final.vasp').read_text(), fmt='poscar')

# Enforce explicit serialization even on pymatgen versions that accept .vasp.
original_to = Structure.to
def explicit_to(self, *args, **kwargs):
    assert kwargs.get('fmt') == 'poscar'
    return original_to(self, *args, **kwargs)
Structure.to = explicit_to

original_popen = subprocess.Popen
def guarded_popen(command, *args, **kwargs):
    if isinstance(command, str) and command in ('lscpu', 'ver'):  # Scientific imports inspect the host.
        return original_popen(command, *args, **kwargs)
    assert command[0] == sys.executable and Path(command[-3]).name == 'mace_worker.py', command
    assert Path(command[-3]).resolve().is_relative_to(runtime.resolve())
    return original_popen(command, *args, **kwargs)
subprocess.Popen = guarded_popen
report = ns['collect'](root, extra_payloads=ns['EXTRA_PAYLOADS'], transform_factory=ns['_reassessment_factory'])
ns['diagnostics'](root, report)
assert report['status'] == os.environ.get('EXPECTED_RECOVERY_STATUS', 'complete'), report
import phase_agent.science.mlip.model_interface as interface
assert Path(interface.__file__).resolve().is_relative_to(runtime.resolve())
(root / 'test-report.json').write_text(json.dumps(report))
'''


@pytest.mark.parametrize("fail_first", [False, True])
def test_generated_recovery_archive_survives_isolated_worker_and_restart(tmp_path, fail_first):
    source, state, original, returned, plan = fixture(tmp_path)
    before = snapshot(source)
    stale = handoff.prepare_reassessment_handoff(state, original, source_directory=str(source), returned_archive=returned)
    with zipfile.ZipFile(Path(stale["return_directory"]) / "diagnostics.zip", "w") as archive:
        archive.writestr("dispatch-error.json", json.dumps({"error": "collection_incomplete:old format error"}))
    delivered = handoff.prepare_reassessment_handoff(state, original, source_directory=str(source), returned_archive=returned)
    folder = Path(delivered["folder"])
    assert not delivered["dft_recompute"]
    with zipfile.ZipFile(folder / "inputs.zip") as archive:
        assert "runtime/phase_agent/science/mlip/model_interface.py" in archive.namelist()
    fake = tmp_path / "fake"
    (fake / "mace").mkdir(parents=True)
    (fake / "mace/__init__.py").write_text("")
    (fake / "mace/calculators.py").write_text('''
import os
from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
class MACECalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, model_paths, head=None, device=None, **kwargs):
        super().__init__()
        assert Path(model_paths).is_file() and head == 'omat_pbe' and device == 'cuda'
        with open(os.environ['FAKE_INFERENCES'], 'a') as stream: stream.write('loaded\\n')
        failure = os.environ.get('FAKE_FAIL_ONCE')
        if failure and not Path(failure).exists():
            Path(failure).write_text('first prediction failed')
            raise RuntimeError('synthetic GPU unavailable')
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': -7.0, 'forces': np.zeros((len(atoms), 3))}
''')
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    count = tmp_path / "inferences.txt"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "",
                   "SLURM_JOB_ID": "synthetic", "PHASE_RECOVERY_SOURCE": str(source),
                   "FAKE_INFERENCES": str(count)}
    if fail_first:
        environment["FAKE_FAIL_ONCE"] = str(tmp_path / "failure.marker")
    for attempt in range(3 if fail_first else 2):
        # Independent interpreters simulate failure recovery and another restart.
        environment["EXPECTED_RECOVERY_STATUS"] = "incomplete" if fail_first and attempt == 0 else "complete"
        completed = subprocess.run([sys.executable, "-I", str(driver), str(folder), str(fake)],
                                   env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert count.read_text().splitlines() == ["loaded"] * (2 if fail_first else 1)
    assert snapshot(source) == before
    output = folder.with_name(folder.name[:-6] + "_output")
    imported = tmp_path / "verified"
    import_bundle(output / "recovered.results.zip", imported)
    record = plan["tasks"][0]
    result_dir = imported / record["result_directory"]
    result = terminal(result_dir, record["expected"])
    assert result["status"] == "completed"
    assert result["outputs"]["energy"] == -8.0
    assert load_pair(result, result_dir)["energy"] == -7.0
    assert json.loads((result_dir / "original_result.json").read_text())["status"] == "failed"
    collected = RemoteBatchRunner(tmp_path / "unused", worker_command=[]).collect_results_with_report(state)
    assert collected["report"]["invalid_count"] == 0, collected["report"]
    assert len(collected["results"]) == 1, collected["report"]
    assert collected["results"][0]["status"] == "completed"
    assert collected["results"][0]["outputs"]["remote_mlip_prediction"]["energy"] == -7.0
    assert len(collected["report"]["transfer_errors"]) == 1  # Old diagnostic remains visible.
    assert snapshot(source) == before


def test_missing_runtime_dependency_rejected_before_delivery(tmp_path, monkeypatch):
    source, state, original, returned, _ = fixture(tmp_path)
    monkeypatch.setattr(handoff, "RUNTIME", tuple(p for p in handoff.RUNTIME if not p.endswith("model_interface.py")))
    with pytest.raises(ValueError, match="reassessment_runtime_dependency_missing"):
        handoff.prepare_reassessment_handoff(state, original, source_directory=str(source), returned_archive=returned)
    assert not list(tmp_path.glob("*_Recovery*_input"))

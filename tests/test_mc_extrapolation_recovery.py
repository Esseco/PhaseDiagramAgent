import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ase import Atoms
from ase.io import write
from phase_agent.tools.remote.export_batch_result import export_batch_result
from phase_agent.tools.remote.resolve_local_relax_structure import resolve_local_relax_structure
from phase_agent.analysis.phase.phase_dft_candidates import extrapolation_dft_candidates
from phase_agent.tools.relax_mc_batch.return_payload import result_files
import pytest


@pytest.mark.parametrize('status', ['completed', 'failed'])
def test_recovered_mc_failures_are_bound_and_available_to_dft(status):
    with TemporaryDirectory(prefix='mc_') as folder:
        root = Path(folder)
        task = root / 'task'
        task.mkdir()
        atoms = Atoms('NaCoO2', positions=[[0,0,0],[1,1,1],[1,0,0],[0,1,0]], cell=[5,5,5], pbc=True)
        normal, bad = task/'final.vasp', task/'extrapolation_000000.vasp'
        write(normal, atoms, format='vasp')
        write(bad, atoms, format='vasp')
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        evidence = {'structure_path': str(bad), 'structure_checksum': sha(bad),
                    'extrapolation_monitor': {'extrapolation_failure': True, 'failures': [{'score': 2}]}}
        result = {'task_id': 't', 'task_key': 'k', 'stage': 'deep_search', 'status': status,
                  'outputs': {'structure_path': str(normal), 'structure_checksum': sha(normal),
                              'extrapolation_failures': [evidence]}}
        result_path = task/'result.json'
        result_path.write_text(json.dumps(result), encoding='utf8')
        (task/'task.finished.json').write_text(json.dumps({'task_id': 't','task_key': 'k',
            'status': status,'result_checksum': sha(result_path)}), encoding='utf8')
        destination = export_batch_result(task, root/'results', create_archive=False)
        pending = {'tasks': [{'task_id': 't', 'stage': 'deep_search', 'status': 'failed',
                              'result_path': str(destination/'result.json'),
                              'input_path': str(task/'task.json')} ]}
        returned = result_files(root, pending, {})
        assert destination / bad.name in returned
        local = resolve_local_relax_structure(result, destination)
        assert local is not None and Path(local['outputs']['extrapolation_failures'][0]['structure_path']).is_file()
        state = {'active_model_version': 'm', 'tasks': [{'task_id': 't', 'structure_id': 's',
                  'model_version': 'm', 'branch_id': 'b', 'outputs': local['outputs']}]}
        rows = extrapolation_dft_candidates(state, [])
        assert len(rows) == 1 and rows[0]['candidate_id'] == 'EXTRAP-t-0'
        assert rows[0]['structure_sha256'] == sha(bad) and rows[0]['predicted_Ehull'] is None
        (destination/bad.name).write_text('changed', encoding='utf8')
        assert resolve_local_relax_structure(result, destination) is None


def test_mc_observed_frames_export_and_rebind_with_checksum():
    with TemporaryDirectory(prefix="mc_frames_") as folder:
        root = Path(folder)
        task = root / "task"
        task.mkdir()
        normal = task / "final.vasp"
        frame = task / "mc_relaxation_frame_000000.vasp"
        normal.write_text("final", encoding="utf8")
        frame.write_text("observed", encoding="utf8")
        checksum = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        observation = {"structure_id": "s", "identity_scheme": "periodic_input_v1", "monitor_complete": True, "extrapolation_failure": True, "structure_path": str(frame), "structure_checksum": checksum(frame)}
        result = {"task_id": "t", "task_key": "k", "stage": "deep_search", "status": "completed", "outputs": {"structure_path": str(normal), "structure_checksum": checksum(normal), "relaxation_observations": [observation]}}
        source = task / "result.json"
        source.write_text(json.dumps(result), encoding="utf8")
        (task / "task.finished.json").write_text(json.dumps({"task_id": "t", "task_key": "k", "status": "completed", "result_checksum": checksum(source)}), encoding="utf8")
        destination = export_batch_result(task, root / "results", create_archive=False)
        assert (destination / frame.name).is_file()
        local = resolve_local_relax_structure(result, destination)
        assert local["outputs"]["relaxation_observations"][0]["structure_path"] == str((destination / frame.name).resolve())
        (destination / frame.name).write_text("changed", encoding="utf8")
        assert resolve_local_relax_structure(result, destination) is None


def test_mc_direct_frames_survive_minimal_zip_and_import():
    from phase_agent.tools.relax_mc_batch.files import rebase
    from phase_agent.tools.remote.transfer_bundle import create_bundle, import_bundle
    from phase_agent.tools.workflows.direct_extrapolation_sampling import _mc_batch_candidates
    with TemporaryDirectory(prefix="mc_round_") as folder:
        root = Path(folder) / "remote"
        worker = root / "task"
        worker.mkdir(parents=True)
        final = worker / "final.vasp"
        frame = worker / "mc_relaxation_frame_000000.vasp"
        final.write_text("final", encoding="utf8")
        frame.write_text("observed", encoding="utf8")
        checksum = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        observation = {"structure_id": "obs", "identity_scheme": "periodic_input_v1", "monitor_complete": True, "extrapolation_failure": True, "structure_path": str(frame), "structure_checksum": checksum(frame)}
        outputs = {"structure_path": str(final), "structure_checksum": checksum(final), "actual_mc_steps": 0, "relaxation_observations_complete": True, "relaxation_observations": [observation]}
        raw = {"task_id": "t", "task_key": "k", "stage": "deep_search", "status": "completed", "outputs": outputs}
        path = worker / "result.json"
        path.write_text(json.dumps(raw), encoding="utf8")
        (worker / "task.finished.json").write_text(json.dumps({"task_id": "t", "task_key": "k", "status": "completed", "result_checksum": checksum(path)}), encoding="utf8")
        dest = export_batch_result(worker, root / "results", create_archive=False)
        resolved = resolve_local_relax_structure(raw, dest)
        task = {"task_id": "t", "task_key": "k", "structure_id": "s", "branch_id": "b", "model_version": "m", "stage": "deep_search", "status": "completed", "input_path": str(worker / "task.json"), "result_path": str(dest / "result.json"), "outputs": resolved["outputs"]}
        state = {"active_model_version": "m", "tasks": [task]}
        files = result_files(root, state, {})
        archive = Path(folder) / "results.zip"
        create_bundle(root, files, archive, kind="results")
        local = Path(folder) / "local"
        import_bundle(archive, local)
        portable = rebase(state, root, portable=True)
        restored = rebase(portable, local)
        rows, _ = _mc_batch_candidates({"t": restored["tasks"][0]}, {"t"}, "m")
        assert len(rows) == 1 and rows[0]["structure_sha256"] == checksum(frame)
        assert Path(rows[0]["structure_path"]).is_file()

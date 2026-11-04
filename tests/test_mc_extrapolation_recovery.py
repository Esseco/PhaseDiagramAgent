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
    with TemporaryDirectory(prefix='mc_', dir=Path.cwd()) as folder:
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

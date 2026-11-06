"""Synthetic validation of HPC DFT preparation after DIRECT selection."""
import hashlib
import json
from pathlib import Path

import pytest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_prepares_only_verified_selected_frames_and_binds_dft_identity(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import parent_first_result
    from phase_agent.science.dft import prepare_pycode_relax
    from phase_agent.tools.workflows.parent_first_dft import prepare_parent_first_dft

    source = tmp_path / 'selected_0000.vasp'
    source.write_bytes(b'POSCAR data')
    manifest = tmp_path / 'input_manifest.json'
    settings = {'config_version': 'config-1', 'model_version': 'model-1',
                'approval_record_id': 'approved-1', 'parameters': {}}
    manifest.write_text(json.dumps({'dft_settings': settings}), encoding='utf8')
    result = tmp_path / 'result.json'
    result.write_bytes(b'sampling')
    checked = {'status': 'verified', 'target': 1, 'result_sha256': sha(result),
               'binding': {'input_manifest_sha256': sha(manifest), 'scope_id': 'scope',
                           'round_evidence_digest': 'digest', 'baseline_hull_version': 'hull'},
               'selected': [{'structure_path': str(source), 'structure_sha256': sha(source),
                             'parent_record_id': 'parent', 'branch_id': 'branch'}]}
    monkeypatch.setattr(parent_first_result, 'validate_parent_first_result', lambda *args: checked)

    def prepare(task, *, manager, comparison_model=None):
        assert task['stage'] == 'dft_single_point'
        assert Path(task['structure_path']).read_bytes() == source.read_bytes()
        Path(task['work_directory'], 'initial_structure.vasp').write_bytes(source.read_bytes())
        Path(task['work_directory'], 'workflow.py').write_text('# workflow', encoding='utf8')
        return {'generator': 'atomate', 'backend': 'pycode_atomate2_static', 'status': 'pending'}

    monkeypatch.setattr(prepare_pycode_relax, 'prepare_pycode_relax', prepare)
    output = tmp_path / 'dft'
    receipt = prepare_parent_first_dft(tmp_path, tmp_path, output, settings)
    assert receipt['task_count'] == 1 and receipt['submitted'] is False
    entry = receipt['tasks'][0]
    task_dir = output / entry['directory']
    task = json.loads((task_dir / 'task.json').read_text(encoding='utf8'))
    assert task['task_id'] == entry['task_id'] and task['input_structure_sha256'] == sha(source)
    assert task['parent_decision_id'] == 'approved-1' and task['branch_id'] == 'branch'
    assert (task_dir / 'initial_structure.vasp').read_bytes() == source.read_bytes()
    assert task['stage'] == 'dft_single_point' and task['status'] == 'pending'
    with pytest.raises(FileExistsError, match='parent_first_dft_exists'):
        prepare_parent_first_dft(tmp_path, tmp_path, output, settings)
    with pytest.raises(ValueError, match='parent_first_dft_settings_binding_invalid'):
        prepare_parent_first_dft(tmp_path, tmp_path, tmp_path/'other',
                                 {**settings, 'model_version': 'different-model'})


def test_refuses_changed_selected_frame_before_preparing_any_dft(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import parent_first_result
    from phase_agent.tools.workflows.parent_first_dft import prepare_parent_first_dft

    source = tmp_path / 'selected.vasp'; source.write_bytes(b'changed')
    settings = {'config_version': 'config-1', 'model_version': 'model-1',
                'approval_record_id': 'approved-1', 'parameters': {}}
    manifest = tmp_path / 'input_manifest.json'
    manifest.write_text(json.dumps({'dft_settings': settings}), encoding='utf8')
    checked = {'status': 'verified', 'target': 1, 'result_sha256': 'a' * 64,
               'binding': {'input_manifest_sha256': sha(manifest), 'scope_id': 'scope',
                           'round_evidence_digest': 'digest', 'baseline_hull_version': 'hull'},
               'selected': [{'structure_path': str(source), 'structure_sha256': '0' * 64,
                             'parent_record_id': 'parent', 'branch_id': 'branch'}]}
    monkeypatch.setattr(parent_first_result, 'validate_parent_first_result', lambda *args: checked)
    with pytest.raises(ValueError, match='parent_first_dft_frame_changed'):
        prepare_parent_first_dft(tmp_path, tmp_path, tmp_path / 'out',
            settings)
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.parent-first-dft-*'))


def test_uses_real_static_generator_without_executing_vasp(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io import write
    from phase_agent.tools.workflows import parent_first_result
    from phase_agent.tools.workflows.parent_first_dft import prepare_parent_first_dft
    from phase_agent.tools.remote.integrity import payload_checksum

    source = tmp_path / 'selected.vasp'
    write(source, Atoms('NaCoO2', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                        cell=[6, 6, 6], pbc=True), format='vasp')
    settings = {'config_version': 'config-1', 'model_version': 'model-1',
                'approval_record_id': 'approved-1', 'parameters': {}}
    manifest = tmp_path / 'input_manifest.json'
    manifest.write_text(json.dumps({'dft_settings': settings}), encoding='utf8')
    checked = {'status': 'verified', 'target': 1, 'result_sha256': 'a' * 64,
               'binding': {'input_manifest_sha256': sha(manifest), 'scope_id': 'scope',
                           'round_evidence_digest': 'digest', 'baseline_hull_version': 'hull'},
               'selected': [{'structure_path': str(source), 'structure_sha256': sha(source),
                             'parent_record_id': 'parent', 'branch_id': 'branch'}]}
    monkeypatch.setattr(parent_first_result, 'validate_parent_first_result', lambda *args: checked)
    output = tmp_path / 'out'
    receipt = prepare_parent_first_dft(tmp_path, tmp_path, output,
        settings)
    task_dir = output / receipt['tasks'][0]['directory']
    task = json.loads((task_dir / 'task.json').read_text(encoding='utf8'))
    checksum = task.pop('task_checksum')
    assert checksum == payload_checksum(task)
    config = json.loads((task_dir / 'workflow.json').read_text(encoding='utf8'))
    assert config['calculation'] == 'static'
    assert config['run_vasp_kwargs']['static']['max_errors'] == 2
    assert (task_dir / 'workflow.py').is_file() and (task_dir / 'submit_gpu.sh').is_file()
    assert (task_dir / 'initial_structure.vasp').read_bytes() == source.read_bytes()

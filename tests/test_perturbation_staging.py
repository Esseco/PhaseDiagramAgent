from pathlib import Path
import json
from hashlib import sha256
from copy import deepcopy

import pytest
from ase import Atoms
from ase.io import write

from phase_agent.tools.workflows import perturbation_staging as staging


def fixture_data(tmp_path, monkeypatch):
    path = tmp_path / 'parent.extxyz'
    write(path, Atoms('NaCoO2', positions=[[0,0,0], [2,2,2], [1,1,1], [3,3,3]],
                      cell=[5,5,5], pbc=True))
    digest = sha256(path.read_bytes()).hexdigest()
    parent = {'record_id': 'r', 'structure_id': 's', 'branch_id': 'b', 'structure_path': str(path),
              'structure_sha256': digest, 'phase': 'O3', 'x_Na_per_O2': 0.5,
              'H': [[2,0,0],[0,1,0],[0,0,1]], 'det_H': 2}
    monkeypatch.setattr(staging, 'required_direct_round', lambda state: {'scope_id': 'scope', 'evidence_digest': 'evidence'})
    monkeypatch.setattr(staging, 'build_perturbation_recovery_pool', lambda *a, **kw: {
        'candidates': [parent], 'hull_version': 'h1', 'model_version': 'old'})
    state = {'active_model_version': 'bad', 'model_registry': {'mace-mh-1': {
        'model': {'framework': 'mace', 'version': 'mace-mh-1', 'model_path': '/models/mh1.model', 'mace_head': 'omat_pbe'},
        'comparison_model_sha256': 'a' * 64}}}
    params = {'scope_id': 'scope', 'round_evidence_digest': 'evidence', 'confirmed': True,
              'parent_record_ids': ['r'], 'max_det_H': 2, 'frames_per_parent': 3, 'seed': 42}
    return state, params, path


def test_generated_frames_are_bound_to_parent_and_mh1(tmp_path, monkeypatch):
    state, params, _ = fixture_data(tmp_path, monkeypatch)
    before = deepcopy(state)
    target = tmp_path / 'batch'
    result = staging.prepare_perturbation_inputs(state, {}, params, target)
    data = json.loads((target / 'manifest.json').read_text(encoding='utf8'))
    assert result['status'] == 'structures_prepared' and not result['submitted']
    assert state == before and data['failed_model_version'] == 'bad'
    assert data['relax_model']['version'] == 'mace-mh-1'
    assert data['relax_model']['remote_content_verified'] is False
    assert len(data['frames']) == result['frame_count'] == 3
    assert all(sha256((target / f['path']).read_bytes()).hexdigest() == f['sha256'] for f in data['frames'])
    assert staging.prepare_perturbation_inputs(state, {}, params, target)['reused'] is True
    (target / data['frames'][0]['path']).write_text('changed')
    with pytest.raises(FileExistsError, match='require_recovery'):
        staging.prepare_perturbation_inputs(state, {}, params, target)


@pytest.mark.parametrize('field,value,error', [
    ('scope_id', 'other', 'round_evidence'), ('confirmed', False, 'not_confirmed'),
    ('parent_record_ids', ['r', 'r'], 'parent_selection'), ('frames_per_parent', 0, 'parameters_invalid')])
def test_invalid_inputs_do_not_write(tmp_path, monkeypatch, field, value, error):
    state, params, _ = fixture_data(tmp_path, monkeypatch)
    params[field] = value
    with pytest.raises(ValueError, match=error):
        staging.prepare_perturbation_inputs(state, {}, params, tmp_path / 'batch')
    assert not (tmp_path / 'batch').exists()


def test_parent_change_aborts_without_partial_batch(tmp_path, monkeypatch):
    state, params, path = fixture_data(tmp_path, monkeypatch)
    path.write_text('changed')
    with pytest.raises(ValueError, match='parent_changed'):
        staging.prepare_perturbation_inputs(state, {}, params, tmp_path / 'batch')
    assert not (tmp_path / 'batch').exists()
    assert not list(tmp_path.glob('.perturbation-*'))


def test_bad_model_binding_is_rejected(tmp_path, monkeypatch):
    state, params, _ = fixture_data(tmp_path, monkeypatch)
    state['model_registry']['mace-mh-1']['comparison_model_sha256'] = None
    with pytest.raises(ValueError, match='mh1_binding'):
        staging.prepare_perturbation_inputs(state, {}, params, tmp_path / 'batch')


def test_sampler_parameters_pass_through_to_process_mlip(tmp_path, monkeypatch):
    from Process_MLIP import nepkit_sampling
    state, params, _ = fixture_data(tmp_path, monkeypatch)
    params.update(min_distance_A=0.9, volume_ratio=[0.8, 1.2])
    original = nepkit_sampling.perturb_layered_oxide_structure
    calls = []
    def checked(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(nepkit_sampling, 'perturb_layered_oxide_structure', checked)
    target = tmp_path / 'batch'
    staging.prepare_perturbation_inputs(state, {}, params, target)
    assert calls[0]['min_distance_A'] == 0.9
    assert calls[0]['volume_ratio'] == [0.8, 1.2]
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf8'))
    assert manifest['parameters']['min_distance_A'] == 0.9
    assert manifest['parameters']['volume_ratio'] == [0.8, 1.2]


def test_staging_binds_only_successful_dft_as_direct_reference(tmp_path, monkeypatch):
    state, params, _ = fixture_data(tmp_path, monkeypatch)
    frame = {'lattice': {'matrix': [[1,0,0],[0,1,0],[0,0,1]]}, 'sites': []}
    valid = {'task_id': 'done', 'status': 'completed', 'checks_passed': True,
             'converged': True, 'final_frame_valid': True, 'training_ready': True,
             'structure': frame}
    failed = {**valid, 'task_id': 'failed', 'converged': False}
    state['dft_dataset_records'] = [valid, failed]
    target = tmp_path / 'batch'
    staging.prepare_perturbation_inputs(state, {}, params, target)
    result = json.loads((target / 'manifest.json').read_text(encoding='utf8'))
    assert result['direct_dft_references'] == [{'task_id': 'done', 'structure': frame}]


def test_approved_perturbation_prepares_only_mh1_inputs(tmp_path, monkeypatch):
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
    from phase_agent.tools.local.prepare_perturbation_relax import prepare_perturbation_relax
    state, params, source = fixture_data(tmp_path, monkeypatch)
    h = [[2,0,0],[0,1,0],[0,0,1]]
    manager = PhaseDataManager({'P': ['O3'], 'H': {'O3': [h]}, 'TM_ratio': {'Co': 1}})
    bid = manager.add_branch(P='O3', H=h, x=0.5, T=['Co'], composition={'Na': 1, 'Co': 1, 'O': 2})
    digest = sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(staging, 'build_perturbation_recovery_pool', lambda *a, **kw: {
        'hull_version': 'h1', 'model_version': 'old', 'candidates': [{'record_id':'r', 'structure_id':'s',
        'branch_id':bid, 'structure_path':str(source), 'structure_sha256':digest, 'phase':'O3', 'x_Na_per_O2':0.5, 'H':h, 'det_H':2}]})
    ledger_path = tmp_path / 'ledger.json'
    manager.save(ledger_path)
    config = {'budgets': default_budget_rules(), 'ledger_path': str(ledger_path), 'upload_batches_directory': str(tmp_path / 'uploads'),
              'supercomputer': {'worker': {'command': ['python', '-m', 'phase_agent.tools.remote.run_slurm_array_task',
                   '--executor', 'phase_agent.science.mlip.slurm_executor:execute_mlip_task']}}}
    context = {'event_state': state, 'manager': manager, 'effective_config': config,
               'config_version': 'cfg', 'approval_record_id': 'review'}
    result = prepare_perturbation_relax(action={'parameters': params}, context=context)
    assert result['status'] == 'prepared' and result['submitted'] is False
    assert result['state']['active_model_version'] == 'bad'
    batch = result['state']['perturbation_recovery_batch']
    assert batch['status'] == 'waiting_results' and len(batch['task_ids']) == 3
    assert result['state']['upload_layout']['model_rounds'] == {'bad': 1}
    prepared_batch = result['state']['slurm_batches'][0]
    assert prepared_batch['model_version'] == 'mace-mh-1' and prepared_batch['upload_epoch_model_version'] == 'bad'
    assert Path(prepared_batch['upload_directory']).is_relative_to(tmp_path / 'uploads' / 'epoch0')
    tasks = [t for t in result['state']['tasks'] if t['task_id'] in batch['task_ids']]
    assert all(t['model_version'] == 'mace-mh-1' and t['model_sha256'] == 'a' * 64 and t.get('input_path') for t in tasks)
    import zipfile
    with zipfile.ZipFile(result['handoff']['upload_archive']) as archive:
        task_files = [name for name in archive.namelist() if name.endswith('/task.json')]
        assert len(task_files) == 3
        payloads = [json.loads(archive.read(name)) for name in task_files]
        assert all(row['model_sha256'] == 'a' * 64 and row['worker_job']['model_version'] == 'mace-mh-1' for row in payloads)
        assert all(row['worker_job']['model_file_evidence'][0]['sha256'] == 'a' * 64 for row in payloads)
    assert prepare_perturbation_relax(action={'parameters': params}, context={**context, 'event_state': result['state']})['status'] == 'already_prepared'


def test_perturbation_recovery_rejects_tm_site_exchange():
    from phase_agent.tools.workflows.perturbation_direct import _tm_order_preserved
    initial = Atoms('NaNiCoO2', positions=[[0,0,0],[1,1,1],[4,4,4],[2,2,2],[5,5,5]],
                    cell=[6,6,6], pbc=True)
    relaxed = initial.copy()
    relaxed.set_cell([6.1,6.1,6.1], scale_atoms=True)
    assert _tm_order_preserved(initial, relaxed)
    first, second = relaxed.positions[1].copy(), relaxed.positions[2].copy()
    relaxed.positions[1], relaxed.positions[2] = second, first
    assert relaxed.get_chemical_symbols() == initial.get_chemical_symbols()
    assert not _tm_order_preserved(initial, relaxed)


def test_legacy_local_perturb_before_relax_cannot_dispatch():
    from phase_agent.tools.local.prepare_local_batch_files import prepare_local_batch_files
    result = prepare_local_batch_files(action={'parameters':{'mode':'perturbation_recovery_inputs'}}, context={})
    assert result['status'] == 'not_configured'
    assert result['submitted'] is False
    assert result['reason'] == 'legacy_perturb_before_relax_disabled_require_hpc_parent_first'


def test_agent_cannot_run_perturbation_direct_locally():
    from phase_agent.tools.local.prepare_local_batch_files import prepare_local_batch_files
    result = prepare_local_batch_files(action={'parameters':{'mode':'direct_extrapolation',
        'recovery_route':'perturbation'}}, context={})
    assert result['status'] == 'not_configured' and result['submitted'] is False
    assert result['reason'] == 'local_perturbation_direct_disabled_require_hpc_sampling'

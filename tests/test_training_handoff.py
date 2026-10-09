import json
from copy import deepcopy
from pathlib import Path

from execution_layer.local.training_handoff import advance_training_handoffs
from tests.test_remote_training_recovery import returned_job


def setup(tmp_path):
    job, results = returned_job(tmp_path)
    inputs = Path(job['directory']) / 'inputs'
    inputs.mkdir()
    (inputs / 'training_plan.json').write_text('{}')
    (inputs / '_shared_data').mkdir()
    (inputs / '_shared_data/train.xyz').write_text('training')
    state = {'active_model_version': 'base', 'remote_finetune_jobs': {'j': job}}
    return state, results, inputs


def config(tmp_path):
    data = tmp_path / 'test.xyz'
    data.write_text('independent')
    return {'mlip': {'version': 'base', 'model_path': '/hpc/base.model', 'sha256': 'b'*64},
        'remote_training_validation': {'data_path': str(data), 'data_version': 'independent-v1',
            'ranking_pairs': [[0, 1]], 'criteria': {'max_energy_mae': .01,
                'max_critical_failure_fraction': 0, 'max_near_hull_ranking_reversals': 0,
                'max_force_rmse': .1}}}


def test_manifest_job_and_restart_preserve_training_and_mtime(tmp_path):
    state, results, inputs = setup(tmp_path)
    models = json.loads((results / 'models.json').read_text())
    models[0]['sha256'] = ''
    (results / 'models.json').write_text(json.dumps(models))
    (inputs / 'run_training.sh').write_text('DO NOT CHANGE')
    before = deepcopy(state)
    updated, waits = advance_training_handoffs(state, {})
    assert state == before and waits[0]['stage'] == 'awaiting_manifest'
    script = inputs / 'GPU_manifest.sh'
    text = script.read_text()
    assert '#SBATCH --partition=v100m3' in text
    assert 'python collect_training_results.py --manifest-only' in text
    assert 'run_training.sh' not in text
    mtime = script.stat().st_mtime_ns
    again, _ = advance_training_handoffs(json.loads(json.dumps(updated)), {})
    assert again == updated and script.stat().st_mtime_ns == mtime
    assert (inputs / 'run_training.sh').read_text() == 'DO NOT CHANGE'
    models[0]['sha256'] = 'a'*64
    (results / 'models.json').write_text(json.dumps(models))
    _, waits = advance_training_handoffs(again, {})
    assert waits[0]['stage'] == 'validation_prerequisites_required'


def test_validation_request_recovery_and_separate_activation(tmp_path):
    state, results, inputs = setup(tmp_path)
    settings = config(tmp_path)
    updated, waits = advance_training_handoffs(state, settings)
    assert waits[0]['stage'] == 'awaiting_validation'
    plan = json.loads((inputs / 'validation_request.json').read_text())
    assert plan['old_model']['version'] == 'base'
    assert plan['training_sha256']
    metrics = {'energy_mae': .003, 'force_rmse': .06,
        'critical_failure_fraction': 0, 'near_hull_ranking_reversals': 0}
    report = {'request_id': plan['request_id'], 'data_sha256': plan['data_sha256'],
        'status': 'completed', 'structures': 2, 'metrics': {'old_model': {**metrics, 'energy_mae': .009}, 'new_model': metrics}}
    (results / ('validation-' + plan['request_id'] + '.json')).write_text(json.dumps(report))
    updated, waits = advance_training_handoffs(updated, settings)
    assert waits[0]['stage'] == 'awaiting_activation_approval'
    candidate = updated['candidate_models'][waits[0]['candidate_model_version']]
    assert candidate['validation']['passed'] is True
    assert updated['active_model_version'] == 'base' and 'active_model' not in updated
    from execution_layer.local.review_candidate_command import review_candidate_command
    assert review_candidate_command('继续', updated) is None
    assert review_candidate_command('同意', updated) is None
    reviewed = review_candidate_command('激活候选 ' + candidate['model']['version'] + ' 原因：独立验证符合标准', updated)
    assert reviewed['state']['active_model_version'] == candidate['model']['version']
    target = results / ('validation-' + plan['request_id'] + '.json')
    original = target.read_bytes()
    target.write_bytes(original + b' ')
    refused = review_candidate_command('激活候选 ' + candidate['model']['version'] + ' 原因：批准', updated)
    assert refused['state']['active_model_version'] == 'base'
    target.write_bytes(original)
    assert advance_training_handoffs(updated, settings)[0] == updated
    updated['active_model_version'] = candidate['model']['version']
    updated, waits = advance_training_handoffs(updated, settings)
    assert waits == [] and updated['remote_finetune_jobs']['j']['activated']


def test_missing_data_does_not_prepare_fake_validation(tmp_path):
    state, _, inputs = setup(tmp_path)
    updated, waits = advance_training_handoffs(state, {})
    assert waits[0]['stage'] == 'validation_prerequisites_required'
    assert not (inputs / 'GPU_validation.sh').exists()
    assert 'candidate_models' not in updated


def test_stale_and_nonfinite_validation_rejected(tmp_path):
    state, results, inputs = setup(tmp_path)
    settings = config(tmp_path)
    updated, waits = advance_training_handoffs(state, settings)
    plan = json.loads((inputs / 'validation_request.json').read_text())
    target = results / ('validation-' + plan['request_id'] + '.json')
    target.write_text(json.dumps({'request_id': 'wrong'}))
    updated, waits = advance_training_handoffs(updated, settings)
    assert waits[0]['stage'] == 'handoff_error'
    assert 'candidate_models' not in updated
    metrics = {'energy_mae': float('nan')}
    target.write_text(json.dumps({'request_id': plan['request_id'], 'data_sha256': plan['data_sha256'],
        'status': 'completed', 'structures': 2, 'metrics': {'old_model': metrics, 'new_model': metrics}}))
    _, waits = advance_training_handoffs(updated, settings)
    assert waits[0]['stage'] == 'handoff_error'


def test_gate_persists_handoff_and_extracts_memory(tmp_path):
    from execution_layer.workflows.lifecycle_recovery import _workflow_wait_gate
    state, _, _ = setup(tmp_path)
    updated, waits = advance_training_handoffs(state, {})
    path = tmp_path / 'state.json'
    frame = {'recovery_question': None, 'execution_mode': 'interactive', 'feedback': {'state': updated},
        'state_path': str(path), 'effective_config': {}, 'recovered_count': 0, 'collection_report': None,
        'snapshot': {}, 'pre_reconciled': {}, 'manual_wait': None, 'rebuilding': False,
        'training_handoffs': waits}
    response = _workflow_wait_gate(frame)
    assert response['status'] == 'training_handoff'
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved['remote_finetune_jobs']['j']['training_handoff']['stage'] == waits[0]['stage']
    assert any(c['kind'] == 'training_handoff' for c in saved['memory_candidates'])


def test_reply_preserves_handoff_instructions():
    from run.workflow_reply_presentation import format_workflow_reply
    assert format_workflow_reply({'status': 'training_handoff', 'reason': 'sbatch GPU_manifest.sh'}, 'state.json') == 'sbatch GPU_manifest.sh'

def test_remote_validator_checks_independence_and_model_hash(tmp_path, monkeypatch):
    import sys
    import types
    import numpy as np
    from ase import Atoms
    from ase.io import write
    from ase.calculators.calculator import Calculator, all_changes
    import execution_layer.remote.validate_remote_training as remote
    from execution_layer.local.training_handoff import digest_file
    root = tmp_path / 'inputs'
    root.mkdir()
    (root / '_shared_data').mkdir()
    monkeypatch.setattr(remote, '__file__', str(root / 'validate_remote_training.py'))
    train = Atoms('H2', positions=[[0,0,0],[1,0,0]])
    write(root / '_shared_data/train.xyz', [train], format='extxyz')
    frames = []
    for length in [2.,3.]:
        a = Atoms('H2', positions=[[0,0,0],[length,0,0]])
        a.info['REF_energy'] = length
        a.arrays['REF_forces'] = np.zeros((2,3))
        frames.append(a)
    write(root / 'validation.xyz', frames, format='extxyz')
    model = root / 'model'
    model.write_bytes(b'weights')
    plan = {'request_id': 'test', 'data_sha256': digest_file(root/'validation.xyz'),
        'training_sha256': digest_file(root/'_shared_data/train.xyz'), 'ranking_pairs': [[0,1]],
        'energy_key': 'REF_energy', 'forces_key': 'REF_forces',
        'old_model': {'model_path': str(model), 'sha256': digest_file(model)},
        'new_model': {'model_path': str(model), 'sha256': digest_file(model)}}
    class FakeCalculator(Calculator):
        implemented_properties = ['energy','forces']
        def __init__(self, **kwargs):
            assert kwargs['device'] == 'cuda'
            super().__init__()
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {'energy': atoms.positions[1,0], 'forces': np.zeros((2,3))}
    monkeypatch.setitem(sys.modules, 'mace.calculators', types.SimpleNamespace(MACECalculator=FakeCalculator))
    report = remote.evaluate(plan)
    assert report['structures'] == 2
    assert report['metrics']['new_model']['energy_mae'] == 0
    assert report['metrics']['new_model']['near_hull_ranking_reversals'] == 0
    import pytest
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Model hash mismatch'):
        remote.evaluate(plan)
    write(root / '_shared_data/train.xyz', frames, format='extxyz')
    plan['training_sha256'] = digest_file(root/'_shared_data/train.xyz')
    with pytest.raises(ValueError, match='overlap'):
        remote.evaluate(plan)

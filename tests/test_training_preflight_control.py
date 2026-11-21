import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from phase_agent.runtime.training_preflight_control import regenerate_preflight


def setup_fakes(tmp_path, monkeypatch, *, save_error=False, changed_identity=False):
    import phase_agent.persistence.state.read_versions as versions
    import phase_agent.tools.step_runner.file_protocol as files
    import phase_agent.tools.workflows.lifecycle_support as lifecycle
    import phase_agent.configuration.runtime.build_effective_run_config as build
    import phase_agent.runtime.project_context as context
    import phase_agent.analysis.convergence.training_provenance as provenance
    import phase_agent.science.training.training_preflight_selection as select
    import phase_agent.science.training.prepare_main_model as main
    import phase_agent.tools.local.regenerate_finetune_inputs as regen

    state_path = tmp_path / 'state.json'
    state_path.write_text('{}')
    folder = tmp_path / 'MLIP-finetune-round-0001'
    folder.mkdir()
    (folder / 'old.txt').write_text('old')
    backup = tmp_path / 'old-backup'
    state = {'active_model': {'version': 'model'}, 'remote_finetune_jobs': {
        'job': {'status': 'inputs_prepared', 'submitted': False, 'activated': False,
                'original_model_version': 'model', 'directory': str(folder)}}}
    identities = iter(['v1', 'v2' if changed_identity else 'v1'])
    monkeypatch.setattr(versions, 'read_identity', lambda *a, **k: next(identities))
    monkeypatch.setattr(files, 'read_json', lambda *a, **k: state)
    monkeypatch.setattr(context, 'current_config_session', lambda handler: {})
    monkeypatch.setattr(build, 'build_effective_run_config', lambda *a: {'mlip_finetune': {}})
    monkeypatch.setattr(provenance, 'model_with_training_provenance', lambda *a: {})
    monkeypatch.setattr(select, 'selected_records', lambda current, policy: ['all'])
    monkeypatch.setattr(select, 'split_independent_holdout', lambda *a: (['train'], ['test'], {'groups': ['g']}))
    monkeypatch.setattr(select, 'group_equivalent_training_records', lambda *a: (['train'], {'independent_group_count': 5}))
    monkeypatch.setattr(main, 'grouped_folds', lambda *a: ([], []))
    monkeypatch.setattr(regen, 'plan_finetune_regeneration', lambda *a: {'job_key': 'job', 'exists': True})
    def regenerate(current, config, plan):
        folder.rename(backup)
        folder.mkdir()
        (folder / 'new.txt').write_text('new')
        current['finetune_regeneration_history'] = [{'directory': str(folder), 'backup_directory': str(backup)}]
        return {'state': current, 'handoffs': []}
    monkeypatch.setattr(regen, 'regenerate_finetune_inputs', regenerate)
    saved = []
    def save(new_state, *args, **kwargs):
        if save_error:
            raise OSError('injected')
        saved.append(new_state)
    monkeypatch.setattr(lifecycle, '_save_runner_state', save)
    control = SimpleNamespace(state_path=str(state_path), chat_handler=SimpleNamespace(lock=threading.RLock(), workflow_kwargs={}))
    args = {'task_id': 'sample', 'input_structure_sha256': 'a' * 64, 'model_version': 'm0',
            'expected_job_key': 'job', 'independent_test_fraction': 0.1, 'seed': 2026,
            'user_instruction': 'approved'}
    return control, args, folder, backup, saved


def test_control_preserves_old_inputs_and_scopes_policy(tmp_path, monkeypatch):
    control, args, folder, backup, saved = setup_fakes(tmp_path, monkeypatch)
    result = regenerate_preflight(control, **args)
    assert result['status'] == 'prepared' and result['submitted'] is False
    assert (backup / 'old.txt').read_text() == 'old'
    assert (folder / 'new.txt').read_text() == 'new'
    assert 'training_preflight_policy' not in saved[0]


@pytest.mark.parametrize('changed', [False, True])
def test_save_failure_never_blindly_reverts_a_committed_state(tmp_path, monkeypatch, changed):
    control, args, folder, backup, saved = setup_fakes(tmp_path, monkeypatch, save_error=True, changed_identity=changed)
    with pytest.raises((OSError, RuntimeError)):
        regenerate_preflight(control, **args)
    if changed:
        assert (backup / 'old.txt').is_file() and (folder / 'new.txt').is_file()
    else:
        assert (folder / 'old.txt').is_file() and not backup.exists()
        assert list(tmp_path.glob('MLIP-finetune-round-0001.failed-save-*'))


def test_empty_user_authorization_rejected_before_any_mutation(tmp_path, monkeypatch):
    control, args, folder, backup, saved = setup_fakes(tmp_path, monkeypatch)
    args['user_instruction'] = ''
    with pytest.raises(ValueError, match='authorization'):
        regenerate_preflight(control, **args)
    assert (folder / 'old.txt').is_file() and not backup.exists()


def test_full_agent_regeneration_keeps_old_and_reserves_real_test(tmp_path, monkeypatch):
    import hashlib
    import json
    from ase.io import read
    from pymatgen.core import Structure
    from phase_agent.configuration.defaults.default_mace_committee_config import default_mace_committee_config
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune
    from tests.test_dft_comparison_csv import add_result
    import phase_agent.persistence.state.read_versions as versions
    import phase_agent.tools.step_runner.file_protocol as files
    import phase_agent.tools.workflows.lifecycle_support as lifecycle
    import phase_agent.configuration.runtime.build_effective_run_config as build
    import phase_agent.runtime.project_context as context

    state = {'active_model_version': 'm1', 'active_model': {'version': 'm1',
        'model_path': '/remote/base.model', 'training_provenance_complete': True,
        'training_data_ids': [], 'training_structure_hashes': []},
        'confirmed_config': {'mlip': {'version': 'm1', 'model_path': '/remote/base.model'}}}
    for i in range(20):
        add_result(state, task_id=f'T{i}', atoms=i + 2)
    target = next(t for t in state['tasks'] if t['task_id'] == 'T0')
    record = next(r for r in state['dft_training_records'] if r['task_id'] == 'T0')
    source = tmp_path / 'target.vasp'
    Structure.from_dict(record['structure']).to(filename=str(source), fmt='poscar')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    target.update(structure_path=str(source), input_structure_sha256=sha)
    state['tasks'].append({'task_id': 'refresh', 'stage': 'relax_and_feature',
        'status': 'completed', 'model_version': 'm1',
        'result_disposition': 'excluded_extrapolation_failure',
        'outputs': {'structure_checksum': sha}})
    settings = default_mace_committee_config()
    settings['labels']['include_stress'] = False
    config = {'mlip_finetune': settings, 'upload_batches_directory': str(tmp_path / 'submissions'),
        'mlip': {'version': 'm1', 'model_path': '/remote/base.model'},
        'python_environments': {'remote_mlip': 'mace'}}
    old = prepare_remote_finetune(state, config)
    assert old['status'] == 'awaiting_remote_training'
    state = old['state']
    key, job = next(iter(state['remote_finetune_jobs'].items()))
    original = Path(job['directory']) / 'inputs' / '_shared_data' / 'train.xyz'
    assert len(read(original, index=':')) == 19
    old_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    state_path = tmp_path / 'state.json'
    state_path.write_text('{}')
    monkeypatch.setattr(versions, 'read_identity', lambda *a, **k: 'version')
    monkeypatch.setattr(files, 'read_json', lambda *a, **k: state)
    monkeypatch.setattr(context, 'current_config_session', lambda handler: {})
    monkeypatch.setattr(build, 'build_effective_run_config', lambda *a: config)
    saved = []
    monkeypatch.setattr(lifecycle, '_save_runner_state', lambda new_state, *a, **k: saved.append(new_state))
    control = SimpleNamespace(state_path=str(state_path), chat_handler=SimpleNamespace(lock=threading.RLock(), workflow_kwargs={}))
    result = regenerate_preflight(control, task_id='T0', input_structure_sha256=sha,
        model_version='m1', expected_job_key=key, independent_test_fraction=0.1,
        seed=2026, user_instruction='user approved target and holdout')
    root = Path(result['training_directory']) / 'inputs'
    plan = json.loads((root / 'training_plan.json').read_text(encoding='utf8'))
    train = read(root / '_shared_data/train.xyz', index=':')
    test = read(root / '_shared_data/test.xyz', index=':')
    assert len(train) == 18 and len(test) == 2
    assert 'T0' in {a.info['task_id'] for a in train}
    hold = {a.info['data_id'] for a in test}
    assert hold.isdisjoint(a.info['data_id'] for a in train)
    assert set(plan['training_snapshot']['training_data_ids']).isdisjoint(hold)
    for i in range(1, 6):
        for split in ('train', 'valid'):
            cv = read(root / f'main_cv_{i}' / f'{split}.xyz', index=':')
            assert hold.isdisjoint(a.info['data_id'] for a in cv)
    old_backup = Path(result['backup_directory']) / 'inputs' / '_shared_data' / 'train.xyz'
    assert hashlib.sha256(old_backup.read_bytes()).hexdigest() == old_sha
    assert saved and 'training_preflight_policy' not in saved[0]
    assert result['submitted'] is False and result['activated'] is False


def test_mcp_exposes_preflight_regeneration_with_strict_arguments():
    import json
    from phase_agent.runtime.codex.mcp import CodexMcpServer
    seen = []
    class Control:
        def regenerate_training_preflight(self, **arguments):
            seen.append(arguments)
            return {'status': 'prepared', 'submitted': False}
    server = CodexMcpServer(codex=SimpleNamespace(control=Control()))
    listing = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
    assert 'phase_regenerate_training_preflight' in {x['name'] for x in listing['result']['tools']}
    args = {'task_id': 'T', 'input_structure_sha256': 'a' * 64, 'model_version': 'm',
            'expected_job_key': 'job', 'independent_test_fraction': 0.1, 'seed': 2026,
            'user_instruction': 'approved'}
    result = server.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
                            'params': {'name': 'phase_regenerate_training_preflight', 'arguments': args}})
    assert json.loads(result['result']['content'][0]['text'])['status'] == 'prepared'
    assert seen == [args]
    invalid = server.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                             'params': {'name': 'phase_regenerate_training_preflight', 'arguments': {**args, 'extra': 1}}})
    assert 'error' in invalid and len(seen) == 1

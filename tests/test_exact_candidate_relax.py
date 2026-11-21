import hashlib
from types import SimpleNamespace

import pytest

from phase_agent.tools.local.prepare_relax_upload_batches import _exact_candidate_frames
from phase_agent.tools.local.prepare_relax_upload_batches import prepare_relax_upload_batches


def make_case(tmp_path):
    source = tmp_path / 'frame.vasp'
    source.write_bytes(b'approved geometry')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate = {'candidate_id': 'c', 'structure_id': 's', 'structure_path': str(source),
                 'structure_sha256': digest, 'atom_count': 10}
    manager = SimpleNamespace(data={'structures': {'s': {'branch_id': 'b'}}, 'branches': {'b': {}}})
    state = {'active_model_version': 'model', 'qbc_candidates': [candidate]}
    action = {'target_ids': ['c'], 'parameters': {'candidate_frame_sha256': {'c': digest},
                                                  'model_version': 'model'}}
    config = {'budgets': {'stage_limits': {'relax_and_feature': {'task_cost': 2}}}}
    return action, state, manager, config, source


def test_exact_frame_uses_candidate_path_not_ledger_source(tmp_path):
    action, state, manager, config, source = make_case(tmp_path)
    manager.data['structures']['s']['source_path'] = 'different.vasp'
    planned = _exact_candidate_frames(action, state, manager, config, 'model')
    assert len(planned) == 1
    assert planned[0][3]['path'] == str(source)
    assert planned[0][3]['sha256'] == action['parameters']['candidate_frame_sha256']['c']


def test_changed_frame_and_model_are_rejected(tmp_path):
    action, state, manager, config, source = make_case(tmp_path)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum_mismatch'):
        _exact_candidate_frames(action, state, manager, config, 'model')
    with pytest.raises(ValueError, match='model_changed'):
        _exact_candidate_frames(action, state, manager, config, 'other')


def test_missing_or_duplicate_candidates_are_rejected(tmp_path):
    action, state, manager, config, _ = make_case(tmp_path)
    action['target_ids'].append('c')
    with pytest.raises(ValueError, match='identity_invalid'):
        _exact_candidate_frames(action, state, manager, config, 'model')
    action['target_ids'] = ['c']
    del manager.data['structures']['s']
    with pytest.raises(ValueError, match='structure_unregistered'):
        _exact_candidate_frames(action, state, manager, config, 'model')


def test_exact_preparation_packages_original_frame_and_phase_identification(tmp_path):
    from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
    from tests.test_debug_relax_inputs import write_structure
    action, state, manager, config, source = make_case(tmp_path)
    write_structure(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    state['qbc_candidates'][0]['structure_sha256'] = digest
    state['dedup_gate'] = {'status': 'ready'}
    state['tasks'] = []
    state['active_model'] = {'version':'model', 'model_path':'/remote/current-model'}
    class HistoryMustNotBeCopied(dict):
        def __deepcopy__(self, memo):
            raise AssertionError('unrelated history must not be copied')
    state['unrelated_history'] = HistoryMustNotBeCopied()
    action['parameters'].update(selection_scope='candidate_ids', candidate_frame_sha256={'c': digest})
    manager.boundary = {'P': ['O3']}
    manager.data['structures']['s'].update(composition={'Na': 1, 'Fe': 1, 'Mn': 1, 'O': 4})
    manager.data['branches']['b'].update(branch_id='b', P='O3')
    config.update(budgets=default_budget_rules(), mlip={'name':'model', 'model_path':'/remote/model'},
                  bohb={}, upload_batches_directory=str(tmp_path / 'upload'),
                  supercomputer={'worker': {'command': ['python3', '-m', 'worker', '--executor',
                     'phase_agent.science.mlip.slurm_executor:execute_mlip_task']}})
    result = prepare_relax_upload_batches(action=action, context={'effective_config': config,
        'manager': manager, 'event_state': state, 'config_version': 'config-test', 'phase_references': {}})
    assert result['status'] == 'prepared'
    assert result['task_count'] == 1 and result['submitted'] is False
    assert state['tasks'] == []
    assert 'budget_reservations' not in state
    import json
    folder = __import__('pathlib').Path(result['batches'][0]['directory'])
    task_file = next(folder.glob('*/task.json'))
    task = json.loads(task_file.read_text(encoding='utf8'))
    assert task['input_structure_sha256'] == digest
    assert hashlib.sha256((task_file.parent / 'initial.vasp').read_bytes()).hexdigest() == digest
    assert task['candidate_id'] == 'c'
    assert task['worker_job']['model_path'] == '/remote/current-model'
    assert result['state']['slurm_batches'][0]['phase_analysis']['script_path'].endswith('CPU.sh')
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.remote.integrity import file_checksum
    output_path = __import__('pathlib').Path(result['state']['tasks'][0]['result_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame = output_path.parent / 'initial_relaxed.vasp'
    final_frame.write_bytes(b'relaxed geometry')
    identity = {k: task[k] for k in ('task_id', 'task_key', 'batch_id', 'config_version',
                'model_version', 'protocol_version', 'input_file_version', 'task_checksum')}
    payload = {**identity, 'stage':'relax_and_feature', 'status':'completed', 'converged':True,
               'outputs': {'structure_path':final_frame.name, 'structure_checksum':file_checksum(final_frame),
                           'energy':-2.0, 'energy_unit':'eV', 'relax_stopped_normally':True}}
    output_path.write_text(json.dumps(payload), encoding='utf8')
    (output_path.parent / 'task.finished.json').write_text(json.dumps({**identity, 'status':'completed',
                'result_checksum':file_checksum(output_path)}), encoding='utf8')
    returned = RemoteBatchRunner(tmp_path / 'unused', worker_command=[]).collect_results_with_report(result['state'])
    assert returned['report']['recovered_count'] == 1, returned['report']
    assert returned['results'][0]['task_id'] == task['task_id']
    assert file_checksum(__import__('pathlib').Path(returned['results'][0]['outputs']['structure_path'])) == file_checksum(final_frame)


def test_recovery_gate_allows_only_exact_relax(monkeypatch):
    from phase_agent.policies import recovery_transition as policy
    monkeypatch.setattr(policy, '_post_recovery_base_tools', lambda state: {'select_dft_candidates'})
    state = {'active_model_version':'m', 'confirmed_config_version':'v',
             'exact_candidate_relax_authorization': {'status':'approved_for_preparation',
                'model_version':'m', 'config_version':'v', 'candidate_frame_sha256':{'c':'sha'}}}
    assert policy.post_recovery_tools({}) == {'select_dft_candidates'}
    assert policy.post_recovery_tools(state) == {'select_dft_candidates', 'prepare_local_batch_files'}
    action = {'tool': 'prepare_local_batch_files', 'operation': 'prepare_mlip_relax',
              'target_ids':['c'], 'parameters': {'mode': 'relax_inputs', 'selection_scope': 'candidate_ids',
                                                 'candidate_frame_sha256':{'c':'sha'}}}
    assert policy.exact_recovery_scope_errors(action, state) == []
    action['parameters']['selection_scope'] = 'target_ids'
    assert policy.exact_recovery_scope_errors(action, state) == ['post_recovery_only_exact_candidate_relax_allowed']


def test_authorization_is_version_bound_and_idempotent(tmp_path):
    import json
    import threading
    from phase_agent.runtime.exact_candidate_relax_authorization import authorize
    action, state, manager, config, source = make_case(tmp_path)
    state.update(confirmed_config_version='v', phase_diagrams={'mlip':{'version':'h'}})
    path = tmp_path / 'state.json'
    path.write_text(json.dumps(state), encoding='utf8')
    control = SimpleNamespace(chat_handler=SimpleNamespace(lock=threading.RLock()),
                              state_path=path, _codex_context_cache=None)
    args = dict(candidate_frame_sha256=action['parameters']['candidate_frame_sha256'],
                expected_model='model', expected_config='v', expected_hull='h',
                user_instruction='Prepare these frames; no submission')
    result = authorize(control, **args)
    assert result['status'] == 'authorized' and result['submit_authorized'] is False
    assert authorize(control, **args)['status'] == 'already_authorized'
    source.write_bytes(b'wrong geometry')
    with pytest.raises(ValueError, match='checksum_mismatch'):
        authorize(control, **args)
    assert json.loads(path.read_text(encoding='utf8'))['exact_candidate_relax_authorization']['status'] == 'approved_for_preparation'


def test_budget_review_does_not_treat_exact_relax_as_new_route():
    from phase_agent.agents.strategic.round_budget_review import round_budget_review_errors
    context={'round_budget_evidence':{'required':True}}
    action={'tool':'prepare_local_batch_files','operation':'prepare_mlip_relax',
            'parameters':{'mode':'relax_inputs','selection_scope':'candidate_ids'}}
    assert round_budget_review_errors(action,context) == []
    action['parameters']['selection_scope']='target_ids'
    assert round_budget_review_errors(action,context)

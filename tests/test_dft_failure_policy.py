from phase_agent.evidence.dft_failure_policy import completed_failure_batches, diagnose_failure_batch


def make_state(failures, successes, pending=0):
    statuses = ['failed'] * failures + ['completed'] * successes + ['pending'] * pending
    tasks = [{'task_id': str(i), 'stage': 'dft_single_point', 'status': status,
              'model_version': 'm', 'upload_operation_id': 'batch',
              'search_group_index': i % 2, 'error': 'SCF' if status == 'failed' else None}
             for i, status in enumerate(statuses)]
    return {'tasks': tasks}, [{'task_id': t['task_id'], 'status': 'settled'}
                              for t in tasks if t['status'] != 'pending']


def test_thresholds():
    assert completed_failure_batches(*make_state(1, 4))[0]['failed_count'] == 1  # exactly 20%
    assert not completed_failure_batches(*make_state(1, 5))
    assert completed_failure_batches(*make_state(1, 3))[0]['failed_count'] == 1
    assert not completed_failure_batches(*make_state(10, 90))
    assert not completed_failure_batches(*make_state(11, 100))  # no absolute-count shortcut
    assert completed_failure_batches(*make_state(20, 80))[0]['failed_count'] == 20
    assert completed_failure_batches(*make_state(1, 3, pending=1))[0]['threshold_basis'] == 'guaranteed_lower_bound'
    assert not completed_failure_batches(*make_state(1, 3, pending=2))
    state, reconciled = make_state(1, 3, pending=1)
    state['tasks'][-1]['status'] = 'submission_unknown'
    assert completed_failure_batches(state, reconciled)[0]['eligible_count'] == 5
    assert completed_failure_batches(*make_state(1, 0))[0]['returned_count'] == 1


def test_pending_are_not_failures_and_guaranteed_threshold_stops_early():
    state, reconciled = make_state(94, 2, pending=4)
    batch = completed_failure_batches(state, reconciled)[0]
    assert (batch['failed_count'], batch['returned_count'], batch['eligible_count'], batch['pending_count']) == (94, 96, 100, 4)
    assert batch['threshold_basis'] == 'guaranteed_lower_bound'
    state['active_model_version'] = 'other'
    assert not completed_failure_batches(state, reconciled)


def test_cancelled_and_retry_scope_do_not_count_as_failed():
    state, reconciled = make_state(1, 5, pending=1)
    state['tasks'][-1]['status'] = 'cancelled'
    assert not completed_failure_batches(state, reconciled)
    state['tasks'][0]['retry_of'] = 'old'
    batches = completed_failure_batches(state, reconciled)
    assert len(batches) == 1 and batches[0]['scope']['retry_index'] == 1


def test_simulation_diagnosis_and_unavailable():
    batch = completed_failure_batches(*make_state(1, 3))[0]
    class Client:
        supports_simulation = True
        def __call__(self, request):
            assert request['allowed_tools'] == []
            return {'assessment': 'SCF failures', 'evidence': ['execution.operation_outcome'],
                    'recommendation': 'request_human_review', 'limitations': 'logs needed'}
    assert diagnose_failure_batch(batch, Client())['diagnosis']['assessment'] == 'SCF failures'
    assert diagnose_failure_batch(batch, None)['diagnosis_unavailable']


def test_workflow_routes_mass_failure_to_simulation():
    from unittest.mock import patch
    from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    state, reconciled = make_state(1, 3)
    frame = {'recovery_question': None, 'execution_mode': 'interactive',
             'feedback': {'state': state}, 'state_path': None, 'effective_config': {},
             'recovered_count': 4, 'collection_report': None, 'snapshot': {},
             'pre_reconciled': {'reconciled': reconciled}, 'manual_wait': None,
             'rebuilding': False, 'agent_client': None}
    with patch('phase_agent.tools.workflows.reference_dft_startup.startup_gate', return_value=None), \
         patch('phase_agent.tools.workflows.lifecycle_recovery._save_runner_state'):
        result = _workflow_wait_gate(frame)
    assert result['status'] == 'dft_failure_review'
    assert result['steps_executed'] == 0 and result['submitted'] is False
    assert result['dft_failure_reviews'][0]['diagnosis_unavailable']
    assert 'Simulation 诊断尚不可用' in format_workflow_reply(result, None)


def test_workflow_shows_simulation_diagnosis():
    from unittest.mock import patch
    from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    state, reconciled = make_state(2, 3)
    class Client:
        supports_simulation = True
        def __call__(self, request):
            return {'assessment': '电子自洽失败', 'evidence': ['execution.operation_outcome'],
                    'recommendation': 'request_human_review', 'limitations': '需要更多日志'}
    frame = {'recovery_question': None, 'execution_mode': 'interactive',
             'feedback': {'state': state}, 'state_path': None, 'effective_config': {},
             'recovered_count': 5, 'collection_report': None, 'snapshot': {},
             'pre_reconciled': {'reconciled': reconciled}, 'manual_wait': None,
             'rebuilding': False, 'agent_client': Client()}
    with patch('phase_agent.tools.workflows.reference_dft_startup.startup_gate', return_value=None), \
         patch('phase_agent.tools.workflows.lifecycle_recovery._save_runner_state'):
        result = _workflow_wait_gate(frame)
    assert '电子自洽失败' in format_workflow_reply(result, None)
    assert state['dft_failure_reviews'][0]['diagnosis']['limitations'] == '需要更多日志'


def test_guard_persists_without_diagnosis_and_blocks_direct_bypass():
    from unittest.mock import patch
    from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate
    from phase_agent.policies.workflow_policy import action_errors
    state, reconciled = make_state(1, 4)
    state['active_model_version'] = 'm'
    frame = {'recovery_question': None, 'execution_mode':'interactive', 'feedback':{'state':state},
             'state_path':None, 'effective_config':{}, 'recovered_count':5, 'collection_report':None,
             'snapshot':{}, 'pre_reconciled':{'reconciled':reconciled}, 'manual_wait':None,
             'rebuilding':False, 'agent_client':None}
    action = {'tool':'prepare_local_batch_files','operation':'prepare_perturbation_recovery',
              'parameters':{'mode':'perturbation_recovery_inputs'}}
    assert 'dft_failure_analysis_required' in action_errors(action, state, {})
    with patch('phase_agent.tools.workflows.reference_dft_startup.startup_gate', return_value=None), \
         patch('phase_agent.tools.workflows.lifecycle_recovery._save_runner_state'):
        first = _workflow_wait_gate(frame)
        assert first['status'] == 'dft_failure_review' and state['dft_failure_guard']['status'] == 'awaiting_analysis'
        assert len(state['dft_failure_reviews']) == 1
        frame['pre_reconciled']['reconciled'] = []
        second = _workflow_wait_gate(frame)
        assert second['status'] == 'dft_failure_review' and len(state['dft_failure_reviews']) == 1
        class Client:
            supports_simulation = True
            def __call__(self, request):
                return {'assessment':'input failure', 'evidence':['execution.operation_outcome'],
                        'recommendation':'request_human_review', 'limitations':'need logs'}
        frame['agent_client'] = Client()
        third = _workflow_wait_gate(frame)
        assert third['status'] == 'dft_failure_review' and state['dft_failure_guard']['status'] == 'analyzed'
        assert 'dft_failure_analysis_required' not in action_errors(action, state, {})
        assert len(state['dft_failure_reviews']) == 2


def test_native_agent_can_register_only_current_bound_failure_analysis(tmp_path):
    import threading
    from types import SimpleNamespace
    import pytest
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json
    from phase_agent.evidence.dft_failure_policy import unreviewed_failure_batches
    state, _ = make_state(2, 8)
    state['active_model_version'] = 'm'
    state_path = tmp_path / 'state.json'
    write_json(state_path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=state_path, lock=threading.RLock()))
    batch = completed_failure_batches(state, [])[0]
    analysis = {'assessment':'Input errors require investigation', 'evidence':['execution.operation_outcome'],
                'recommendation':'return_to_strategic', 'limitations':'Failure logs not yet reviewed'}
    with pytest.raises(ValueError, match='stale_or_unknown'):
        control.register_dft_failure_analysis('wrong-digest', analysis)
    assert 'dft_failure_reviews' not in read_json(state_path, {})
    result = control.register_dft_failure_analysis(batch['failure_evidence_digest'], analysis)
    assert result['status'] == 'recorded' and result['guard_status'] == 'analyzed' and not result['submitted']
    saved = read_json(state_path, {})
    assert not unreviewed_failure_batches(saved)
    assert len(saved['dft_failure_reviews']) == 1
    assert control.register_dft_failure_analysis(batch['failure_evidence_digest'], analysis)['status'] == 'reused'
    assert len(read_json(state_path, {})['dft_failure_reviews']) == 1
    changed = read_json(state_path, {})
    changed['tasks'][2]['status'] = 'failed'
    write_json(state_path, changed)
    with pytest.raises(ValueError, match='stale_or_unknown'):
        control.register_dft_failure_analysis(batch['failure_evidence_digest'], analysis)
    assert unreviewed_failure_batches(read_json(state_path, {}))


def test_missing_model_identity_does_not_hide_failed_batch():
    state, reconciled = make_state(1, 4)
    state['active_model_version'] = 'current'
    for task in state['tasks']:
        task['model_version'] = None
    result = completed_failure_batches(state, reconciled)
    assert len(result) == 1 and result[0]['scope']['model_version'] is None


def test_read_only_failure_status_tracks_live_evidence():
    from copy import deepcopy
    from phase_agent.evidence.dft_failure_policy import failure_guard_facts
    state, _ = make_state(1, 4)
    before = deepcopy(state)
    facts = failure_guard_facts(state)
    assert facts['status'] == 'awaiting_analysis' and state == before
    batch = facts['batches'][0]
    state['dft_failure_reviews'] = [{'batch':batch,'diagnosis':{
        'assessment':'inspect errors','evidence':['execution.operation_outcome'],
        'recommendation':'return_to_strategic','limitations':'unknown cause'}}]
    assert failure_guard_facts(state)['status'] == 'analyzed'
    state['tasks'][1]['status'] = 'failed'
    assert failure_guard_facts(state)['status'] == 'awaiting_analysis'


def test_unbound_or_invalid_diagnosis_does_not_clear_guard():
    from phase_agent.evidence.dft_failure_policy import failure_guard_facts
    state, reconciled = make_state(1, 4)
    batch = completed_failure_batches(state, reconciled)[0]
    class Client:
        supports_simulation = True
        def __call__(self, request):
            return {'assessment':'unknown', 'evidence':['execution.status'],
                    'recommendation':'return_to_strategic', 'limitations':'no batch evidence'}
    response = diagnose_failure_batch(batch, Client())
    assert 'diagnosis' not in response and response['diagnosis_unavailable'] == 'ValueError'
    state['dft_failure_reviews'] = [{'batch':batch, 'diagnosis':{
        'assessment':'unknown','evidence':['execution.status'],
        'recommendation':'return_to_strategic','limitations':'no batch evidence'}}]
    assert failure_guard_facts(state)['status'] == 'awaiting_analysis'


def test_agent_records_only_exact_pending_dft_scope_without_remote_cancel(tmp_path):
    import threading
    from types import SimpleNamespace
    import pytest
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json
    state, _ = make_state(2, 2, pending=2)
    for task in state['tasks']:
        task['search_group_index'] = 0
    path = tmp_path / 'state.json'
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path, lock=threading.RLock()))
    with pytest.raises(ValueError, match='scope_changed'):
        control.register_dft_wait_waiver(['4'], '不再等待')
    assert read_json(path, {})['tasks'][4]['status'] == 'pending'
    result = control.register_dft_wait_waiver(['4', '5'], '不再等待这两项')
    saved = read_json(path, {})
    assert result['remote_cancel_sent'] is False and result['submitted'] is False
    assert all(t['status'] == 'cancelled' and t['recovery_wait_waived'] for t in saved['tasks'][4:])
    assert all(t['status'] == 'completed' for t in saved['tasks'][2:4])
    assert len(saved['dft_recovery_cancellations']) == 2

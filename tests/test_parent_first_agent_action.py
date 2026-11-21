from types import SimpleNamespace

from phase_agent.tools.local.prepare_local_batch_files import prepare_local_batch_files


def test_approved_action_registers_prepared_handoff_without_submitting(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import parent_first_submission

    captured = {}
    def prepare(state, ledger, parameters, target, **kwargs):
        captured.update(parameters=parameters, target=target, **kwargs)
        return {'status':'prepared', 'submitted':False, 'parent_count':2, 'reference_count':3,
                'target':100, 'task_id':'parent-first-id', 'input_dir':str(target/'input'),
                'handoff':{'upload_archive':'inputs.zip'}}
    monkeypatch.setattr(parent_first_submission, 'prepare_parent_first_submission', prepare)
    state = {'confirmed_config_version':'config-1', 'active_model_version':'model-1',
             'upload_layout':{'model_rounds':{'model-1':2}}}
    params = {'mode':'parent_first_recovery_inputs', 'confirmed':True, 'scope_id':'scope',
              'round_evidence_digest':'digest', 'dft_parameters':{}, 'feature_model_path':'/remote/m3gnet',
              'feature_model_sha256':'a'*64, 'direct_target':100, 'parent_record_ids':['p1','p2']}
    action = {'operation':'prepare_parent_first_recovery', 'parameters':params}
    context = {'event_state':state, 'effective_config':{'upload_batches_directory':str(tmp_path),'dft':{'parameters':{}}},
               'manager':SimpleNamespace(data={}), 'approval_record_id':'approved-1', 'config_version':'config-1'}
    result = prepare_local_batch_files(action=action, context=context)
    batch = result['parent_first_recovery_batch']
    assert result['status']=='prepared' and result['submitted'] is False
    assert batch['approval_record_id']=='approved-1' and batch['target']==100
    assert captured['dft_settings']['model_version']=='model-1'
    assert captured['target'].is_relative_to(tmp_path/'epoch1')
    assert 'parent_first_recovery_batches' not in state
    context['event_state'] = result['state']
    assert prepare_local_batch_files(action=action, context=context)['reason'] == 'parent_first_existing_batch_requires_reconciliation'
    context['event_state'] = state
    context['effective_config']['dft']['parameters'] = {'incar_settings': {'static': {'ENCUT': 300}}}
    assert prepare_local_batch_files(action=action, context=context)['reason'] == 'parent_first_approval_or_config_invalid'


def test_reliability_route_accepts_new_action_and_waits_on_existing_batch(monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    from phase_agent.tools.dispatch.scientific_operations import bind_operation
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
    monkeypatch.setattr(extrapolation_round,'required_direct_round',lambda state:{
        'scope_id':'epoch_refresh','evidence_digest':'bound','failure_fraction':0.1})
    action = bind_operation({'operation':'prepare_parent_first_recovery',
        'parameters':{'scope_id':'epoch_refresh','round_evidence_digest':'bound'},
        'task_key':'p','budget':0,'reason':'coverage'})
    assert action['tool']=='prepare_local_batch_files' and action['parameters']['mode']=='parent_first_recovery_inputs'
    args = dict(mode='interactive',config={'agent':{'allowed_tools':['prepare_local_batch_files']}},
        context={},decision_state={},registry={'prepare_local_batch_files':{'handler':lambda **kwargs: None}},
        agent_client=lambda payload:None,invocation_id='i',propose=lambda *a,**k:action,revise=None,
        prepare_debug=None,is_verified_second=None,mc_block=None,model_failed=None,record_id_factory=lambda *a:'record')
    proposed = select_initial_proposal(current={'active_model_version':'m','upload_layout':{'model_rounds':{'m':1}}}, **args)
    assert proposed['_proposal_selected'] and proposed['proposal']['raw_action']['operation']=='prepare_parent_first_recovery'
    state = {'parent_first_recovery_batches':{'b':{'status':'waiting_dft_results','scope_id':'epoch_refresh',
              'round_evidence_digest':'bound'}}}
    result = select_initial_proposal(current=state, **args)
    assert result['status']=='waiting_results' and result['submitted'] is False


def test_initial_direct_cold_start_defaults_to_epoch0_when_upload_layout_empty(
    tmp_path, monkeypatch
):
    from phase_agent.tools.workflows import parent_first_submission

    captured = {}

    def prepare(state, ledger, parameters, target, **kwargs):
        captured.update(parameters=parameters, target=target, **kwargs)
        return {
            'status': 'prepared',
            'submitted': False,
            'parent_count': 2,
            'reference_count': 0,
            'target': 100,
            'task_id': 'parent-first-cold',
            'input_dir': str(target / 'input'),
            'handoff': {'upload_archive': 'inputs.zip'},
        }

    monkeypatch.setattr(parent_first_submission, 'prepare_parent_first_submission', prepare)
    state = {'confirmed_config_version': 'config-1', 'active_model_version': 'model-1'}
    params = {
        'mode': 'parent_first_recovery_inputs',
        'confirmed': True,
        'scope_id': 'initial_direct',
        'round_evidence_digest': 'initial_direct_v1',
        'dft_parameters': {},
        'direct_target': 100,
        'initial_parents': ['s1', 's2'],
    }
    action = {'operation': 'prepare_parent_first_recovery', 'parameters': params}
    context = {
        'event_state': state,
        'effective_config': {'upload_batches_directory': str(tmp_path), 'dft': {'parameters': {}}},
        'manager': SimpleNamespace(data={}),
        'approval_record_id': 'approved-cold',
        'config_version': 'config-1',
    }
    result = prepare_local_batch_files(action=action, context=context)
    assert result['status'] == 'prepared' and result['submitted'] is False
    assert captured['target'].is_relative_to(tmp_path / 'epoch0')

import json
from pathlib import Path
from tests.test_training_handoff import setup, config
from execution_layer.local.training_handoff import advance_training_handoffs
from orchestration.training_handoff_graph import build_training_handoff_graph


def test_sqlite_restart_resumes_native_interrupt_with_latest_state(tmp_path):
    state,results,inputs=setup(tmp_path)
    models=json.loads((results/'models.json').read_text())
    models[0]['sha256']=''
    (results/'models.json').write_text(json.dumps(models))
    path=tmp_path/'workflow_state/state.json'
    first,waits=advance_training_handoffs(state,{},state_path=path)
    assert waits[0]['stage']=='awaiting_manifest'
    db=path.parent/'langgraph_training.sqlite'
    assert db.is_file()
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(str(db)) as saver:
        entries=list(saver.list(None))
        thread={"configurable":{"thread_id":entries[0].config["configurable"]["thread_id"]}}
        graph=build_training_handoff_graph(saver)
        snapshot=graph.get_state(thread)
        assert snapshot.next==('wait_for_manifest_return',)
        assert snapshot.tasks[0].interrupts
    mtime=(inputs/'GPU_manifest.sh').stat().st_mtime_ns
    again,_=advance_training_handoffs(first,{},state_path=path)
    assert again==first and (inputs/'GPU_manifest.sh').stat().st_mtime_ns==mtime
    models[0]['sha256']='a'*64
    (results/'models.json').write_text(json.dumps(models))
    again['user_note']='canonical state added after checkpoint'
    recovered,waits=advance_training_handoffs(again,{},state_path=path)
    assert waits[0]['stage']=='validation_prerequisites_required'
    assert recovered['user_note']=='canonical state added after checkpoint'
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph=build_training_handoff_graph(saver)
        snapshot=graph.get_state(thread)
        assert snapshot.next==('wait_for_configuration_or_repair',)
        assert 'check_validation_prerequisites' in snapshot.values['node_trace']


def test_native_graph_validation_wait_to_approval_to_activation_exit(tmp_path):
    state,results,inputs=setup(tmp_path)
    settings=config(tmp_path);path=tmp_path/'state.json'
    current,waits=advance_training_handoffs(state,settings,state_path=path)
    plan=json.loads((inputs/'validation_request.json').read_text())
    metrics={'energy_mae':.003,'force_rmse':.06,'critical_failure_fraction':0,'near_hull_ranking_reversals':0}
    report={'request_id':plan['request_id'],'data_sha256':plan['data_sha256'],'status':'completed',
        'structures':2,'metrics':{'old_model':metrics,'new_model':metrics}}
    (results/('validation-'+plan['request_id']+'.json')).write_text(json.dumps(report))
    current,waits=advance_training_handoffs(current,settings,state_path=path)
    assert waits[0]['stage']=='awaiting_activation_approval'
    assert current['active_model_version']=='base'
    from execution_layer.local.review_candidate_command import review_candidate_command
    version=waits[0]['candidate_model_version']
    reviewed=review_candidate_command('激活候选 '+version+' 原因：独立验证通过',current)
    assert reviewed['state']['active_model_version']==version
    current,waits=advance_training_handoffs(reviewed['state'],settings,state_path=path)
    assert waits==[]
    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(str(tmp_path/'langgraph_training.sqlite')) as saver:
        thread={"configurable":{"thread_id":list(saver.list(None))[0].config["configurable"]["thread_id"]}}
        assert build_training_handoff_graph(saver).get_state(thread).next==()


def test_named_graph_contains_real_routes():
    graph=build_training_handoff_graph()
    nodes=graph.get_graph().nodes
    assert {'collect_training_results','check_training_manifest','prepare_manifest_job',
        'check_validation_prerequisites','prepare_validation_job','validate_and_register_candidate',
        'wait_for_manifest_return','wait_for_validation_return','wait_for_activation_decision'} <= set(nodes)

def test_training_child_is_discoverable_from_studio():
    from orchestration.studio_chat_graph import build_studio_graph
    children=dict(build_studio_graph(lambda *_:'unused').get_subgraphs(recurse=True))
    assert 'confirmed_local_chat|training_lifecycle' in children
    assert 'wait_for_activation_decision' in children['confirmed_local_chat|training_lifecycle'].nodes

def test_unfinished_node_recovery_rechecks_canonical_inputs(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    import hashlib
    state,results,inputs=setup(tmp_path)
    models=json.loads((results/'models.json').read_text())
    models[0]['sha256']=''
    (results/'models.json').write_text(json.dumps(models))
    path=tmp_path/'state.json'; db=tmp_path/'langgraph_training.sqlite'
    identity=json.dumps({'state_path':str(path.resolve()),'job_key':'j',
        'directory':state['remote_finetune_jobs']['j']['directory']},sort_keys=True)
    invocation={'configurable':{'thread_id':hashlib.sha256(identity.encode()).hexdigest()}}
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph=build_training_handoff_graph(saver)
        # Emulate an exited process after the preceding node checkpoint, before
        # an unstarted metadata side effect. The old checkpoint inspection is stale.
        graph.update_state(invocation,{'business':state,'config':{},'job_key':'j',
            'inspection':{'status':'metadata_required'},'route':'manifest','handoff':{}},
            as_node='check_training_manifest')
        assert graph.get_state(invocation).next==('prepare_manifest_job',)
    models[0]['sha256']='a'*64
    (results/'models.json').write_text(json.dumps(models))
    current,waits=advance_training_handoffs(state,{},state_path=path)
    assert waits[0]['stage']=='validation_prerequisites_required'
    assert not (inputs/'GPU_manifest.sh').exists()

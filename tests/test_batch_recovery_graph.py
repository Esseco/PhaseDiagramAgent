import json
from orchestration.batch_recovery_graph import recover_batch_stages
from orchestration.node_trace import append_node_trace


def test_partial_restart_and_failure_reports(tmp_path):
    state={'tasks':[{'task_id':'r1','stage':'relax_and_feature','status':'completed'},
        {'task_id':'r2','stage':'relax_and_feature','status':'submitted'},
        {'task_id':'m1','stage':'deep_search','status':'failed'},
        {'task_id':'d1','stage':'dft_single_point','status':'unknown'}]}
    path=tmp_path/'state.json'
    report=recover_batch_stages(state,path)
    assert report['relax_and_feature']['status']=='partial'
    assert report['deep_search']['failed_ids']==['m1']
    assert report['dft_single_point']['waiting_ids']==['d1']
    assert recover_batch_stages(state,path)==report
    state['tasks'][1]['status']='completed'
    state['tasks'][3]['recovery_wait_waived']=True
    report=recover_batch_stages(state,path)
    assert report['relax_and_feature']['completed_ids']==['r1','r2']
    assert report['dft_single_point']['waived_ids']==['d1']
    assert report['deep_search']['status']=='failures_require_review'
    from langgraph.checkpoint.sqlite import SqliteSaver
    from orchestration.batch_recovery_graph import build_batch_recovery_graph
    with SqliteSaver.from_conn_string(str(tmp_path/'langgraph_batches.sqlite')) as saver:
        thread=list(saver.list(None))[0].config['configurable']['thread_id']
        assert build_batch_recovery_graph(saver).get_state({'configurable':{'thread_id':thread}}).next==()


def test_trace_contains_no_prompt_or_raw_state(tmp_path):
    append_node_trace({'state_path':str(tmp_path/'state.json'),'prompt':'secret',
        'manual_wait':{'waiting_by_stage':{'deep_search':2}},'recovered_count':1},'collect',.2)
    row=json.loads((tmp_path/'node_trace.jsonl').read_text())
    assert row['elapsed_seconds']==.2 and row['llm_usage'] is None
    assert 'secret' not in json.dumps(row)


def test_batch_child_is_visible():
    from orchestration.studio_chat_graph import build_studio_graph
    assert 'confirmed_local_chat|batch_recovery' in dict(build_studio_graph(lambda *_:'unused').get_subgraphs(recurse=True))

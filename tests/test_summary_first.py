from copy import deepcopy
import json
import pytest
from phase_agent.decisions.agent.summary_request import request_summary_first,summary_payload
from phase_agent.runtime.token_usage import begin_api_call,finish_api_call,usage_summary,cached_request
from phase_agent.runtime.turn_process import _current,record_turn


def payload():
    return {'mode':'autonomous_search','state':{'snapshot_id':'s1','current_project':{'configuration':{'limits':5}},'mlip_status':{'active_version':'m1'},'current_convex_hull':{'status':'unknown'},'available_budget':12,'long_term_memory':{'human_system_knowledge':['do not add relax']},'available_branches':[{'candidate_id':str(i),'energy':i,'model_version':'m1'} for i in range(100)]},'decision_context':{'qbc_candidates':[{'candidate_id':str(i),'ehull':i/10} for i in range(200)],'retryable_tasks':[{'task_id':'failed'}]},'allowed_tools':['select_dft_candidates'],'current_proposal':{'plan_id':'p','parameters':{'limit':1}}}


def test_summary_is_small_and_full_candidates_are_only_read_when_requested():
    original=payload();before=deepcopy(original);requests=[]
    def model(p):
        requests.append(deepcopy(p))
        if len(requests)==1:return {'kind':'inspect_evidence','topics':['dft'],'_llm_usage':{'calls':1,'input_tokens':10,'output_tokens':2}}
        return {'tool':'select_dft_candidates','target_ids':['199'],'_llm_usage':{'calls':1,'input_tokens':20,'output_tokens':3}}
    result=request_summary_first(model,original)
    assert 'qbc_candidates' not in requests[0]['decision_context']
    assert 'available_branches' not in requests[0]['state']
    assert len(requests[1]['decision_context']['qbc_candidates'])==200
    assert 'available_branches' not in requests[1]['decision_context']
    assert requests[0]['state']['human_constraints']==['do not add relax']
    assert result['_llm_usage']['calls']==3 and result['_llm_usage']['input_tokens']==50
    assert original==before


def test_selection_without_loaded_pool_must_load_complete_evidence():
    requests=[]
    def model(p):requests.append(deepcopy(p));return {'tool':'select_dft_candidates','target_ids':['199']}
    request_summary_first(model,payload())
    assert len(requests)==2 and len(requests[1]['decision_context']['qbc_candidates'])==200


def test_repeated_reads_stop_without_replay_or_write():
    calls=[]
    def model(p):calls.append(p);return {'kind':'inspect_evidence','topics':['dft']}
    with pytest.raises(ValueError,match='上限|重复'):request_summary_first(model,payload())
    assert len(calls)==2


def test_changed_or_unrelated_requests_are_not_cached():
    record={};token=_current.set(record);calls=[]
    def model(p):calls.append(deepcopy(p));return {'answer':'saved','_llm_usage':{'calls':1,'input_tokens':5}}
    try:
        cached_request({'v':1},model)
        assert '_llm_usage' not in cached_request({'v':1},model)
        cached_request({'v':2},model)
        assert len(calls)==2 and record['reused_model_requests']==1
    finally:_current.reset(token)


def test_api_usage_tracks_retry_and_unknown_calls():
    record={};token=_current.set(record)
    try:
        finish_api_call(begin_api_call('m','search',0,100),{'prompt_tokens':12,'completion_tokens':5})
        begin_api_call('m','search',1,100)
        result=usage_summary(record)
        assert result['api_calls']==2 and result['retries']==1
        assert result['total_tokens']==17 and result['unknown_usage_calls']==1 and not result['complete']
    finally:_current.reset(token)


def test_diagnostics_persist_usage_without_cached_payloads(tmp_path):
    config=tmp_path/'agent_runtime.json';config.write_text('{}')
    with record_turn(config,'question','t'):
        cached_request({'credential':'never save this'},lambda p:{'answer':'x'})
        finish_api_call(begin_api_call('m','search',0,5),{'prompt_tokens':1,'completion_tokens':2})
    record=json.loads(next((tmp_path/'workflow_state/dialogue_process').glob('*.json')).read_text(encoding='utf-8'))
    assert record['token_usage']['total_tokens']==3
    assert '_model_request_cache' not in record and 'never save this' not in json.dumps(record)


def test_non_scientific_modes_remain_unchanged():
    request={'mode':'configuration_dialogue','state':{'x':1}}
    seen=[]
    assert request_summary_first(lambda p:seen.append(p) or {'kind':'answer'},request)=={'kind':'answer'}
    assert seen==[request]


def test_fact_reuse_adapter_observes_original_payload():
    original=payload();seen=[]
    def reuse(p):seen.append(p);return {'tool':'check_convergence'}
    reuse._validates_original_facts=True
    request_summary_first(reuse,original)
    assert seen==[original]


def test_provider_callback_records_internal_calls():
    from phase_agent.runtime.token_usage import provider_usage_callback
    from types import SimpleNamespace
    record={};token=_current.set(record)
    try:
        callback=provider_usage_callback('search')
        callback.on_llm_start({},['hello'],run_id='a')
        callback.on_llm_end(SimpleNamespace(llm_output={'token_usage':{'prompt_tokens':7,'completion_tokens':3}},generations=[]),run_id='a')
        callback.on_llm_start({},['failed'],run_id='b')
        result=usage_summary(record)
        assert result['api_calls']==2 and result['total_tokens']==10 and result['unknown_usage_calls']==1
    finally:_current.reset(token)


def test_evidence_request_cannot_smuggle_execution():
    with pytest.raises(ValueError,match='只读契约'):
        request_summary_first(lambda p:{'kind':'inspect_evidence','topics':['dft'],'tool':'prepare_local_batch_files'},payload())


def test_failure_is_not_cached():
    record={};token=_current.set(record);calls=[]
    def failed(p):calls.append(p);raise TimeoutError('offline')
    try:
        for _ in range(2):
            with pytest.raises(TimeoutError):cached_request({'same':1},failed)
        assert len(calls)==2
    finally:_current.reset(token)


def test_hidden_metric_change_invalidates_summary_cache_identity():
    original=payload();changed=deepcopy(original)
    changed['decision_context']['qbc_candidates'][0]['ehull']=99
    first,_=summary_payload(original);second,_=summary_payload(changed)
    assert first['state']==second['state']
    assert first['evidence_revision']!=second['evidence_revision']

from copy import deepcopy
import pytest
from phase_agent.decisions.agent.summary_request import request_summary_first, summary_payload, expand_topics, TOPICS, required_topics


def facts():
    return {'mode':'autonomous_search','allowed_tools':['select_dft_candidates','update_mlip'],
            'state':{'available_budget':12,'current_project':{'project':'p','configuration':{'limit':5}},'mlip_status':{'active_version':'m1'},
                     'long_term_memory':{'human_system_knowledge':['no new relax']},
                     'available_branches':[{'candidate_id':str(i),'model_version':'m1'} for i in range(150)]},
            'decision_context':{'qbc_candidates':[{'candidate_id':str(i),'ehull':i/100} for i in range(250)],
                                'new_model_generation':{'required':True,'status':'missing'},'task_round_evidence':{'active_ids':['old']},
                                'current_phase_diagram':{'coverage':'incomplete'},'training_validation_planning':{'independent_test':'failed'}}}


def test_critical_action_reads_all_prerequisites_together():
    payload=facts();before=deepcopy(payload);requests=[]
    def model(p):requests.append(deepcopy(p));return {'tool':'select_dft_candidates','target_ids':['249']}
    result=request_summary_first(model,payload)
    assert len(requests)==2
    assert set(requests[-1]['decision_context']['loaded_topics'])==required_topics(result)
    assert len(requests[-1]['decision_context']['qbc_candidates'])==250
    assert requests[-1]['decision_context']['new_model_generation']['required']
    assert requests[-1]['state']['current_project']['configuration']=={'limit':5}
    assert payload==before


def test_complex_analysis_can_take_more_than_two_reads_then_full_fallback():
    requests=[];topics=['mc','dft','tasks','model','budget']
    def model(p):
        requests.append(deepcopy(p))
        return {'kind':'inspect_evidence','topics':[topics[len(requests)-1]]} if len(requests)<=5 else {'tool':'update_mlip','parameters':{'action':'ACTIVATE_CANDIDATE_MODEL'}}
    request_summary_first(model,facts())
    assert len(requests)==6
    assert set(requests[-1]['decision_context']['loaded_topics'])==set(TOPICS)
    assert requests[-1]['decision_context']['training_validation_planning']['independent_test']=='failed'


def test_absent_pool_cannot_be_guessed_but_empty_pool_is_explicit():
    payload=facts();payload['decision_context'].pop('qbc_candidates')
    with pytest.raises(ValueError,match='缺少完整'):
        request_summary_first(lambda p:{'tool':'select_dft_candidates'},payload)
    payload['decision_context']['qbc_candidates']=[];seen=[]
    request_summary_first(lambda p:seen.append(deepcopy(p)) or {'tool':'select_dft_candidates'},payload)
    assert seen[-1]['decision_context']['qbc_candidates']==[]


def test_explicit_operations_require_training_and_hull_evidence():
    required=required_topics({'operation':'activate_mlip_candidate'})
    assert {'model','dft','convergence','budget','tasks','configuration'} <= required
    with pytest.raises(ValueError):required_topics({'operation':'activate_mlip_candidate','tool':'generate_branches'})


def test_full_expansion_restores_original_values_without_modifying_sources():
    payload=facts();before=deepcopy(payload);request,sources=summary_payload(payload)
    expand_topics(request,sources,set(TOPICS))
    for key,value in sources.items():
        if key!='decision_context':assert request['state'].get(key,request['decision_context'].get(key))==value
    assert payload==before


def test_missing_model_evidence_blocks_guessing():
    payload=facts();payload['state'].pop('mlip_status')
    for key in ('new_model_generation','training_validation_planning'):payload['decision_context'].pop(key)
    with pytest.raises(ValueError,match='关键决策证据缺失'):
        request_summary_first(lambda p:{'tool':'select_dft_candidates'},payload)

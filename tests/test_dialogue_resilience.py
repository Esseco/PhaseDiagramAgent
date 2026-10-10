from copy import deepcopy
import pytest
from phase_agent.decisions.agent.dialogue_contract import dialogue_errors, normalize_dialogue
from phase_agent.decisions.agent.decision_backend import request_decision_action
from tests.test_unified_react_dialogue import make_handler


def request(responses):
    calls=[]
    def client(payload):
        calls.append(deepcopy(payload))
        response=deepcopy(responses[len(calls)-1])
        response['_llm_usage']={'calls':1,'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}
        return response
    return client,calls


@pytest.mark.parametrize('metadata',[{'reason':'context'}, {'confidence':.8}, {'details':{'summary':'explanation'}}, {'new_annotation':['evidence']}])
def test_optional_annotations_cannot_break_read_only_dialogue(metadata):
    action={'kind':'answer','answer':'Explanation',**metadata}
    original=deepcopy(action)
    assert dialogue_errors(action,True)==[]
    assert normalize_dialogue(action)=={'kind':'answer','answer':'Explanation'}
    assert action==original


@pytest.mark.parametrize('field',['tool','parameters','budget','patch','approved','execute','state','tool_calls','write_requested'])
def test_side_effect_fields_are_not_quarantined(field):
    action={'kind':'answer','answer':'Explanation',field:None}
    assert dialogue_errors(action,True)
    assert normalize_dialogue(action)==action


def test_configuration_routing_remains_strict():
    assert dialogue_errors({'kind':'configure','answer':'Update settings','patch':{}},True)


def test_malformed_dialogue_gets_one_correction_and_usage_is_preserved():
    client,calls=request([{'kind':'answer','answer':'','evidence_refs':[]}, {'kind':'answer','answer':'Corrected explanation','evidence_refs':[]}])
    result=request_decision_action(client,{'unified_dialogue':True,'allowed_tools':['check_convergence'],'instruction':'Explain previous issue'})
    assert result['answer']=='Corrected explanation'
    assert len(calls)==2 and calls[1]['mode']=='dialogue_contract_repair'
    assert calls[1]['validation_errors']
    assert result['_llm_usage']['calls']==2


def test_failed_correction_is_bounded():
    client,calls=request([{'kind':'answer','answer':''},{'kind':'answer','answer':''}])
    with pytest.raises(ValueError):
        request_decision_action(client,{'unified_dialogue':True,'allowed_tools':[]})
    assert len(calls)==2


def test_correction_cannot_escalate_to_a_scientific_action():
    client,calls=request([{'kind':'answer','answer':'','tool':'generate_branches'},{'tool':'generate_branches','parameters':{},'budget':0}])
    with pytest.raises(ValueError,match='remain a dialogue'):
        request_decision_action(client,{'unified_dialogue':True,'allowed_tools':['generate_branches']})
    assert len(calls)==2


@pytest.mark.parametrize('message',['\u4ec0\u4e48\u95ee\u9898','\u4e3a\u4ec0\u4e48','\u5148\u89e3\u91ca\u4e0d\u8981\u6267\u884c','\u6211\u6ca1\u61c2'])
def test_real_dialogue_graph_answers_without_scientific_execution(tmp_path,message):
    client,calls=request([{'kind':'answer','answer':'Explanation only','reason':'optional model annotation','confidence':.9}])
    handler=make_handler(tmp_path,client)
    handler._run=lambda *a,**kw:pytest.fail('Explanation entered science')
    reply=handler([{'role':'user','content':message}],conversation_id='ordinary')
    assert 'Explanation only' in reply and len(calls)==1

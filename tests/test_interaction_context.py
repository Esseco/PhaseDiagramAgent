from copy import deepcopy
from tests.test_unified_react_dialogue import make_handler
from phase_agent.graphs.dialogue.memory import remember_turn
from phase_agent.tools.step_runner.file_protocol import read_json


def test_followup_receives_previous_failure_and_concise_policy(tmp_path):
    calls=[]
    def model(payload):
        calls.append(payload)
        assert 'validation failed' in payload['conversation_facts']['previous_reply']
        assert 'not a retry' in payload['interaction_policy']['semantics']
        assert '1-3' in payload['interaction_policy']['reply_style']
        return {'kind':'answer','answer':'The previous output failed validation. No task was executed.'}
    handler=make_handler(tmp_path,model)
    remember_turn(handler.state_path,'a','continue','validation failed; no execution')
    handler._run=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Question must not retry science'))
    answer=handler([{'role':'user','content':'\u4ec0\u4e48\u95ee\u9898'}],conversation_id='a')
    assert 'No task was executed' in answer and len(calls)==1


def test_multiple_pending_plans_do_not_block_conversation(tmp_path):
    from phase_agent.tools.step_runner.file_protocol import write_json
    calls=[]
    handler=make_handler(tmp_path,lambda p:calls.append(p) or {'kind':'answer','answer':'Two proposals await review; neither is approved.'})
    handler([{'role':'user','content':'hello'}],conversation_id='a')
    state=read_json(handler.state_path,{})
    state['pending_execution_policies']={key:{'status':'awaiting_approval','agent_proposal':{'recommended_action':'generate_branches','raw_action':{'task_key':key}}} for key in ['p1','p2']}
    state['effective_decisions']={'p1':{'status':'reserved'}}
    pending=deepcopy(state['pending_execution_policies'])
    write_json(handler.state_path,state)
    handler._run=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Question entered science'))
    reply=handler([{'role':'user','content':'Explain both proposals'}],conversation_id='a')
    assert 'Two proposals' in reply
    facts=calls[-1]['conversation_facts']['pending_plan_facts']
    assert {row['plan_id'] for row in facts}=={'p1','p2'}
    assert next(row for row in facts if row['plan_id']=='p1')['task_status']=='reserved'
    assert read_json(handler.state_path,{})['pending_execution_policies']==pending

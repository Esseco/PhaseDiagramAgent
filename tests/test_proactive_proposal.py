import pytest
from phase_agent.tools.workflows.propose_followup import propose_after_generation


def fixture():
    result={'status':'completed','state':{},'events':[{'action':{'tool':'generate_branches'},'execution':{'status':'completed','result':{'summary':{'registered_structures':156,'registered_branches':72}}}}],'steps_executed':1}
    frame={'execution_mode':'interactive','agent_client':lambda _:None,'invocation_id':'g1','config_session':{},'effective_registry':{},'context':{},'effective_config':{}}
    return frame,result


def test_new_proposal_never_inherits_approval():
    frame,result=fixture();calls=[]
    def proposal(state,session,**kwargs):
        calls.append(kwargs)
        assert kwargs['human_feedback'] is None and kwargs['replay_record'] is None
        assert kwargs['execution_mode']=='interactive' and kwargs['invocation_id']=='g1:next-proposal'
        return {'status':'awaiting_approval','state':{'pending_execution_policies':{'next':{}}},'events':[{'status':'awaiting_approval'}],'steps_executed':1}
    combined=propose_after_generation(frame,result,event_loop=proposal)
    assert len(calls)==1 and len(combined['events'])==2
    assert combined['steps_executed']==1
    assert combined['completed_generation']['registered_structures']==156


@pytest.mark.parametrize('case',['failed','pending','empty','offline','no_model'])
def test_skip_when_not_safe(case):
    frame,result=fixture()
    if case=='failed':result['status']='failed'
    elif case=='pending':result['state']['pending_tasks']=[{}]
    elif case=='empty':result['events'][0]['execution']['result']['summary']['registered_structures']=0
    elif case=='offline':frame['execution_mode']='autonomous'
    else:frame['agent_client']=None
    def forbidden(*args,**kwargs):raise AssertionError('unexpected request')
    assert propose_after_generation(frame,result,event_loop=forbidden)==result


def test_model_error_preserves_completed_generation():
    frame,result=fixture()
    def fail(*args,**kwargs):raise RuntimeError('model failed')
    combined=propose_after_generation(frame,result,event_loop=fail)
    assert combined['status']=='completed'
    assert combined['events']==result['events']
    assert 'model failed' in combined['followup_error']

from copy import deepcopy
from phase_agent.analysis.state.decision_feedback import decision_summary, recent_decision_feedback
from phase_agent.analysis.state.build_decision_context import build_decision_context
from phase_agent.decisions.agent.action_contracts import action_contract_errors
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from tests import test_execution_policy as policy_tests


def model_action(tool='check_convergence', **extra):
    return {'tool': tool, 'task_key': 'review-1', 'target_ids': [], 'parameters': {}, 'budget': 0, 'reason': 'Use current evidence', 'expected_purpose': 'Review progress', **extra}


def test_optional_explanation_is_bounded_and_does_not_break_execution_contract():
    action = model_action(decision_review='unexpected annotation')
    assert not action_contract_errors(action)
    assert decision_summary(action,{})['observation'] is None
    action['decision_review'] = {'observation': 'x'*1000, 'rationale': 'public reason', 'alternatives':[{'tool':'pause_search','reason':'insufficient evidence','execute':True}], 'expected_outcome':'verify coverage', 'approval':'approve'}
    summary = decision_summary(action, {'active_model_version':'m1', 'confirmed_config_version':'v1'})
    assert len(summary['observation']) == 240
    assert summary['alternatives'] == [{'tool':'pause_search','reason':'insufficient evidence'}]
    assert 'approval' not in summary
    assert summary['model_version'] == 'm1'


def test_feedback_uses_only_matching_outcomes_and_does_not_promote_expectation():
    action = model_action('generate_branches')
    action['task_key']='g1'
    summary=decision_summary(action, {'active_model_version':'m1'})
    state={'action_records':[{'record_id':'r1','status':'completed','final_action':action,'agent_proposal':{'decision_summary':summary}, 'execution_result':{'status':'completed'}}], 'tasks':[{'task_key':'other','status':'completed','actual_cost':99}, {'task_key':'g1','status':'pending','actual_cost':None}], 'generation_history':[{'task_key':'g1','registered_ids':['s1','s2']}]}
    original=deepcopy(state)
    feedback=recent_decision_feedback(state)[0]
    assert feedback['outcome']['matched_task_status']=={'pending':1}
    assert feedback['outcome']['measured_cost'] is None
    assert feedback['outcome']['registered_structure_count']==2
    assert feedback['decision']['expected_outcome']=='Review progress'
    assert state==original
    assert build_decision_context(state)['decision_feedback']==[feedback]


def test_llm_interactive_uses_model_even_for_continue_and_reuses_approval():
    case=policy_tests.ExecutionPolicyTest();case.setUp()
    calls=[]; tools=[]
    def model(payload):
        calls.append(payload)
        return model_action(decision_review={'observation':'No new results yet','rationale':'Inspect evidence before spending','expected_outcome':'Locate coverage gaps','alternatives':[{'tool':'pause_search','reason':'inspection remains possible'}]})
    def handler(**kwargs):
        tools.append(kwargs['action'])
        return {'converged':False}
    registry=create_tool_registry({'check_convergence':handler})
    proposed=run_tool_step(None,case.session,registry=registry,agent_client=model,context={'user_message':'继续'},execution_mode='interactive',invocation_id='online-1')
    assert proposed['status']=='awaiting_approval'
    assert proposed['agent_proposal']['decision_owner']=='llm'
    assert proposed['agent_proposal']['decision_summary']['observation']=='No new results yet'
    assert len(calls)==1 and not tools
    approved=run_tool_step(proposed['state'],case.session,registry=registry,human_feedback='approve',execution_mode='interactive',invocation_id='online-1')
    assert approved['status']=='completed'
    assert len(calls)==1 and len(tools)==1
    feedback=recent_decision_feedback(approved['state'])
    assert feedback[-1]['decision']['alternatives'][0]['tool']=='pause_search'


def test_same_user_message_with_changed_feedback_reaches_model_and_changes_proposal():
    case=policy_tests.ExecutionPolicyTest();case.setUp()
    seen=[]
    def model(payload):
        feedback=payload['decision_context'].get('decision_feedback') or []
        seen.append(feedback)
        failed=bool(feedback and feedback[-1]['outcome']['action_status']=='failed')
        return model_action('pause_search' if failed else 'check_convergence')
    registry=create_tool_registry({'check_convergence':lambda **kwargs:{}, 'pause_search':lambda **kwargs:{}})
    old=model_action();summary=decision_summary(old,{})
    states=[{}, {'action_records':[{'record_id':'old','status':'failed','agent_proposal':{'raw_action':old,'decision_summary':summary}}]}]
    results=[run_tool_step(state,case.session,registry=registry,agent_client=model,context={'user_message':'继续'},execution_mode='interactive',invocation_id=f'new-{i}') for i,state in enumerate(states)]
    assert [r['agent_proposal']['recommended_action'] for r in results]==['check_convergence','pause_search']
    assert seen[0]==[] and seen[1][0]['outcome']['action_status']=='failed'
    assert all(r['status']=='awaiting_approval' and r.get('execution') is None for r in results)

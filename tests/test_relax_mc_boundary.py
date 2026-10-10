from copy import deepcopy
from phase_agent.tools.policy.branch_conditions import branch_entry_errors, mc_entry_facts
from phase_agent.decisions.agent.proposal_validation import proposal_errors
from phase_agent.graphs.proposal_graph import request_with_langgraph
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def initial():
    return {'generation_history':[{'registered_ids':['s1']}], 'active_model_version':'m1'}


def mc():
    return {'tool':'allocate_mc_bohb','task_key':'mc1','parameters':{'mc_budget':300},'budget':300,'reason':'search'}


def test_generated_structure_is_not_relax_evidence():
    state=initial(); old=deepcopy(state)
    assert not mc_entry_facts(state)['ready']
    assert 'mc_requires_recovered_relax_and_current_hull' in branch_entry_errors(mc(),state)
    assert branch_entry_errors({'tool':'prepare_local_batch_files','parameters':{'mode':'relax_inputs'}},state)==[]
    assert state==old


def test_ready_pool_requires_current_model_and_completed_diagram():
    state=initial()
    state.update(current_branch_hull_version='h1',branch_hull_batches={'h1':{'model_version':'m1','records':[{'structure_id':'s1'}]}},phase_diagrams={'mlip':{'status':'completed','model_version':'m1','version':'p1'}})
    assert mc_entry_facts(state)['ready']
    assert not branch_entry_errors(mc(),state)
    state['phase_diagrams']['mlip']['model_version']='m0'
    assert not mc_entry_facts(state)['ready']


def test_model_repairs_premature_mc_into_separate_relax_proposal():
    payload={'instruction':'choose one action','allowed_tools':['allocate_mc_bohb','prepare_local_batch_files'],'decision_context':{'mc_entry_conditions':mc_entry_facts(initial())}}
    calls=[]
    def model(request):
        calls.append(request)
        if len(calls)==1:
            return mc()
        return {'tool':'prepare_local_batch_files','task_key':'relax1','parameters':{'mode':'relax_inputs'},'budget':0,'reason':'Need Relax results before allocating MC'}
    result=request_with_langgraph(model,payload)
    assert len(calls)==2
    assert result['tool']=='prepare_local_batch_files'
    assert 'mc_budget' not in result['parameters']
    assert any(e.startswith('action.prerequisite:') for e in calls[1]['validation_errors'])


def test_mc_input_files_also_blocked_before_relax():
    action={'tool':'prepare_local_batch_files','parameters':{'mode':'mc_inputs'},'budget':0}
    payload={'allowed_tools':['prepare_local_batch_files'],'decision_context':{'mc_entry_conditions':mc_entry_facts(initial())}}
    assert any(e.startswith('action.prerequisite:') for e in proposal_errors(action,payload))


def test_relax_proposal_explains_upload_then_recovery_then_mc():
    action={'tool':'prepare_local_batch_files','parameters':{'mode':'relax_inputs'}}
    result={'status':'awaiting_approval','agent_proposal':{'recommended_action':action['tool'],'raw_action':action,'estimated_cost':{}}}
    reply=format_workflow_reply(result,None,verbose=False)
    assert '由你上传超算并提交' in reply
    assert '本次不分配 MC 步数' in reply
    assert '回收弛豫结果后' in reply

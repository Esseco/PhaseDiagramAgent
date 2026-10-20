from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from phase_agent.tools.policy.model_generation import new_model_generation_facts, new_model_search_errors
from phase_agent.tools.policy.branch_conditions import branch_entry_errors
from phase_agent.decisions.agent.branch_pool_review import review_errors, readable_review, pool_context
from phase_agent.analysis.state.branch_search_evidence import existing_branch_evidence
from phase_agent.tools.relax_mc_batch.plan import diagnose
from phase_agent.decisions.agent.deepagents_proposal import proposal_material


def switched():
    return {'active_model_version':'new','model_refresh':{'status':'completed','new_model_version':'new','old_model_version':'old','plan':{'checksum':'r'}},'generation_history':[],'tasks':[]}


def test_first_new_generation_is_required_then_mixed_targets_are_allowed():
    state = switched()
    facts = new_model_generation_facts(state)
    assert facts["required"] and facts["post_refresh"] and facts["prefer_new_branches"]
    assert new_model_search_errors(state, ["old"]) == ["new_model_branch_generation_required"]
    state["generation_history"] = [
        {
            "task_key": "fresh",
            "model_version": "new",
            "new_branch_ids": ["new"],
            "registered_ids": ["s"],
            "refresh_plan_checksum": "r",
        }
    ]
    assert new_model_search_errors(state, ["old"]) == []
    assert new_model_search_errors(state, ["old", "new"]) == []
    assert new_model_search_errors(state, dft=True) == []
    assert (
        branch_entry_errors(
            {
                "tool": "prepare_local_batch_files",
                "parameters": {"mode": "relax_mc_inputs"},
                "target_ids": ["old"],
            },
            state,
        )
        == []
    )


def test_refresh_is_still_required_before_search():
    state=switched();state['model_refresh']['status']='waiting_results'
    assert 'new_model_structure_refresh_required' in branch_entry_errors({'tool':'prepare_local_batch_files','parameters':{'mode':'relax_mc_inputs'},'target_ids':['old']},state)


def test_retired_history_is_not_generation_and_retired_ids_rejected():
    state=switched()
    state['generation_history']=[{'task_key':'quota','model_version':'new','new_branch_ids':['b'],'registered_ids':['s'],'refresh_plan_checksum':'r'}]
    state['branch_region_review']={'retired_batches':[{'task_key':'quota','branch_ids':['b']}]}
    assert not new_model_generation_facts(state)['ready']
    assert new_model_search_errors(state,['b']) == ['retired_generation_branch_not_available']
    assert pool_context(state)['registered_new_branch_count']==0
    state['generation_history'].append({'task_key':'fresh-approved','model_version':'new','new_branch_ids':['b'],'registered_ids':['s2'],'refresh_plan_checksum':'r'})
    assert new_model_generation_facts(state)['ready']
    assert new_model_search_errors(state,['b'])==[]


def test_same_model_mc_cannot_be_reset_by_repackaging():
    state = switched()
    state["dedup_gate"] = {"status": "ready"}
    state["tasks"] = [
        {"stage": "deep_search", "branch_id": "old", "model_version": "new", "status": "completed"}
    ]
    state["generation_history"] = [
        {
            "task_key": "fresh",
            "model_version": "new",
            "new_branch_ids": ["new"],
            "registered_ids": ["s"],
            "refresh_plan_checksum": "r",
        }
    ]
    with pytest.raises(ValueError, match="branches_already_searched_under_current_model"):
        diagnose(["old"], state, None, {"mlip": {"version": "new"}}, "c")


def review():
    return {'tool':'generate_branches','parameters':{'regions':[]},'branch_pool_review':{'choice':'reuse_existing','sufficiency_for_next_search':'按需旧branch续搜','rationale':'依据当前模型','alternative_reason':'新增收益未证实','missing_evidence':[],'revisit_conditions':['回收后复审'],'evidence_refs':['current'],'existing_branch_groups':[{'decision':'continue_search','branch_ids':['a'],'reason':'新模型基态','evidence_refs':['current']},{'decision':'defer','branch_ids':['b'],'reason':'本轮收益不明','evidence_refs':['current']}]}}


def context():
    return {'branch_pool_review_required':True,'branch_pool_facts':{'existing_branches':[{'branch_id':'a'},{'branch_id':'b'}]},'evidence_catalog':{'references':{'current':{}}}}


@pytest.mark.parametrize('fault',['missing','duplicate','unknown','no_reason','unknown_evidence'])
def test_review_partition_must_be_complete_unique_and_grounded(fault):
    action=review();groups=action['branch_pool_review']['existing_branch_groups']
    if fault=='missing':groups.pop()
    elif fault=='duplicate':groups[1]['branch_ids']=['a']
    elif fault=='unknown':groups[1]['branch_ids']=['retired']
    elif fault=='no_reason':groups[0]['reason']=''
    else:groups[0]['evidence_refs']=['invented']
    assert review_errors(action,context())


def test_review_counts_are_derived_from_selected_ids_not_model_numbers():
    assert not review_errors(review(),context())
    readable=readable_review(review()['branch_pool_review'])
    assert readable['旧branch数量汇总']=={'建议继续搜索':1,'建议暂缓':1}


def test_branch_evidence_only_uses_bound_current_diagram_and_preserves_unknown():
    state={'active_model_version':'new','tasks':[{'task_id':'t','branch_id':'a','structure_id':'s','stage':'deep_search','model_version':'old','status':'completed'}],'phase_records':[{'record_id':'r','source_task_id':'t'}],'phase_diagrams':{'mlip':{'status':'completed','model_version':'new','entries':[{'record_id':'r','source_version':'new','phase_identification_status':'identified','ehull_unit':'eV/atom','ehull':0.0,'is_stable':True}]}}}
    branches={'a':{'P':'O3','x':'1/2'},'b':{'P':'O1','x':'0'}}
    facts=existing_branch_evidence(state,branches)
    assert facts[0]['is_hull_ground_state'] and facts[0]['mc_task_states']=={'other_model:completed':1}
    assert facts[1]['current_model_ehull_ev_per_atom'] is None and not facts[1]['has_mobile_ions']
    state['phase_diagrams']['mlip']['model_version']='old'
    assert all(row['current_model_ehull_ev_per_atom'] is None for row in existing_branch_evidence(state,branches))


def test_deepagent_transport_keeps_full_branch_review_contract():
    facts={'existing_branches':[{'branch_id':'a','current_model_ehull_ev_per_atom':0.001}]}
    material=proposal_material({'decision_context':{'branch_pool_facts':facts,'branch_pool_review_contract':'按需，完整分组','generation_region_contract':'无配额'}})
    assert material['decision_context']['branch_pool_facts']==facts
    assert material['decision_context']['branch_pool_review_contract']=='按需，完整分组'



def test_index_transport_is_exact_mapping_and_invalid_indices_rejected():
    from phase_agent.decisions.agent.branch_pool_review import expand_review_indices
    action=review()
    for index, group in enumerate(action['branch_pool_review']['existing_branch_groups']):
        group.pop('branch_ids')
        group['branch_indices']=[index]
    assert not review_errors(action,context())
    expanded=expand_review_indices(action,context())
    assert expanded['branch_pool_review']['existing_branch_groups'][0]['branch_ids']==['a']
    action['branch_pool_review']['existing_branch_groups'][0]['branch_indices']=[True]
    assert review_errors(action,context())


def test_explicit_model_default_is_expanded_without_program_choosing_it():
    from phase_agent.decisions.agent.branch_pool_review import expand_review_indices
    action=review();review_row=action['branch_pool_review']
    review_row['existing_branch_groups']=[review_row['existing_branch_groups'][0]]
    review_row.update(default_existing_branch_decision='defer',default_existing_branch_reason='Agent明确暂缓其余',default_existing_branch_evidence_refs=['current'])
    expanded=expand_review_indices(action,context())
    assert not review_errors(expanded,context())
    assert expanded['branch_pool_review']['existing_branch_groups'][-1]['branch_ids']==['b']
    assert expanded['branch_pool_review']['existing_branch_groups'][-1]['reason']=='Agent明确暂缓其余'


def test_no_mobile_ion_branch_cannot_be_hidden_in_continuation_group():
    facts=context()
    facts['branch_pool_facts']['existing_branches'][0]['has_mobile_ions']=False
    assert any('x=0' in error for error in review_errors(review(),facts))


def test_generation_origin_is_read_from_executed_proposal_not_invented():
    state={'generation_history':[{'task_key':'old600','new_branch_ids':['b'],'registered_ids':['s'],'model_version':'new'}],
           'action_records':[{'agent_proposal':{'raw_action':{'tool':'generate_branches','task_key':'old600','parameters':{'total_quota':600},'reason':'原配额方案'}}}]}
    facts=pool_context(state)
    assert facts['registered_batch_origin']['sampling_mode']=='quota_plan'
    assert facts['registered_batch_origin']['planned_quota']==600
    assert facts['registered_batch_origin']['reason']=='原配额方案'


def test_normal_agent_call_corrects_once_and_program_exports_its_exact_choices(tmp_path):
    import json
    from phase_agent.decisions.agent.propose_tool_action import propose_agent_tool_action
    from phase_agent.tools.policy.region_sampling import policy
    from phase_agent.tools.policy.execution_policy import build_agent_proposal, apply_execution_policy
    from phase_agent.tools.workflows.region_review import write_proposal
    calls=[]
    valid=review()
    valid.update(task_key='agent-review',target_ids=[],budget=0,reason='Agent决定先复用',expected_purpose='审阅')
    valid['parameters'].update(sampling_mode='region_adaptive',defer_reason='本轮收益证据不足，由Agent暂缓新增')
    # The model explicitly chooses a default; the program only expands identifiers.
    row=valid['branch_pool_review']
    row['existing_branch_groups']=[row['existing_branch_groups'][0]]
    row['existing_branch_groups'][0].pop('branch_ids')
    row['existing_branch_groups'][0]['branch_indices']=[0]
    row.update(default_existing_branch_decision='defer',default_existing_branch_reason='Agent选择暂缓其余候选',default_existing_branch_evidence_refs=['current'])
    def client(payload):
        calls.append(payload)
        if len(calls)==1:
            bad=deepcopy(valid);bad.pop('branch_pool_review');return bad
        assert payload['validation_errors']
        return deepcopy(valid)
    state={'branch_count_policy':policy(),'active_model_version':'new','decision_context':{'branch_pool_facts':context()['branch_pool_facts'],'evidence_catalog':context()['evidence_catalog']},'workspace_layout':{'workspace_root':str(tmp_path)}}
    before=deepcopy(state)
    action=propose_agent_tool_action(state,agent_client=client,allowed_tools=['generate_branches'],config={})
    assert action['decision_source']=='llm_agent',action.get('fallback_reason')
    assert len(calls)==2 and state==before
    assert action['branch_pool_review']['existing_branch_groups'][0]['branch_ids']==['a']
    assert action['branch_pool_review']['existing_branch_groups'][1]['branch_ids']==['b']
    assert action['branch_pool_review']['existing_branch_groups'][1]['reason']=='Agent选择暂缓其余候选'
    envelope=build_agent_proposal(action,{},runtime_state=state)
    assert not apply_execution_policy(envelope,execution_mode='interactive')['execute']
    payload=json.loads(Path(write_proposal(envelope,state)['json']).read_text(encoding='utf-8'))
    assert payload['已有branch是否需要补充']['旧branch数量汇总']=={'建议继续搜索':1,'建议暂缓':1}
    assert payload['已有branch是否需要补充']['旧branch搜索建议'][1]['理由']=='Agent选择暂缓其余候选'


def test_invalid_model_review_stops_after_one_correction():
    from phase_agent.decisions.agent.branch_review_repair import repair_branch_review
    calls=[]
    bad={'tool':'generate_branches','parameters':{'sampling_mode':'region_adaptive','regions':[],'defer_reason':'暂缓'}}
    def client(payload):
        calls.append(payload)
        return deepcopy(bad)
    with pytest.raises(ValueError,match='invalid_after_one_correction'):
        repair_branch_review(client,{'decision_context':context()},bad)
    assert len(calls)==1

def test_region_review_path_uses_confirmed_storage(tmp_path):
    from phase_agent.tools.workflows.region_review import path_for
    state = {'confirmed_config': {'config': {'storage': {'workspace_root': str(tmp_path)}}}}
    assert path_for(state, {'task_key': 'test'}).parent == tmp_path / 'workflow_state' / 'branch_plans'

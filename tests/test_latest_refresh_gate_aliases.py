from copy import deepcopy
from tests.test_model_refresh_workflow import recovered_state
from phase_agent.tools.workflows.model_refresh_gate import advance_refresh


def prepared_with_alias(tmp_path):
    state,_,_,_=recovered_state(tmp_path)
    original=state['tasks'][0]
    second=deepcopy(original);second['task_id']='second';state['tasks'].append(second)
    state['model_refresh']['task_ids'].append('second')
    plan=state['model_refresh']['plan'];plan['energy_policy']='latest_valid_per_structure_v1';plan['policy'].pop('force_followup_version',None)
    state['phase_records']=[{'record_id':'r1','source_task_id':original['task_id'],'source_version':'new','structure_id':'s','structure_sha256':'sha'}, {'record_id':'r2','source_task_id':'second','source_version':'new','structure_id':'s','structure_sha256':'sha'}]
    state['phase_diagrams']['mlip']={'status':'completed','model_version':'new','energy_policy':'latest_valid_per_structure_v1','entries':[{'record_id':'r2','structure_id':'s','source_version':'new','structure_sha256':'sha','ehull':0.002,'ehull_unit':'eV/atom'}], 'equivalent_record_aliases':{'r1':'r2'}}
    return state


def test_matching_alias_completes_without_second_calculation(tmp_path):
    state=prepared_with_alias(tmp_path);updated,waiting=advance_refresh(state)
    assert waiting is None and updated['model_refresh']['status']=='completed'
    assert len(updated['tasks'])==2


def test_cross_structure_alias_blocks_refresh(tmp_path):
    state=prepared_with_alias(tmp_path);state['phase_diagrams']['mlip']['entries'][0]['structure_id']='other'
    updated,waiting=advance_refresh(state)
    assert waiting and '谱系无法核对' in waiting and updated['model_refresh']['status']=='waiting_results'

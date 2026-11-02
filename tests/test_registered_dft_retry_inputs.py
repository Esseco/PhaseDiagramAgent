import pytest
from phase_agent.tools.workflows.create_dft_selection_handler import registered_retry_input_ids


def test_only_explicit_same_model_unpacked_retry_can_join_input_preparation():
    task={'task_id':'R','retry_of':'F','stage':'dft_single_point','status':'pending',
          'model_version':'m','config_version':'c'}
    state={'tasks':[task],'active_model_version':'m','confirmed_config_version':'c'}
    assert registered_retry_input_ids({'parameters':{}},state)==[]
    action={'parameters':{'registered_retry_input_ids':['R']}}
    assert registered_retry_input_ids(action,state)==['R']
    for field,value in [('retry_of',None),('stage','deep_search'),('status','completed'),
                        ('model_version','old'),('config_version','old'),('slurm_batch_id','submitted')]:
        with pytest.raises(ValueError):
            registered_retry_input_ids(action,{**state,'tasks':[{**task,field:value}]})


def test_unknown_or_duplicate_retry_input_scope_is_rejected():
    for ids in [['unknown'],['R','R'],'R']:
        with pytest.raises(ValueError):
            registered_retry_input_ids({'parameters':{'registered_retry_input_ids':ids}},{'tasks':[]})

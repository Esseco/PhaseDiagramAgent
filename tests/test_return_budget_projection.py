import copy
from phase_agent.tools.relax_mc_batch.results import _reserve_returned_budget
from phase_agent.tools.budget.reserve_budget import reserve_budget


def test_projected_reservation_matches_full_policy():
    limits={'enforcement':'hard','total_relative_cost':3,'stage_limits':{'deep_search':{'max_tasks':2,'max_cost':3}}}
    state={'tasks':[{'status':'completed'}], 'pending_execution_policies':{'original':{'decision':'pending'}}}
    expected=copy.deepcopy(state)
    for task_key in ['one','one','two','three']:
        args=dict(task_key=task_key,stage='deep_search',amount=1,limits=limits,config_version='c',model_version='m')
        full=reserve_budget(expected,**args)
        actual=_reserve_returned_budget(state,**args)
        assert actual['status']==full['status']
        if full['status']!='rejected': expected=full['state']
        assert state==expected


def test_scientific_state_not_copied_per_returned_task():
    class NoCopy:
        def __deepcopy__(self,memo):raise AssertionError('large scientific state must not be copied')
    large=NoCopy();state={'scientific_results':large}
    limits={'enforcement':'advisory','total_relative_cost':1,'stage_limits':{'deep_search':{}}}
    for i in range(100):
        assert _reserve_returned_budget(state,task_key=str(i),stage='deep_search',amount=1,limits=limits)['status']=='reserved'
    assert state['scientific_results'] is large
    assert len(state['budget_reservations'])==100

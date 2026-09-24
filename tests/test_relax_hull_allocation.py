import pytest
from pymatgen.core import Lattice, Structure
from analysis_layer.phase.branch_relax_hull import build_relax_hull, hull_energy_per_atom, rank_relaxed_branches
from execution_layer.workflows.prepare_branch_relaxation import prepare_branch_relaxation
from config_layer.defaults.default_budget_rules import default_budget_rules
from data_layer.ledger.phase_data_manager import PhaseDataManager
from scientific_layer.bohb.select_bohb_candidates import select_bohb_candidates
from scientific_layer.bohb.default_bohb_config import default_bohb_config
from execution_layer.workflows.create_active_learning_handlers import _allocate_mc_bohb


def row(branch, composition, energy):
    return dict(branch_id=branch, structure_id=branch, structure_path='relaxed.vasp',
                composition=composition, energy=energy, energy_unit='eV', converged=True,
                model_version='m1')


def test_convex_mixture_not_same_composition_minimum():
    pool = build_relax_hull([row('a', {'Na': 1}, 0), row('b', {'Fe': 1}, -2),
                            row('c', {'Na': 1, 'Fe': 1}, -1),
                            row('c', {'Na': 1, 'Fe': 1}, -.8)], model_version='m1')
    assert hull_energy_per_atom(pool, {'Na': 1, 'Fe': 1}) == pytest.approx(-1)
    assert hull_energy_per_atom(pool, {'O': 1}) is None
    ranked, missing = rank_relaxed_branches([{'branch_id': 'c'}], pool)
    assert ranked[0]['relaxed_ehull'] == pytest.approx(.5)
    assert not missing
    assert not build_relax_hull(pool['records'], model_version='m2')['records']


def test_uncertainty_affects_allocation_with_exploration():
    pool = [dict(branch_id=str(i), allocation_score=score) for i, score in enumerate([.4, .1, -.2, .2])]
    result = select_bohb_candidates(pool, [], count=2, seed=1,
        config={'selection_policy': 'relax_hull_uncertainty', 'random_fraction': .0})
    assert result['model_mode'] == 'relax_hull_uncertainty'
    assert any(c['branch_id'] == '2' for c in result['selected'])


def test_screening_reserves_relax_before_any_mc_and_reuses_results(tmp_path):
    h = [[1,0,0],[0,1,0],[0,0,1]]
    manager = PhaseDataManager({'P':['O3'], 'H':{'O3':[h]}, 'TM_ratio':{'Fe':1}})
    bid = manager.add_branch(P='O3', H=h, x=1, T=['Fe'], composition={'Fe':1})
    path = tmp_path/'initial.vasp'
    Structure(Lattice.cubic(3), ['Fe'], [[0,0,0]]).to(filename=str(path), fmt='poscar')
    for i in range(3): manager.add_structure(branch_id=bid, arrangement={'i':i}, source_path=path)
    config = {'bohb':{'relax_structures_per_branch':3}, 'mlip':{'version':'m1'}, 'budgets':default_budget_rules()}
    context = {'manager':manager, 'phase_references':{}, 'effective_config':config, 'config_version':'v1'}
    candidates = [manager.data['branches'][bid]]
    first = prepare_branch_relaxation(candidates, {}, context)
    assert len(first['state']['pending_tasks']) == 3
    assert {t['stage'] for t in first['state']['tasks']} == {'relax_and_feature'}
    assert len(prepare_branch_relaxation(candidates, first['state'], context)['state']['tasks']) == 3
    state = first['state']; state['pending_tasks'] = []
    for i, task in enumerate(state['tasks']):
        task.update(status='completed', converged=True, outputs={'composition':{'Fe':1},
            'structure_path':str(path), 'energy':-i, 'energy_unit':'eV'})
    ready = prepare_branch_relaxation(candidates, state, context)
    assert ready['status'] == 'ready'
    assert ready['candidates'][0]['relaxed_energy_per_atom'] == -2
    assert ready['candidates'][0]['branch_energy_std_per_atom'] == pytest.approx(0.81649658)
    assert ready['candidates'][0]['structure_id'] == state['tasks'][-1]['structure_id']


def test_agent_branch_batch_is_filtered_before_relax(tmp_path):
    h = [[1,0,0],[0,1,0],[0,0,1]]
    manager = PhaseDataManager({'P':['O3'], 'H':{'O3':[h]}, 'TM_ratio':{'Fe':1}})
    selected = manager.add_branch(P='O3', H=h, x=1, T=['Fe'], composition={'Fe':1})
    excluded = manager.add_branch(P='O3', H=h, x=.5, T=['Fe'], composition={'Fe':2})
    path = tmp_path/'initial.vasp'
    Structure(Lattice.cubic(3), ['Fe'], [[0,0,0]]).to(filename=str(path), fmt='poscar')
    for branch_id in (selected, excluded):
        for i in range(3):
            manager.add_structure(branch_id=branch_id, arrangement={'i':i}, source_path=path)
    config = {'bohb': default_bohb_config(), 'mlip': {'version':'m1'},
              'budgets': default_budget_rules(), 'seed': 42,
              'structure_directory': str(tmp_path/'structures'),
              'round_strategy': {'minimum_exploration_fraction': .1}}
    result = _allocate_mc_bohb(action={'task_key':'agent-batch', 'target_ids':[selected],
        'parameters':{'mc_budget':90, 'dft_budget':0, 'exploration_fraction':.1},
        'reason':'selected by agent'}, context={'manager':manager, 'phase_references':{},
        'effective_config':config, 'event_state':{}, 'config_version':'v1'})
    assert result['screening'] == 'screening_pending'
    assert result['state']['branch_batch']['status'] == 'relax_pending'
    assert {task['branch_id'] for task in result['state']['tasks']} == {selected}
    assert len(result['state']['tasks']) == 3

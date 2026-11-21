from copy import deepcopy

import pytest

from phase_agent.science.training.training_preflight_selection import selected_records, split_independent_holdout


def fake_state():
    old = [{'data_id': f'data-{i}', 'task_id': f'task-{i}', 'branch_id': f'b-{i}',
            'geometry': f'geo-{i}', 'checks_passed': True, 'training_ready': True,
            'status': 'completed', 'converged': True} for i in range(30)]
    new = {'data_id': 'approved', 'task_id': 'except', 'branch_id': 'b-except',
           'geometry': 'geo-except', 'checks_passed': True, 'training_ready': True,
           'status': 'completed', 'converged': True}
    state = {'dft_training_records': old + [new], 'tasks': [{'task_id': 'except',
             'status': 'completed', 'input_structure_sha256': 'sha', 'model_version': 'model'}]}
    policy = {'include_extrapolated_dft': [{'task_id': 'except', 'input_structure_sha256': 'sha',
              'model_version': 'model', 'user_instruction': 'approved'}],
              'independent_test_fraction': 0.1, 'seed': 2}
    return old, new, state, policy


@pytest.fixture
def stubs(monkeypatch):
    import phase_agent.science.training.cumulative_training_records as cumulative
    import phase_agent.science.training.prepare_mace_finetune as module
    import phase_agent.evidence.dft_extrapolation as extrapolation
    import phase_agent.analysis.convergence.independent_evidence as evidence
    import phase_agent.science.training.training_preflight_selection as design
    monkeypatch.setattr(cumulative, 'cumulative_training_records',
                        lambda state: [x for x in state['dft_training_records'] if x['task_id'] != 'except'])
    monkeypatch.setattr(extrapolation, 'failed_dft_task_ids', lambda state: {'except'})
    monkeypatch.setattr(module, '_to_atoms', lambda record, labels: object())
    monkeypatch.setattr(evidence, 'geometry_key', lambda record: record.get('geometry'))
    monkeypatch.setattr(design, 'structure_conflicts', lambda training, test: [])
    monkeypatch.setattr(design, 'validate_exception_frame', lambda task, record: None)


def test_exact_approved_exception_and_no_broad_override(stubs):
    old, new, state, policy = fake_state()
    selected = selected_records(state, policy)
    assert len(selected) == 31
    assert sum(x['task_id'] == 'except' for x in selected) == 1
    assert len(selected_records(state, {})) == len(old)
    bad = deepcopy(policy)
    bad['include_extrapolated_dft'][0]['input_structure_sha256'] = 'changed'
    with pytest.raises(ValueError, match='identity_or_status'):
        selected_records(state, bad)
    state['dft_training_records'][-1]['training_ready'] = False
    with pytest.raises(ValueError, match='not_validated'):
        selected_records(state, policy)


def test_holdout_is_novel_disjoint_grouped_and_approved_frame_trains(stubs):
    _, _, state, policy = fake_state()
    rows = selected_records(state, policy)
    base = {'training_provenance_complete': True, 'training_data_ids': ['data-0'],
            'training_structure_hashes': ['geo-1']}
    tr, te, spec = split_independent_holdout(rows, base, {}, policy)
    assert any(x['task_id'] == 'except' for x in tr)
    assert len(te) >= 4 and len(tr) + len(te) == 31
    assert {x['branch_id'] for x in tr}.isdisjoint(x['branch_id'] for x in te)
    assert not ({x['data_id'] for x in te} & set(base['training_data_ids']))
    assert not ({x['geometry'] for x in te} & set(base['training_structure_hashes']))
    assert spec['data_ids'] == sorted(x['data_id'] for x in te)
    with pytest.raises(ValueError, match='provenance_incomplete'):
        split_independent_holdout(rows, {}, {}, policy)


def test_holdout_fails_closed_if_duplicate_geometry_crosses_groups(stubs):
    _, _, state, policy = fake_state()
    rows = selected_records(state, policy)
    base = {'training_provenance_complete': True, 'training_data_ids': [],
            'training_structure_hashes': []}
    tr, te, _ = split_independent_holdout(rows, base, {}, policy)
    holdout_geometry = te[0]['geometry']
    next(x for x in rows if x['data_id'] == tr[0]['data_id'])['geometry'] = holdout_geometry
    with pytest.raises(ValueError, match='geometry_overlap'):
        split_independent_holdout(rows, base, {}, policy)


def test_rejects_conflicting_holdout_group_and_selects_other_groups(stubs, monkeypatch):
    import phase_agent.science.training.training_preflight_selection as design
    _, _, state, policy = fake_state()
    rows = selected_records(state, policy)
    base = {'training_provenance_complete': True, 'training_data_ids': [],
            'training_structure_hashes': []}
    first = []
    def conflict_once(training, test):
        if not first:
            first.append(test[0]['data_id'])
            return [(first[0], training[0]['data_id'])]
        return []
    monkeypatch.setattr(design, 'structure_conflicts', conflict_once)
    _, test, spec = split_independent_holdout(rows, base, {}, policy)
    assert first[0] not in {row['data_id'] for row in test}
    assert spec['rejected_overlap_groups']


def test_prepared_package_reserves_test_from_all_training_and_cv(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from ase.io import read
    from phase_agent.configuration.defaults.default_mace_committee_config import default_mace_committee_config
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune
    from phase_agent.science.training.cumulative_training_records import cumulative_training_records
    import phase_agent.science.training.training_preflight_selection as selection
    from tests.test_dft_comparison_csv import add_result

    state = {}
    for index in range(10):
        add_result(state, task_id=f'T{index}')
    state['training_preflight_policy'] = {'test': True}
    def selected(current, policy):
        return cumulative_training_records(current)
    def split(records, base, settings, policy):
        return records[:-2], records[-2:], {
            'groups': ['branch_id:B-T8', 'branch_id:B-T9'],
            'data_ids': sorted(r['data_id'] for r in records[-2:]),
            'fraction_requested': 0.2, 'seed': 2026, 'rejected_overlap_groups': [],
        }
    monkeypatch.setattr(selection, 'selected_records', selected)
    monkeypatch.setattr(selection, 'split_independent_holdout', split)
    monkeypatch.setattr(selection, 'group_equivalent_training_records', lambda rows, settings: ([{**r, 'independent_group_id': r['branch_id']} for r in rows], {'source_group_count': len(rows), 'independent_group_count': len(rows), 'matched_group_merges': 0, 'source_to_independent_group': {}}))
    settings = default_mace_committee_config()
    settings['labels']['include_stress'] = False
    result = prepare_remote_finetune(state, {'mlip_finetune': settings,
        'upload_batches_directory': str(tmp_path),
        'mlip': {'version': 'm1', 'model_path': '/remote/base.model'},
        'python_environments': {'remote_mlip': 'mace'}})
    assert result['status'] == 'awaiting_remote_training' and not result['submitted']
    job = next(iter(result['state']['remote_finetune_jobs'].values()))
    root = Path(job['directory']) / 'inputs'
    plan = json.loads((root / 'training_plan.json').read_text(encoding='utf8'))
    train = read(root / '_shared_data/train.xyz', index=':')
    test = read(root / '_shared_data/test.xyz', index=':')
    ids = {a.info['data_id'] for a in test}
    assert len(train) == 8 and len(test) == 2
    assert ids.isdisjoint(a.info['data_id'] for a in train)
    assert plan['independent_holdout']['status'] == 'reserved_not_evaluated'
    assert plan['main_model']['independent_test'] is False
    assert set(plan['training_snapshot']['training_data_ids']).isdisjoint(ids)
    for i in range(1, 6):
        for split_name in ('train', 'valid'):
            cv = read(root / f'main_cv_{i}' / f'{split_name}.xyz', index=':')
            assert ids.isdisjoint(a.info['data_id'] for a in cv)


def test_equivalent_structures_share_cv_group():
    from pymatgen.core import Lattice, Structure
    from phase_agent.science.training.training_preflight_selection import group_equivalent_training_records

    species = ['H', 'H', 'He', 'Li', 'Be', 'B', 'C']
    records = [{'branch_id': f'b-{i}', 'data_id': f'd-{i}',
                'structure': Structure(Lattice.cubic(5), [el], [[0, 0, 0]]).as_dict()}
               for i, el in enumerate(species)]
    rows, summary = group_equivalent_training_records(records, {})
    assert rows[0]['independent_group_id'] == rows[1]['independent_group_id']
    assert summary['source_group_count'] == 7
    assert summary['independent_group_count'] == 6
    assert summary['matched_group_merges'] == 1


def test_exception_accepts_only_identical_queue_mirror(stubs):
    _, new, state, policy = fake_state()
    state['new_dft_records'] = [deepcopy(new)]
    rows = selected_records(state, policy)
    row = next(r for r in rows if r['task_id'] == 'except')
    assert row['is_current_round'] is True
    state['new_dft_records'][0]['geometry'] = 'different'
    with pytest.raises(ValueError, match='ambiguous_record'):
        selected_records(state, policy)


def test_authorized_static_input_must_match_dft_frame(tmp_path):
    import hashlib
    from pymatgen.core import Lattice, Structure
    from phase_agent.science.training.training_preflight_selection import validate_exception_frame

    frame = Structure(Lattice.cubic(5), ['Na', 'O'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    path = tmp_path / 'POSCAR'
    frame.to(filename=str(path), fmt='poscar')
    task = {'stage': 'dft_single_point', 'structure_path': str(path),
            'input_structure_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {'stage': 'dft_single_point', 'structure': frame.as_dict()}
    validate_exception_frame(task, record)
    changed = frame.copy()
    changed.translate_sites([1], [0.1, 0, 0])
    with pytest.raises(ValueError, match='frame_mismatch'):
        validate_exception_frame(task, {'stage': 'dft_single_point', 'structure': changed.as_dict()})
    path.write_text('changed')
    with pytest.raises(ValueError, match='unavailable_or_changed'):
        validate_exception_frame(task, record)

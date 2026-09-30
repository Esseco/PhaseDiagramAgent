import json
from pathlib import Path

from execution_layer.local.migrate_allocation_layout import migrate
from execution_layer.remote.build_upload_batch_directory import build_upload_batch_directory


def test_migration_preserves_inputs_and_relinks_second_segment(tmp_path):
    current = tmp_path / 'current'
    current.mkdir()
    root = tmp_path / 'upload_batches' / 'MLIP-round-0001_model'
    stage = root / 'MC-search'
    batch = stage / 'MC-sampling-0001_remote-000001'
    batch.mkdir(parents=True)
    task_path = batch / 'task.json'
    task_path.write_bytes(b'immutable input')
    structure = stage / 'results' / 'MC-first' / 'final.vasp'
    structure.parent.mkdir(parents=True)
    structure.write_bytes(b'structure')
    second = stage / 'allocation-0002_segment-02_second'
    second.mkdir()
    state = {'generation_history': [{'registered_ids': ['S1']}],
             'slurm_batches': [{'batch_id': 'remote-000001', 'model_version': 'model',
              'calculation_group': 'MC-search', 'task_ids': ['first'],
              'upload_directory': str(batch), 'results_directory': str(stage / 'results')}],
             'tasks': [{'task_id': 'first', 'slurm_batch_id': 'remote-000001',
                        'input_path': str(task_path), 'structure_path': str(structure), 'energy': -5},
                       {'task_id': 'second', 'structure_path': str(structure)}]}
    state_path = current / 'state.json'
    state_path.write_text(json.dumps(state))
    cache = current / 'phase_identification_cache.json'
    cache.write_text(json.dumps({str(structure): {'phase': 'O3', 'hash': 'unchanged'}}))
    preview = migrate(state_path, root)
    assert preview['status'] == 'validated_only' and batch.exists()
    result = migrate(state_path, root, apply=True)
    updated = json.loads(state_path.read_text())
    assert result['batch_count'] == 1 and result['file_count'] == 2
    assert not batch.exists() and second.exists()
    assert Path(updated['tasks'][0]['input_path']).read_bytes() == b'immutable input'
    assert Path(updated['tasks'][1]['structure_path']).read_bytes() == b'structure'
    assert updated['tasks'][0]['energy'] == -5
    assert json.loads(cache.read_text())[updated['tasks'][1]['structure_path']]['hash'] == 'unchanged'
    assert (Path(result['backup_directory']) / 'current' / 'state.json').is_file()
    assert migrate(state_path, root)['status'] == 'already_organized'
    moved = migrate(state_path, root, apply=True, search_groups=True)
    assert moved['status'] == 'migrated'
    updated = json.loads(state_path.read_text())
    assert 'Search-group-0001' in Path(updated['tasks'][0]['input_path']).parts
    assert Path(updated['tasks'][1]['structure_path']).read_bytes() == b'structure'


def test_dft_selection_uses_one_round_across_sp_and_relax(tmp_path):
    state = {}
    single, one = build_upload_batch_directory(tmp_path, state, batch_id='remote-000001',
        stage='dft_single_point', model_version='model', operation_id='selection-a')
    relax, same = build_upload_batch_directory(tmp_path, state, batch_id='remote-000002',
        stage='dft_relax', model_version='model', operation_id='selection-a')
    later, two = build_upload_batch_directory(tmp_path, state, batch_id='remote-000003',
        stage='dft_single_point', model_version='model', operation_id='selection-b')
    assert single.parent.name == relax.parent.name == 'DFT-round-0001_selection-a'
    assert later.parent.name == 'DFT-round-0002_selection-b'
    assert one['operation_index'] == same['operation_index'] == 1
    assert two['operation_index'] == 2


def test_relax_and_two_mc_rounds_share_branch_generation_group(tmp_path):
    state = {}
    relax, _ = build_upload_batch_directory(tmp_path, state, batch_id='r1',
        stage='relax_and_feature', model_version='model', operation_id='relax', search_group_index=1)
    mc1, _ = build_upload_batch_directory(tmp_path, state, batch_id='r2', stage='deep_search',
        model_version='model', operation_id='mc1', segment_index=0, search_group_index=1)
    mc2, _ = build_upload_batch_directory(tmp_path, state, batch_id='r3', stage='deep_search',
        model_version='model', operation_id='mc2', segment_index=1, search_group_index=1)
    assert relax.parent.parent == mc1.parent.parent == mc2.parent.parent
    assert relax.parent.name == 'Relax-0001'
    assert mc1.parent.name == 'Relax-0001_MC-round-0001'
    assert mc2.parent.name == 'Relax-0001_MC-round-0002'

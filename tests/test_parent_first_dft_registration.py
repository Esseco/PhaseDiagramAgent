import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_registration_binds_approved_selection_and_rejects_task_tampering(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.io import write
    from phase_agent.tools.workflows import parent_first_result
    from phase_agent.tools.workflows.parent_first_dft import prepare_parent_first_dft
    from phase_agent.tools.workflows.parent_first_dft_registration import register_parent_first_dft_tasks

    source = tmp_path / 'selected.vasp'
    write(source, Atoms('NaCoO2', positions=[[0,0,0],[2,0,0],[0,2,0],[0,0,2]],
                        cell=[6,6,6], pbc=True), format='vasp')
    settings = {'config_version': 'config-1', 'model_version': 'model-1',
                'approval_record_id': 'approved-1', 'parameters': {}}
    manifest = {'schema': 'parent-first-hpc-input-v1', 'scope_id': 'scope',
                'round_evidence_digest': 'digest', 'target': 1, 'parents': [{'record_id':'parent'}],
                'frames_per_parent': 36, 'seed': 42, 'min_distance_A': 1.0,
                'volume_ratio': [0.75,1.25], 'dft_settings': settings}
    manifest_path = tmp_path/'input_manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf8')
    checked = {'status':'verified', 'target':1, 'result_sha256':'a'*64,
               'binding':{'input_manifest_sha256':sha(manifest_path), 'scope_id':'scope',
                          'round_evidence_digest':'digest', 'baseline_hull_version':'hull'},
               'selected':[{'structure_path':str(source), 'structure_sha256':sha(source),
                            'parent_record_id':'parent', 'branch_id':'branch'}]}
    monkeypatch.setattr(parent_first_result, 'validate_parent_first_result', lambda *args: checked)
    dft_dir = tmp_path/'dft'
    receipt = prepare_parent_first_dft(tmp_path, tmp_path, dft_dir, settings)
    params = {'mode':'parent_first_recovery_inputs', 'scope_id':'scope', 'round_evidence_digest':'digest',
              'parent_record_ids':['parent'], 'direct_target':1, 'frames_per_parent':36, 'seed':42,
              'min_distance_A':1.0, 'volume_ratio':[0.75,1.25], 'dft_parameters':{}}
    state = {'active_model_version':'model-1', 'confirmed_config_version':'config-1',
             'parent_first_recovery_batches':{'batch':{'approval_record_id':'approved-1',
                 'scope_id':'scope','round_evidence_digest':'digest','target':1,'status':'prepared'}},
             'tasks':[], 'action_records':[{'record_id':'approved-1','status':'prepared',
                'human_feedback':{'decision':'approve'}, 'final_action':{
                    'tool':'prepare_local_batch_files', 'operation':'prepare_parent_first_recovery',
                    'parameters':params}}]}
    result = register_parent_first_dft_tasks(state, tmp_path, tmp_path, dft_dir)
    assert not state['tasks'] and len(result['state']['tasks']) == 1
    task = result['state']['tasks'][0]
    assert task['task_id'] == receipt['tasks'][0]['task_id']
    assert task['structure_path'] == str(source) and task['upload_operation_id'] == receipt['batch_id']
    stale = deepcopy(state)
    stale['action_records'][0]['final_action']['parameters']['dft_parameters'] = {'incar_settings': {'static': {'ENCUT': 300}}}
    with pytest.raises(ValueError, match='parent_first_dft_approval_invalid'):
        register_parent_first_dft_tasks(stale, tmp_path, tmp_path, dft_dir)
    assert not stale['tasks']
    assert register_parent_first_dft_tasks(result['state'], tmp_path, tmp_path, dft_dir)['status'] == 'already_registered'
    registered_with_results = register_parent_first_dft_tasks(state, tmp_path, tmp_path, dft_dir, results_dir=tmp_path/'results')
    completed = deepcopy(registered_with_results['state'])
    expected_result = completed['tasks'][0]['result_path']
    completed['tasks'][0].update(status='failed',result_path='.',calculation_result_path=expected_result)
    assert register_parent_first_dft_tasks(completed,tmp_path,tmp_path,dft_dir,results_dir=tmp_path/'results')['status']=='already_registered'
    completed['tasks'][0]['calculation_result_path'] = str(tmp_path/'other'/'result.json')
    with pytest.raises(ValueError,match='registration_conflict'):
        register_parent_first_dft_tasks(completed,tmp_path,tmp_path,dft_dir,results_dir=tmp_path/'results')
    task_file = dft_dir/receipt['tasks'][0]['directory']/'task.json'
    data = json.loads(task_file.read_text(encoding='utf8'))
    data['model_version'] = 'other-model'
    task_file.write_text(json.dumps(data), encoding='utf8')
    with pytest.raises(ValueError, match='parent_first_dft_task_invalid'):
        register_parent_first_dft_tasks(state, tmp_path, tmp_path, dft_dir)
    assert not state['tasks']


def test_registration_requires_approved_action(tmp_path):
    from phase_agent.tools.workflows.parent_first_dft_registration import register_parent_first_dft_tasks
    with pytest.raises(ValueError, match='parent_first_dft_approval_invalid'):
        register_parent_first_dft_tasks({}, tmp_path, tmp_path, tmp_path)


def test_stages_exact_structures_in_private_ledger_copy(tmp_path):
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager, _stable_id
    from phase_agent.tools.workflows.parent_first_dft_registration import stage_parent_first_structures
    h = [[1,0,0],[0,1,0],[0,0,1]]
    manager = PhaseDataManager({'P':['O3'], 'H':{'O3':[h]}, 'TM_ratio':{'Ni':1,'Co':1}})
    branch = manager.add_branch(P='O3', H=h, x=1, T=['Ni','Co'], composition={'Na':1,'Ni':1,'Co':1,'O':2})
    source = tmp_path/'selected.vasp'; source.write_bytes(b'structure')
    arrangement = {'parent_first_sha256':sha(source), 'parent_record_id':'parent'}
    sid = _stable_id('S', {'branch_id':branch, 'arrangement':arrangement})
    task = {'branch_id':branch, 'arrangement':arrangement, 'structure_path':str(source),
            'input_structure_sha256':sha(source), 'structure_id':sid, 'upload_operation_id':'batch'}
    staged = stage_parent_first_structures(manager, [task])
    assert not manager.data['structures'] and staged.data['structures'][sid]['source_path'] == str(source)
    assert staged.data['structures'][sid]['metadata']['perturbation_recovery_batch'] == 'batch'
    with pytest.raises(ValueError, match='structure_identity_mismatch'):
        stage_parent_first_structures(manager, [{**task, 'structure_id':'incorrect'}])
    assert not manager.data['structures']

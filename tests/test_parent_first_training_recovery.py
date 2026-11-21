import hashlib

from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
from phase_agent.evidence.parent_first_recovery import advance_parent_first_recovery
from phase_agent.tools.workflows.perturbation_direct import previously_approved_recovery_action


def make_state(tmp_path, monkeypatch, count):
    from phase_agent.tools.workflows import parent_first_result
    from phase_agent.science.dft import spin_acceptance
    rows, labels, selected = [], [], []
    for i in range(count):
        path = tmp_path/f'{i}.vasp'; path.write_bytes(str(i).encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'task_id':f't{i}', 'structure_id':f's{i}', 'structure_path':str(path),
                     'input_structure_sha256':digest, 'stage':'dft_single_point',
                     'status':'failed' if i == 0 else 'completed', 'model_version':'bad',
                     'upload_operation_id':'operation'})
        selected.append({'structure_sha256':digest})
        if i:
            labels.append({'task_id':f't{i}', 'status':'completed','converged':True,'checks_passed':True,
                           'final_frame_valid':True, 'training_ready':True})
    monkeypatch.setattr(parent_first_result, 'validate_parent_first_result', lambda *a:{
        'status':'verified', 'target':count, 'result_sha256':'digest', 'selected':selected})
    monkeypatch.setattr(spin_acceptance, 'spin_standard_passed', lambda row: True)
    baseline = {'status':'completed', 'version':'old-hull', 'model_version':'old'}
    return {'active_model_version':'bad', 'tasks':rows, 'dft_dataset_records':labels,
            'phase_diagrams':{'mlip':baseline}, 'model_refresh':{'new_model_version':'bad','old_diagram':baseline},
            'action_records':[{'record_id':'approval','status':'prepared','human_feedback':{'decision':'approve'},
                'final_action':{'operation':'prepare_parent_first_recovery',
                                'parameters':{'scope_id':'epoch_refresh','round_evidence_digest':'round'}}}],
            'parent_first_recovery_batches':{'batch':{'status':'waiting_dft_results', 'target':count,
                'scope_id':'epoch_refresh','round_evidence_digest':'round','model_version':'bad','approval_record_id':'approval',
                'input_dir':'input','selection_dir':'selection','sampling_result_sha256':'digest',
                'dft_task_ids':[f't{i}' for i in range(count)]}}}


def test_less_than_twenty_percent_failure_uses_only_successful_labels(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch, 6)
    updated = advance_parent_first_recovery(state)
    facts = recovery_label_facts(updated)
    assert facts['status']=='complete' and facts['labelled_count']==5 and facts['failed_excluded_count']==1
    assert updated['phase_diagrams']['mlip']['version']=='old-hull'
    action = {'tool':'update_mlip','operation':'prepare_mlip_training',
              'parameters':{'action':'RETRAIN_MLIP','prepare_inputs_only':True}}
    assert previously_approved_recovery_action(updated, action)
    assert state['parent_first_recovery_batches']['batch']['status']=='waiting_dft_results'


def test_twenty_percent_failure_stops_before_training(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch, 5)
    updated = advance_parent_first_recovery(state)
    assert updated['extrapolation_recovery_cycle']['recovery_route'] == 'parent_first_perturbation'
    assert updated['parent_first_recovery_batches']['batch']['status']=='waiting_dft_results'
    assert recovery_label_facts(updated)['status'] == 'waiting'


def test_parent_first_replaces_only_a_settled_prior_recovery(tmp_path, monkeypatch):
    from phase_agent.agents.reliability import extrapolation_round
    state = make_state(tmp_path, monkeypatch, 6)
    baseline = state['model_refresh']['old_diagram']
    state['extrapolation_recovery_cycle'] = {'scope_id':'epoch_refresh','invalid_model_version':'bad',
        'status':'awaiting_reviewed_dft_labels','baseline_diagram':baseline,
        'selected':[{'structure_sha256':'old'}]}
    monkeypatch.setattr(extrapolation_round, 'recovery_dft_failure_settled', lambda *args: False)
    import pytest
    with pytest.raises(ValueError, match='prior_dft_not_settled'):
        advance_parent_first_recovery(state)
    monkeypatch.setattr(extrapolation_round, 'recovery_dft_failure_settled', lambda *args: True)
    updated = advance_parent_first_recovery(state)
    assert updated['extrapolation_recovery_history'][-1]['status']=='superseded_by_parent_first_perturbation'
    assert updated['extrapolation_recovery_cycle']['baseline_hull_version']=='old-hull'

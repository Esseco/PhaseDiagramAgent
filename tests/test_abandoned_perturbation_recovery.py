import hashlib
import json
from pathlib import Path

import pytest
from phase_agent.policies.file_approval import proposal_hash
from phase_agent.tools.state.execution_receipts import begin_execution, record_execution_return, recovery_report
from phase_agent.tools.state.execution_reconciliation import propose_reconciliation, apply_reconciliation
from phase_agent.tools.step_runner.file_protocol import write_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    state_path=tmp_path/'workflow_state'/'state.json'
    params={'mode':'perturbation_recovery_inputs','confirmed':True,'scope_id':'scope',
            'round_evidence_digest':'digest','parent_record_ids':['r1'],'max_det_H':2,
            'frames_per_parent':2,'seed':42,'min_distance_A':1.0,'volume_ratio':[0.75,1.25]}
    action={'tool':'prepare_local_batch_files','operation':'prepare_perturbation_recovery',
            'parameters':params,'task_key':'key'}
    identity={'tool':action['tool'],'invocation_id':'old','config_version':'c','action_hash':proposal_hash(action)}
    parent=tmp_path/'parent.vasp';parent.write_text('parent',encoding='utf8')
    key=hashlib.sha256(json.dumps({'approval':'old','parameters':params},sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    directory=tmp_path/'submissions'/'perturbation_inputs'/key;directory.mkdir(parents=True)
    frame=directory/'frame.vasp';frame.write_text('frame',encoding='utf8')
    manifest={'schema':'perturbation-recovery-inputs-v1','status':'structures_prepared','submitted':False,
        'scope_id':'scope','round_evidence_digest':'digest','failed_model_version':'m',
        'parameters':{'frames_per_parent':2,'seed':42,'max_det_H':2,'min_distance_A':1.0,'volume_ratio':[0.75,1.25]},
        'parents':[{'record_id':'r1','structure_path':str(parent),'structure_sha256':sha(parent)}],
        'frames':[{'path':'frame.vasp','sha256':sha(frame),'parent_record_id':'r1','parent_sha256':sha(parent)}]}
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    record={'record_id':'old','status':'failed','final_action':action,'human_feedback':{'decision':'approve'},
        'execution_result':{'status':'failed','error':'ValueError: perturbation_upload_incomplete',
                            'operation_outcome':{'model_version':'m'}}}
    state={'confirmed_config':{'storage':{'workspace_root':str(tmp_path),'paths':{'upload_batches':'submissions'}}},
           'action_records':[record],'tasks':[],'slurm_batches':[],'reserved_relative_cost':0,
           'pending_execution_policies':{},'invocations':{}}
    write_json(state_path,state)
    assert begin_execution(state_path,identity,action=action)['allowed']
    record_execution_return(state_path,identity,'failed')
    return state_path,state,identity,frame,directory


def test_abandoned_staging_reconciles_without_deleting_or_reusing(tmp_path):
    path,state,identity,frame,directory=fixture(tmp_path)
    attestation={'evidence_note':'用户改为先弛豫，保留旧帧但不使用', 'external_jobs':'not_applicable',
                 'output_inventory':'abandoned_local_staging','external_cost':0}
    with pytest.raises(ValueError,match='not_no_effect'):
        propose_reconciliation(path,state,'old','verified_no_effect',
            {**attestation,'output_inventory':'checked_no_outputs'})
    proposal=propose_reconciliation(path,state,'old','verified_abandoned_staging',attestation)
    evidence=proposal['state']['pending_execution_recoveries'][proposal['plan_id']]['proposal']['checks']['abandoned_staging']
    assert evidence['frame_count']==1 and evidence['submitted'] is False
    from phase_agent.runtime.review_requests import additional_reviews
    card = next(row['review_card'] for row in additional_reviews(proposal['state'])
                if row['plan_id'] == proposal['plan_id'])
    assert card['title'] == '保留并弃用旧暂存帧'
    assert '1 个母帧，1 个旧暂存帧' in card['scope']
    assert '旧帧不用于后续流程' in card['approval_boundary']
    result=apply_reconciliation(path,proposal['state'],proposal['plan_id'],'confirm_sensitive')
    assert result['status']=='reconciled_abandoned_staging' and recovery_report(path,result['state'])['status']=='clear'
    assert frame.is_file() and (directory/'manifest.json').is_file()
    assert not begin_execution(path,identity)['allowed']


def test_abandoned_staging_changed_after_proposal_is_rejected(tmp_path):
    path,state,_,frame,_=fixture(tmp_path)
    attestation={'evidence_note':'保留且弃用旧帧','external_jobs':'not_applicable',
                 'output_inventory':'abandoned_local_staging','external_cost':0}
    proposal=propose_reconciliation(path,state,'old','verified_abandoned_staging',attestation)
    frame.write_text('altered',encoding='utf8')
    with pytest.raises(ValueError,match='frame_invalid'):
        apply_reconciliation(path,proposal['state'],proposal['plan_id'],'confirm_sensitive')


def test_abandoned_staging_blocks_unregistered_task_input(tmp_path):
    path,state,identity,_,directory=fixture(tmp_path)
    task=tmp_path/'submissions'/'epoch1'/'calculations'/'x'/'inputs'/'batch'/'00000'/'task.json'
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({'task_key':'perturbation:'+directory.name+':abc'}),encoding='utf8')
    attestation={'evidence_note':'保留且弃用旧帧','external_jobs':'not_applicable',
                 'output_inventory':'abandoned_local_staging','external_cost':0}
    with pytest.raises(ValueError,match='unregistered_task_inputs'):
        propose_reconciliation(path,state,'old','verified_abandoned_staging',attestation)


def test_abandoned_staging_rejects_mismatched_receipt_identity(tmp_path):
    from phase_agent.tools.state.abandoned_perturbation import inspect_abandoned_perturbation
    _,state,identity,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='action_identity_invalid'):
        inspect_abandoned_perturbation(state,{**identity,'action_hash':'sha256:'+'0'*64})

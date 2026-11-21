import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.tools.remote.integrity import payload_checksum
from phase_agent.tools.state.abandoned_exact_relax import inspect_abandoned_exact_relax
from phase_agent.tools.state.execution_reconciliation import _validate_checks


def make_case(tmp_path, monkeypatch):
    state_path=tmp_path/'workflow_state'/'state.json'
    state_path.parent.mkdir()
    directory=tmp_path/'submissions'/'epoch2'/'inputs'/'batch'
    task_dir=directory/'00000-RELAX-a';task_dir.mkdir(parents=True)
    frame=task_dir/'initial.vasp';frame.write_bytes(b'frame')
    digest=hashlib.sha256(b'frame').hexdigest()
    hashes={'c':digest,'other':'b'*64}
    identity={'invocation_id':'p','tool':'prepare_local_batch_files','config_version':'v','action_hash':'h'}
    action={'operation':'prepare_mlip_relax','target_ids':['c','other'],'parameters':{'mode':'relax_inputs',
            'selection_scope':'candidate_ids','candidate_frame_sha256':hashes,'model_version':'m'}}
    task={'task_id':'t','candidate_id':'c','parent_decision_id':'p','status':'pending',
          'input_structure_sha256':digest,'model_version':'m','config_version':'v','result_path':'../../results/t/result.json'}
    task['task_checksum']=payload_checksum(task)
    (task_dir/'task.json').write_text(json.dumps(task),encoding='utf8')
    manifest=[{'task_id':'t','input_path':'00000-RELAX-a/task.json'}]
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    snapshot={'manifest_checksum':payload_checksum(manifest),'model_version':'m','config_version':'v','task_ids':['t'],'batch_id':'b'}
    (directory/'batch.snapshot.json').write_text(json.dumps(snapshot),encoding='utf8')
    events=[{'event':'claimed','payload':{'identity':identity,'action':action}},
            {'event':'tool_returned','payload':{'identity':identity,'outcome':'failed','evidence':{'error':'MemoryError: '}}}]
    monkeypatch.setattr('phase_agent.tools.state.abandoned_exact_relax.execution_events',lambda *_:events)
    state={'exact_candidate_relax_authorization':{'candidate_frame_sha256':hashes},'active_model_version':'m',
           'confirmed_config_version':'v','tasks':[],'slurm_batches':[]}
    issue={'identity':identity,'phase':'returned','tool_outcome':'failed'}
    return state_path,state,issue,directory,frame


def test_unregistered_staging_is_bound_and_preserved(tmp_path,monkeypatch):
    path,state,issue,directory,frame=make_case(tmp_path,monkeypatch)
    evidence=inspect_abandoned_exact_relax(path,state,issue,directory)
    assert evidence['missing_candidate_ids']==['other'] and evidence['submitted'] is False
    checks={'abandoned_staging':evidence,'task_count':0,'unregistered_returned_task_ids':[],
            'phase_record_ids':[],'budget_reservations':[]}
    _validate_checks(checks,'verified_abandoned_staging',{'external_jobs':'not_applicable',
                    'output_inventory':'abandoned_local_staging','external_cost':0,'evidence_note':'verified'})
    frame.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='task_binding_invalid'):
        inspect_abandoned_exact_relax(path,state,issue,directory)


def test_registered_batch_is_not_abandoned(tmp_path,monkeypatch):
    path,state,issue,directory,_=make_case(tmp_path,monkeypatch)
    state['slurm_batches']=[{'batch_id':'b'}]
    with pytest.raises(ValueError,match='batch_already_registered'):
        inspect_abandoned_exact_relax(path,state,issue,directory)

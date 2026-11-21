import hashlib
import json
from pathlib import Path
import pytest
from phase_agent.tools.remote.integrity import payload_checksum
from phase_agent.tools.remote.transfer_bundle import create_bundle
from phase_agent.tools.workflows.diagnostic_retry_import import register_diagnostic_retry


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture(tmp_path):
    parent=tmp_path/'parent.vasp'; parent.write_bytes(b'parent-frame')
    state={'active_model_version':'model','confirmed_config_version':'cfg','tasks':[{'task_id':'DFT-a','status':'failed',
       'task_key':'orig','stage':'dft_single_point','structure_id':'S-a','structure_path':str(parent),
       'input_structure_sha256':_sha(parent),'config_version':'cfg','model_version':'model'}]}
    src=tmp_path/'src'; folder=src/'inputs'/'batch'/'00000-DFT-a'; folder.mkdir(parents=True)
    task={'task_id':'RETRY-a','task_key':'orig:retry','retry_of':'DFT-a','status':'pending',
       'stage':'dft_single_point','structure_id':'S-a','structure_path':'initial_structure.vasp',
       'input_structure_sha256':_sha(parent),'config_version':'cfg','model_version':'model','batch_id':'test',
       'protocol_version':1,'input_file_version':'v1','parameters':{'incar_settings':{'static':{'NELM':240}}}}
    task['task_checksum']=payload_checksum(task)
    (folder/'task.json').write_text(json.dumps(task)); (folder/'initial_structure.vasp').write_bytes(parent.read_bytes())
    (folder/'workflow.json').write_text(json.dumps({'incar_settings':{'static':{'NELM':240}}}))
    (folder/'submit_gpu.sh').write_text('#!/bin/bash\n')
    (src/'submission_plan.json').write_text(json.dumps({'scripts':['inputs/batch/00000-DFT-a/submit_gpu.sh']}))
    inp=tmp_path/'inputs.zip'; create_bundle(src,[p for p in src.rglob('*') if p.is_file()],inp,kind='inputs')
    out=tmp_path/'out'/'results'/'00000-DFT-a';out.mkdir(parents=True)
    result={k:task[k] for k in ('task_id','task_key','batch_id','config_version','model_version','protocol_version','input_file_version','stage')}
    result.update(status='completed',converged=True)
    result_path=out/'result.json'; result_path.write_text(json.dumps(result))
    marker={k:task[k] for k in ('task_id','task_key','batch_id','config_version','model_version','protocol_version','input_file_version','task_checksum')}
    marker.update(status='completed',result_checksum=_sha(result_path))
    (out/'task.finished.json').write_text(json.dumps(marker)); (out/'training.json').write_text('[{}]')
    archive=tmp_path/'results.zip'; create_bundle(tmp_path/'out',[p for p in (tmp_path/'out').rglob('*') if p.is_file()],archive,kind='results',binding={'input_sha256':_sha(inp)})
    return state,inp,archive,parent


def test_retry_registration_binds_prior_failure_and_is_idempotent(tmp_path):
    state,inp,result,parent=fixture(tmp_path)
    registered=register_diagnostic_retry(state,inp,result,tmp_path/'evidence')
    assert registered['status']=='registered' and registered['task_count']==1
    assert len(state['tasks'])==1 and registered['state']['tasks'][-1]['retry_of']=='DFT-a'
    assert register_diagnostic_retry(registered['state'],inp,result,tmp_path/'evidence')['status']=='already_registered'
    parent.write_bytes(b'changed')
    with pytest.raises(ValueError,match='parent_structure_changed'):
        register_diagnostic_retry(state,inp,result,tmp_path/'other')


def test_retry_registration_rejects_completed_parent(tmp_path):
    state,inp,result,parent=fixture(tmp_path); state['tasks'][0]['status']='completed'
    with pytest.raises(ValueError,match='task_binding_invalid'):
        register_diagnostic_retry(state,inp,result,tmp_path/'evidence')
    assert not (tmp_path/'evidence').exists()


def test_agent_control_commits_once_and_rejects_changed_parent(tmp_path):
    import threading
    from types import SimpleNamespace
    from phase_agent.runtime.diagnostic_retry_control import register
    state,inp,result,parent=fixture(tmp_path)
    project=tmp_path/'project'; state_dir=project/'workflow_state'; state_dir.mkdir(parents=True)
    input_dir=project/'submissions'/'epoch1'/'inputs'; input_dir.mkdir(parents=True)
    state['parent_first_recovery_batches']={'batch':{'input_dir':str(input_dir)}}
    state_path=state_dir/'state.json'; state_path.write_text(json.dumps(state))
    control=SimpleNamespace(state_path=str(state_path),chat_handler=SimpleNamespace(lock=threading.RLock()),_codex_context_cache={})
    first=register(control,inp,result)
    assert first['status']=='registered' and first['task_count']==1
    saved=json.loads(state_path.read_text()); assert len(saved['tasks'])==2
    second=register(control,inp,result)
    assert second['status']=='already_registered' and len(json.loads(state_path.read_text())['tasks'])==2

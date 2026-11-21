"""Synthetic transport round trip; no scientific project or HPC is modified."""
import hashlib
import json
import zipfile
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor

from phase_agent.science.mlip.parent_first_sampling import run_parent_first_sampling
from phase_agent.tools.remote.transfer_bundle import create_bundle, import_bundle
from phase_agent.tools.workflows.parent_first_handoff import prepare_parent_first_bundle
from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_minimal_bundle_remote_sampling_and_checked_return(tmp_path, monkeypatch):
    import phase_agent.agents.reliability.extrapolation_round as reliability
    import phase_agent.evidence.perturbation_recovery as recovery
    import phase_agent.tools.workflows.comparison_model_registry as registry
    import phase_agent.tools.workflows.direct_extrapolation_sampling as direct
    atoms=Atoms('NaCoNiO2', positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0],[0,0,2]],
                cell=[6,6,6], pbc=True)
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    h = [[1,0,0],[0,2,0],[0,0,1]]
    manager = PhaseDataManager({'P':['O3'], 'H':{'O3':[h]}, 'TM_ratio':{'Ni':1,'Co':1}})
    branch = manager.add_branch(P='O3', H=h, x=0.5, T=['Ni','Co'],
                                composition={'Na':1,'Ni':1,'Co':1,'O':2})
    parent=tmp_path/'parent.vasp';write(parent,atoms,format='vasp')
    model=tmp_path/'model';model.write_bytes(b'model')
    row={'record_id':'r1','branch_id':branch,'phase':'O3','H':[[1,0,0],[0,2,0],[0,0,1]],
         'x_Na_per_O2':0.5,'structure_path':str(parent),'structure_sha256':sha(parent)}
    monkeypatch.setattr(reliability,'required_direct_round',lambda state:{'scope_id':'scope','evidence_digest':'digest'})
    monkeypatch.setattr(recovery,'build_perturbation_recovery_pool',lambda *a,**k:{'hull_version':'hull','candidates':[row]})
    monkeypatch.setattr(registry,'original_round_model',lambda *a:({'framework':'mace','mace_head':'omat_pbe',
        'model_path':'/remote/model'},sha(model),None))
    monkeypatch.setattr(direct,'successful_dft_references',lambda state:[{'task_id':'DFT-1',
        'structure':AseAtomsAdaptor.get_structure(atoms).as_dict()}])
    params={'confirmed':True,'scope_id':'scope','round_evidence_digest':'digest',
            'parent_record_ids':['r1'],'max_det_H':3,'frames_per_parent':2,'direct_target':1,
            'seed':42, 'min_distance_A':1.0, 'volume_ratio':[0.75,1.25]}
    dft_settings = {'config_version':'config-1','model_version':'failed-model',
                    'approval_record_id':'approved-1','parameters':{}}
    bundle=prepare_parent_first_bundle({}, {}, params, tmp_path/'handoff',
        feature_model_path='/remote/m3gnet',feature_model_sha256='b'*64, dft_settings=dft_settings)
    from phase_agent.tools.local.prepare_local_batch_files import prepare_local_batch_files
    action_params = {**params, 'mode':'parent_first_recovery_inputs', 'dft_parameters':{},
                     'feature_model_path':'/remote/m3gnet', 'feature_model_sha256':'b'*64}
    analysis = tmp_path/'analysis_outputs'
    latest = analysis/'latest'/'phase_diagram'; latest.mkdir(parents=True)
    (latest/'manifest.json').write_text(json.dumps({'required_files':[],'files':{},
        'identity':{'model_version':'old-model','hull_version':'old-hull'}}), encoding='utf8')
    state = {'confirmed_config_version':'config-1', 'active_model_version':'failed-model',
             'analysis_output_directory':str(analysis),
             'upload_layout':{'model_rounds':{'failed-model':2}},
             'model_refresh':{'new_model_version':'failed-model',
                 'old_diagram':{'status':'completed','version':'old-hull','model_version':'old-model'}},
             'phase_diagrams':{'mlip':{'status':'completed','version':'old-hull','model_version':'old-model'}}}
    submission = prepare_local_batch_files(action={'operation':'prepare_parent_first_recovery',
        'parameters':action_params}, context={'event_state':state,'manager':manager,
        'effective_config':{'upload_batches_directory':str(tmp_path/'submissions'), 'dft':{'parameters':{}}},
        'config_version':'config-1','approval_record_id':'approved-1'})
    assert submission['status']=='prepared' and submission['submitted'] is False
    assert len(submission['state']['parent_first_recovery_batches']) == 1
    assert submission['handoff']['verification']['status'] == 'passed'
    with zipfile.ZipFile(submission['handoff']['upload_archive']) as initial_zip:
        initial_names = initial_zip.namelist()
        assert any(name.endswith('/input_manifest.json') for name in initial_names)
        assert any(name.endswith('/parents/0000.vasp') for name in initial_names)
        assert any(name.endswith('/references/0000.vasp') for name in initial_names)
        assert not any('selected_' in name for name in initial_names)
        script_name = next(name for name in initial_names if name.endswith('/GPU.sh'))
        gpu_script = initial_zip.read(script_name).decode('utf8')
        assert 'export PYTHONPATH="/data/home/lichaoyue/PhaseDiagramAgent_repo:/data/home/lichaoyue/Process_code:${PYTHONPATH:-}"' in gpu_script
        assert gpu_script.index('export PYTHONPATH=') < gpu_script.index('python3 -m phase_agent.science.mlip.parent_first_pipeline')
    remote=tmp_path/'remote_input'; remote.mkdir()
    from phase_agent.tools.remote.input_validation import _inventory
    with zipfile.ZipFile(bundle['archive']) as archive:
        rows=_inventory(archive)
        for entry in rows:
            target=remote/entry['path']; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(entry['path']))
            assert sha(target)==entry['sha256']
    manifest=remote/'input_manifest.json';data=json.loads(manifest.read_text(encoding='utf8'))
    assert data['target']==1 and len(data['parents'])==1 and len(data['references'])==1
    binding={k:data[k] for k in ('scope_id','round_evidence_digest','baseline_hull_version')}
    binding['input_manifest_sha256']=sha(manifest)
    parents=[{**x,'path':str(remote/x['path'])} for x in data['parents']]
    refs=[{**x,'path':str(remote/x['path'])} for x in data['references']]
    paths=[]
    def relax(job):
        path=Path(job['output_directory'])/'initial_relaxed.vasp';path.parent.mkdir()
        write(path,read(job['structure_path']),format='vasp')
        return {'status':'completed','optimizer_converged':True,'structure_path':str(path),
                'structure_checksum':sha(path),'extrapolation_monitor':{'extrapolation_failure':False,
                'failures':[],'final_feature_difference':0.1}}
    def extract(inputs, encoder, cache):
        paths.extend(inputs);_fake_features(inputs, cache)
    def sample(cache, output, target, reference_count):
        assert reference_count==1 and target==1
        output.mkdir();(output/'selected_indices.json').write_text(json.dumps([{'source':paths[2],'frame':0}]),encoding='utf8')
        return {'n_samples':1}
    class Encoder: metadata={'model_sha256':'b'*64}
    selected=tmp_path/'remote_results'/'selected'
    run_parent_first_sampling(parents,refs,selected,model_path=model,model_sha256=sha(model),
        feature_model_path='unused',feature_model_sha256='b'*64,target=1,frames_per_parent=2,binding=binding,
        checkpoint_dir=tmp_path/'remote_work'/'parents',relax_fn=relax,encoder_factory=lambda _:Encoder(),
        extract_fn=extract,sample_fn=sample)
    result_zip=tmp_path/'result.zip'
    create_bundle(selected,[x for x in selected.iterdir() if x.is_file()],result_zip,kind='results')
    returned=tmp_path/'returned'; import_bundle(result_zip,returned)
    checked=validate_parent_first_result(remote,returned)
    assert checked['status']=='verified' and checked['target']==1
    assert sorted(p.name for p in returned.iterdir()) == ['result.json','selected_0000.vasp']
    from phase_agent.science.mlip.parent_first_pipeline import run_parent_first_pipeline
    operation = tmp_path/'epoch1'/'calculations'/'parent-first-dft'
    dft_dir = operation/'inputs'/'batch'
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic')
    def fake_dispatch(directory):
        directory = Path(directory)
        with zipfile.ZipFile(directory/'inputs.zip') as zipped:
            scripts = json.loads(zipped.read('submission_plan.json'))['scripts']
        (directory/'.submitted_jobs.json').write_text(json.dumps({
            'archive_sha256':sha(directory/'inputs.zip'),
            'jobs':{script:{'status':'submitted','job_id':'123'} for script in scripts}}), encoding='utf8')
    pipeline = run_parent_first_pipeline(remote, returned, tmp_path/'remote_work'/'parents', operation,
                                         dispatch=fake_dispatch)
    assert pipeline['status']=='submitted'
    dft = json.loads((dft_dir/'dft_manifest.json').read_text(encoding='utf8'))
    task_dir = dft_dir/dft['tasks'][0]['directory']
    task = json.loads((task_dir/'task.json').read_text(encoding='utf8'))
    assert dft['task_count'] == 1 and task['input_structure_sha256'] == checked['selected'][0]['sha256']
    assert json.loads((task_dir/'workflow.json').read_text(encoding='utf8'))['calculation'] == 'static'
    dispatch = Path(pipeline['dispatch_directory'])
    handoff = {'upload_archive':str(dispatch/'inputs.zip')}
    package = Path(pipeline['provenance_archive'])
    with zipfile.ZipFile(package) as zipped:
        names = set(zipped.namelist())
        assert 'selection/result.json' in names and 'selection/selected_0000.vasp' in names
        assert 'dft/submission_receipt.json' in names and 'dft/dft_manifest.json' in names
        assert 'dft/'+dft['tasks'][0]['directory']+'/task.json' in names
        assert len(names) == 6  # Includes the transfer manifest, but no logs or trajectories.
    from phase_agent.tools.workflows.parent_first_return_import import import_parent_first_return
    registered_state = submission['state']
    registered_state['action_records'] = [{'record_id':'approved-1','status':'prepared',
        'human_feedback':{'decision':'approve'},'final_action':{'tool':'prepare_local_batch_files',
        'operation':'prepare_parent_first_recovery','parameters':action_params}}]
    batch_id = next(iter(registered_state['parent_first_recovery_batches']))
    imported = import_parent_first_return(registered_state, manager, batch_id, package, tmp_path/'evidence')
    assert imported['status'] == 'registered' and len(imported['state']['tasks']) == 1
    assert not manager.data['structures'] and len(imported['manager'].data['structures']) == 1
    assert imported['state']['parent_first_recovery_batches'][batch_id]['status'] == 'waiting_dft_results'
    # A failed DFT is recovered through the existing identity checker and trips
    # the existing failure threshold without repeating the calculation.
    result_root = tmp_path/'remote-dft-return'
    result_folder = result_root/'results'/dft['tasks'][0]['directory']
    result_folder.mkdir(parents=True)
    dft_task = json.loads((task_dir/'task.json').read_text(encoding='utf8'))
    keys = ('task_id','task_key','batch_id','config_version','model_version','protocol_version','input_file_version')
    dft_result = {**{key:dft_task[key] for key in keys}, 'task_checksum':dft_task['task_checksum'],
                  'structure_id':dft_task['structure_id'], 'stage':'dft_single_point',
                  'status':'failed', 'error':'synthetic failure', 'outputs':{}}
    result_file = result_folder/'result.json'
    result_file.write_text(json.dumps(dft_result),encoding='utf8')
    marker = {**{key:dft_task[key] for key in keys}, 'task_checksum':dft_task['task_checksum'],
              'status':'failed', 'result_checksum':sha(result_file)}
    (result_folder/'task.finished.json').write_text(json.dumps(marker),encoding='utf8')
    dft_zip = tmp_path/'dft-results.zip'
    create_bundle(result_root, list(result_folder.iterdir()), dft_zip, kind='results',
                  binding={'batch':'dynamic', 'input_sha256':sha(handoff['upload_archive'])})
    received = import_parent_first_return(imported['state'], manager, batch_id, package, tmp_path/'evidence',
                                          dft_result_archives=[dft_zip])
    assert received['status']=='already_registered'
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
    from phase_agent.evidence.dft_failure_policy import failure_guard_facts
    collection = RemoteBatchRunner(tmp_path/'unused', worker_command=[]).collect_results_with_report(received['state'])
    assert len(collection['results']) == 1 and collection['results'][0]['status'] == 'failed'
    reconciled = reconcile_task_results(received['state'], collection['results'])['state']
    assert failure_guard_facts(reconciled)['status'] == 'awaiting_analysis'
    # The Agent commits new structure and task identities in one transaction,
    # then the normal collector reads the same verified return files.
    from shutil import copyfile
    from phase_agent.runtime.parent_first_recovery import recover_available_parent_first
    state_path = tmp_path/'workflow_state'/'state.json'
    state_path.parent.mkdir()
    ledger_path = state_path.parent/'ledgers'/'phase.json'
    manager.save(ledger_path)
    state_path.write_text(json.dumps(registered_state), encoding='utf8')
    return_dir = Path(submission['handoff']['return_directory'])
    copyfile(package, return_dir/'parent-first-provenance.results.zip')
    copyfile(dft_zip, return_dir/'dft-results.results.zip')
    committed = recover_available_parent_first(registered_state, manager=manager, state_path=state_path)
    assert len(committed['tasks']) == 1 and len(manager.data['structures']) == 1
    assert json.loads(state_path.read_text(encoding='utf8'))['tasks'][0]['task_id'] == dft_task['task_id']
    normal = RemoteBatchRunner(tmp_path/'unused2', worker_command=[]).collect_results_with_report(committed)
    assert len(normal['results']) == 1 and normal['results'][0]['status'] == 'failed'


def _fake_features(inputs, cache):
    import h5py
    import numpy as np
    from ase.io import read
    indices = [(source, frame) for source, path in enumerate(inputs)
               for frame, _ in enumerate(read(path, index=':'))]
    with h5py.File(cache, 'x') as h:
        h.create_dataset('features', data=np.zeros((len(indices), 2), dtype='f4'), maxshape=(None, 2))
        h.create_dataset('indices', data=np.array(indices, dtype='i8'))


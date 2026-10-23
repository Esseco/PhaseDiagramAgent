"""Real old selections/labels, fake VASP parsing and MLIP inference in private copies."""
from pathlib import Path
from copy import deepcopy
import json
import runpy
import shutil
import zipfile

import pytest
from pymatgen.core import Structure

from tests.test_legacy_delivery_recovery import setup_original,frozen
from phase_agent.tools.remote.dft_reassessment_handoff import prepare_reassessment_handoff
from phase_agent.tools.remote import dft_result_reassessment as adapter
from phase_agent.tools.remote import predict_dft_final_frame as prediction
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.transfer_bundle import import_bundle
from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.analysis.feedback.dft_result_products import record_dft_products
from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment
from phase_agent.tools.workflows.create_dft_comparison_evaluator import create_dft_comparison_evaluator
from phase_agent.tools.workflows.create_workflow_handlers import _update_mlip
from Process_Vasp.workflows import atomate_runner

@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp('p')


@pytest.mark.parametrize('project,count,accepted',[
    ('O3-1ele-NaCoO2',35,30),('O3-1ele-NaCrO2',15,12)])
@pytest.mark.real_project
def test_actual_selection_reassessment_pair_import_and_training_inputs(tmp_path,monkeypatch,project,count,accepted):
    real=Path('E:/0-PhaseDiagram')/project
    state,original=setup_original(tmp_path,project)
    real_archive=real/'submissions/archives/01-upload/epoch0_DFTSP_Batch01_Recovery02_output/recovered.results.zip'
    real_diagnostics=real_archive.with_name('diagnostics.zip')
    derived=prepare_reassessment_handoff(state,original,source_directory='/fake/original',returned_archive=real_archive)
    assert derived['collect_only'] is False and derived['dft_recompute'] is False
    input_folder=Path(derived['folder'])
    assert {p.name for p in input_folder.iterdir()}=={'inputs.zip','submit_all.sh','recover_results.py'}
    script=(input_folder/'submit_all.sh').read_text()
    assert 'source /data/app/anaconda3/2024.10-1/bin/activate' in script
    assert 'conda activate mace' in script and 'python3 recover_results.py' in script
    source=tmp_path/'hpc'/Path(original['folder']).name
    source.mkdir(parents=True)
    shutil.copy2(original['upload_archive'],source/'inputs.zip')
    with zipfile.ZipFile(source/'inputs.zip') as z:
        for name in z.namelist():
            if name!='transfer_manifest.json':
                target=source/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(name))
    import_bundle(real_archive,source)
    with zipfile.ZipFile(real_diagnostics) as z:
        diagnostics=json.loads(z.read('recovery_status.json'))
    with zipfile.ZipFile(input_folder/'inputs.zip') as z:
        plan=json.loads(z.read('recovery_plan.json'))
        options=json.loads(z.read('repair_options.json'))
    by_calculation={}
    for task in plan['tasks']:
        tid=task['expected']['task_id']
        worker=source/task['input_directory']
        log=diagnostics['task_diagnostics'][tid]['files']
        for filename in ['workflow_state.json','workflow_status.json','failure_report.txt']:
            if filename in log:(worker/filename).write_text(log[filename]['text'],encoding='utf-8')
        checkpoint=json.loads((worker/'workflow_state.json').read_text())
        calculation=worker/checkpoint['stages']['static']['directory']
        calculation.mkdir(parents=True,exist_ok=True)
        for name in ['vasprun.xml','OUTCAR','CONTCAR']:(calculation/name).write_text('fake local raw output: parser is mocked')
        row=json.loads((source/task['result_directory']/'result.json').read_text())
        by_calculation[str(calculation.resolve())]=row
    original_before=frozen(source)
    remote=source.with_name(input_folder.name)
    shutil.copytree(input_folder,remote)
    ns=runpy.run_path(str(remote/'recover_results.py'))
    calls=[]
    def validator(directory,stage):
        assert stage=='static'
        return Structure.from_dict(by_calculation[str(directory.resolve())]['outputs']['structure'])
    def predict(structure,**kwargs):
        calls.append(kwargs)
        assert kwargs['operation']=='predict' and kwargs['parameters']['device']=='cuda'
        return {'status':'completed','energy':-2.0*len(structure),'forces':[[0.01,0,0]]*len(structure),'energy_unit':'eV','forces_unit':'eV/angstrom'}
    model_path=options['model']['model_path']
    old_is_file=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda p:True if str(p).replace('\\','/')==model_path else old_is_file(p))
    def checksum(path):
        return 'a'*64 if str(path).replace('\\','/')==model_path else file_checksum(path)
    monkeypatch.setattr(adapter,'file_checksum',checksum)
    monkeypatch.setattr(prediction,'file_checksum',checksum)
    monkeypatch.setattr(atomate_runner,'_validate',validator)
    monkeypatch.setattr(prediction,'run_mace_with_py_mace',predict)
    monkeypatch.setenv('PHASE_RECOVERY_SOURCE',str(source))
    monkeypatch.setenv('SLURM_JOB_ID','fake-no-hpc-submission')
    def forbidden(*a,**kw):raise AssertionError('VASP/Slurm/training subprocess forbidden')
    monkeypatch.setattr('subprocess.run',forbidden)
    monkeypatch.setattr('subprocess.Popen',forbidden)
    report=ns['collect'](remote,extra_payloads=ns['EXTRA_PAYLOADS'],transform_factory=ns['_reassessment_factory'])
    ns['diagnostics'](remote,report)
    assert report['status']=='complete',report['errors']
    assert len(calls)==accepted
    repeated=ns['collect'](remote,extra_payloads=ns['EXTRA_PAYLOADS'],transform_factory=ns['_reassessment_factory'])
    assert repeated['status']=='complete'
    assert len(calls)==accepted, 'Already completed predictions must be cached'
    assert frozen(source)==original_before
    # Return to local filesystem semantics before checking remote-only model references.
    monkeypatch.setattr(Path,'is_file',old_is_file)
    output=remote.with_name(remote.name[:-6]+'_output')
    shutil.copytree(output,derived['return_directory'],dirs_exist_ok=True)
    # Keep old returns present too: the new corrected copies must not overwrite them.
    legacy_output=Path(original['folder']).with_name('epoch0_DFTSP_Batch01_Recovery02_output')
    legacy_input=legacy_output.with_name('epoch0_DFTSP_Batch01_Recovery02_input')
    real_inputs=real_archive.parent.with_name(legacy_input.name)
    shutil.copytree(real_inputs,legacy_input)
    shutil.copytree(real_archive.parent,legacy_output)
    recovered=RemoteBatchRunner(tmp_path/'unused',worker_command=[]).collect_results_with_report(state)
    assert recovered['report']['transfer_errors']==[],recovered['report']
    assert len(recovered['results'])==count and recovered['report']['invalid_count']==0
    assert sum(r['status']=='completed' for r in recovered['results'])==accepted
    assert sum(bool(r['outputs'].get('remote_mlip_prediction')) for r in recovered['results'])==accepted
    state=reconcile_task_results(state,recovered['results'])['state']
    config=deepcopy(json.loads((real/'workflow_state/state.json').read_text(encoding='utf-8'))['confirmed_config'])
    config['upload_batches_directory']=str(tmp_path/'local/submissions')
    evaluator=create_dft_comparison_evaluator(config,state=state)
    for row in recovered['results']:record_dft_products(state,row,evaluator=evaluator)
    assessment=post_dft_assessment(state,config)
    assert assessment['status']=='evaluated',assessment
    assert len(state['new_dft_records'])==accepted
    assert sum(r['status']=='completed' for r in state['dft_mlip_comparisons'])==accepted
    training=_update_mlip(action={'parameters':{'prepare_inputs_only':True}},context={'event_state':state,'effective_config':config})
    assert training['status']=='awaiting_remote_training',training.get('reason')
    assert training['submitted'] is False
    evidence={'project':project,'recovered':count,'completed_pairs':accepted,'skipped_unconverged':count-accepted,
        'repeat_new_predictions':len(calls)-accepted,'original_remote_files_unchanged':True,
        'assessment_status':assessment['status'],'training_input_status':training['status'],
        'training_input':training['handoffs'][0]['folder'],'mock_predictions_only':True}
    (tmp_path/'end_to_end.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence),flush=True)

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
from ase.io import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_compute_pipeline_prepares_dispatch_and_minimal_return(tmp_path, monkeypatch):
    from phase_agent.science.mlip.parent_first_pipeline import run_parent_first_pipeline
    from phase_agent.tools.workflows import parent_first_result

    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic')
    root = tmp_path/'input'; root.mkdir()
    settings = {'config_version':'config-1','model_version':'bad','approval_record_id':'approved', 'parameters':{}}
    manifest = root/'input_manifest.json'
    manifest.write_text(json.dumps({'dft_settings':settings}),encoding='utf8')
    selected = tmp_path/'selection'; selected.mkdir()
    frame = selected/'selected_0000.vasp'
    write(frame, Atoms('NaCoO2',positions=[[0,0,0],[2,0,0],[0,2,0],[0,0,2]],cell=[6,6,6],pbc=True),format='vasp')
    (selected/'result.json').write_text('{}',encoding='utf8')
    checked = {'status':'verified','target':1,'result_sha256':sha(selected/'result.json'),
               'binding':{'input_manifest_sha256':sha(manifest),'scope_id':'scope',
                          'round_evidence_digest':'round','baseline_hull_version':'old'},
               'selected':[{'structure_path':str(frame),'structure_sha256':sha(frame),
                            'parent_record_id':'parent','branch_id':'branch'}]}
    monkeypatch.setattr(parent_first_result,'validate_parent_first_result',lambda *a:checked)
    def dispatch(directory):
        directory = Path(directory)
        with zipfile.ZipFile(directory/'inputs.zip') as archive:
            scripts = json.loads(archive.read('submission_plan.json'))['scripts']
        (directory/'.submitted_jobs.json').write_text(json.dumps({
            'archive_sha256':sha(directory/'inputs.zip'),
            'jobs':{name:{'status':'submitted','job_id':'123'} for name in scripts}}),encoding='utf8')
    outcome = run_parent_first_pipeline(root, selected, tmp_path/'checkpoints',
                                        tmp_path/'epoch1'/'calculation',dispatch=dispatch)
    assert outcome['status']=='submitted' and outcome['task_count']==1
    with zipfile.ZipFile(outcome['provenance_archive']) as archive:
        assert 'selection/selected_0000.vasp' in archive.namelist()
        assert 'dft/submission_receipt.json' in archive.namelist()

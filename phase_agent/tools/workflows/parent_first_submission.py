"""Prepare the existing dispatcher handoff for parent-first HPC sampling."""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


def prepare_parent_first_submission(state, ledger, parameters, target, *, feature_model_path,
                                    feature_model_sha256, dft_settings):
    """Produce one input archive and dispatcher entry; never submit a job."""
    from phase_agent.tools.remote.input_validation import _inventory
    from phase_agent.tools.remote.submission_handoff import prepare_handoff
    from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script
    from phase_agent.tools.workflows.parent_first_handoff import prepare_parent_first_bundle

    target = Path(target)
    if target.exists():
        raise FileExistsError('parent_first_submission_exists')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.parent-first-bundle-', dir=target.parent) as temp:
        source = prepare_parent_first_bundle(state, ledger, parameters, Path(temp)/'bundle',
            feature_model_path=feature_model_path, feature_model_sha256=feature_model_sha256,
            dft_settings=dft_settings)
        operation = target / 'calculation'
        slot = operation / 'inputs' / 'parent-first' / '00000'
        slot.mkdir(parents=True)
        with zipfile.ZipFile(source['archive']) as archive:
            rows = _inventory(archive)
            for row in rows:
                data = archive.read(row['path'])
                if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise ValueError('parent_first_submission_input_corrupt')
                destination = slot / row['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        manifest = json.loads((slot/'input_manifest.json').read_text(encoding='utf8'))
        task = {'task_id':'parent-first-'+hashlib.sha256((slot/'input_manifest.json').read_bytes()).hexdigest()[:16],
                'stage':'parent_first_sampling', 'worker_job':{
                    'model_path':manifest['model_path'], 'model_file_evidence':[{
                        'model_path':manifest['model_path'], 'sha256':manifest['model_sha256']}]}}
        (slot/'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf8')
        template = Path(__file__).parents[1] / 'remote' / 'mlip_gpu_template.sh'
        script = template.read_text(encoding='utf8').replace('__JOB_NAME__', 'parent-first')
        command = ('export PYTHONPATH="/data/home/lichaoyue/PhaseDiagramAgent_repo:/data/home/lichaoyue/Process_code:${PYTHONPATH:-}"\n'
                   'export TF_USE_LEGACY_KERAS=1\n'
                   'cd "${SLURM_SUBMIT_DIR:?}"\n'
                   'python3 -m phase_agent.science.mlip.parent_first_pipeline '
                   '--input-dir "$PWD" --results-dir "$PWD/work/selected" '
                   '--checkpoint-dir "$PWD/checkpoints" --operation-dir "$PWD/work/dft" '
                   '--status-path "$PWD/../../../results/status_summary.json"')
        if script.count('python3 run_mlip_task.py') != 1:
            raise ValueError('parent_first_submission_template_invalid')
        script = script.replace('python3 run_mlip_task.py', command)
        script_path = slot/'GPU.sh'
        write_unix_shell_script(script_path, script)
        handoff = prepare_handoff(operation, [script_path])
        if not handoff or handoff.get('job_count') != 1:
            raise ValueError('parent_first_submission_handoff_invalid')
        return {'status':'prepared', 'submitted':False,
                'handoff':handoff, 'parent_count':source['receipt']['parent_count'],
                'reference_count':source['receipt']['dft_reference_count'],
                'target':source['receipt']['target'], 'task_id':task['task_id'], 'input_dir':str(slot)}

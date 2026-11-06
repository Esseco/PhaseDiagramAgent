import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.science.mlip.parent_first_pipeline import run_parent_first_pipeline


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_pipeline_resumes_dispatch_without_repeating_sampling_or_preparation(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    source = tmp_path/'input'; source.mkdir()
    manifest = source/'input_manifest.json'
    manifest.write_text(json.dumps({'dft_settings': {'x': 1}}), encoding='utf8')
    result = tmp_path/'result'; checkpoint = tmp_path/'checkpoint'; operation = tmp_path/'operation'
    calls = []
    def sampling(args):
        calls.append('sampling'); result.mkdir(); (result/'result.json').write_text('result', encoding='utf8')
    def validate(input_dir, result_dir):
        return {'status':'verified', 'binding':{'input_manifest_sha256':sha(manifest)},
                'result_sha256':sha(result/'result.json')}
    def prepare(input_dir, result_dir, operation_dir):
        calls.append('prepare'); operation.mkdir(); dispatch_dir = tmp_path/'dispatch_input'; dispatch_dir.mkdir()
        archive = dispatch_dir/'inputs.zip'; archive.write_bytes(b'archive')
        return {'dft_manifest':str(operation/'manifest.json'), 'task_count':100,
                'handoff':{'upload_archive':str(archive)}}
    def dispatch(directory):
        calls.append('dispatch')
        if calls.count('dispatch') == 1:
            raise RuntimeError('uncertain submission')
    def publish(source, result_dir, dft_dir, dispatch_dir, output):
        calls.append('publish'); output.parent.mkdir(); output.write_bytes(b'provenance')
    options = dict(sampling=sampling, prepare=prepare, dispatch=dispatch, validate=validate, publish=publish)
    with pytest.raises(RuntimeError, match='uncertain submission'):
        run_parent_first_pipeline(source, result, checkpoint, operation, **options)
    output = run_parent_first_pipeline(source, result, checkpoint, operation, **options)
    assert output['status'] == 'submitted' and output['task_count'] == 100
    assert calls == ['sampling', 'prepare', 'dispatch', 'dispatch', 'publish']
    (tmp_path/'dispatch_input'/'inputs.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='dft_archive_changed'):
        run_parent_first_pipeline(source, result, checkpoint, operation, **options)
    assert calls == ['sampling', 'prepare', 'dispatch', 'dispatch', 'publish']


def test_pipeline_forbids_running_outside_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='requires_compute_job'):
        run_parent_first_pipeline(tmp_path, tmp_path/'out', tmp_path/'check', tmp_path/'operation')

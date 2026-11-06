"""Build the minimal parent-first provenance archive after DFT dispatch."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_parent_first_return(input_dir, result_dir, dft_dir, dispatch_dir, output_archive):
    """Export the selected frames and exact task identities, never calculation logs."""
    from phase_agent.tools.remote.integrity import payload_checksum
    from phase_agent.tools.remote.transfer_bundle import create_bundle
    from phase_agent.tools.workflows.parent_first_result import validate_parent_first_result

    input_root, result_root, task_root, dispatch_root = map(
        Path, (input_dir, result_dir, dft_dir, dispatch_dir))
    checked = validate_parent_first_result(input_root, result_root)
    manifest_path = task_root / 'dft_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    entries = manifest.get('tasks') or []
    if (manifest.get('schema') != 'parent-first-dft-prepared-v1'
            or manifest.get('binding') != checked['binding']
            or manifest.get('sampling_result_sha256') != checked['result_sha256']
            or len(entries) != checked['target'] or manifest.get('task_count') != len(entries)):
        raise ValueError('parent_first_return_dft_manifest_invalid')
    receipt_path = dispatch_root / '.submitted_jobs.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    archive = dispatch_root / 'inputs.zip'
    if receipt.get('archive_sha256') != _sha(archive):
        raise ValueError('parent_first_return_dispatch_binding_invalid')
    jobs = receipt.get('jobs') or {}
    selected = checked['selected']
    for row, entry in zip(selected, entries):
        directory = str(entry.get('directory') or '')
        if not directory or Path(directory).name != directory:
            raise ValueError('parent_first_return_task_identity_invalid')
        task = json.loads((task_root / directory / 'task.json').read_text(encoding='utf8'))
        checksum = task.pop('task_checksum', None)
        if (checksum != payload_checksum(task) or checksum != entry.get('task_checksum')
                or task.get('task_id') != entry.get('task_id')
                or task.get('input_structure_sha256') != row.get('structure_sha256')
                or entry.get('input_structure_sha256') != row.get('structure_sha256')):
            raise ValueError('parent_first_return_task_identity_invalid')
        matching = [v for key, v in jobs.items() if key.endswith('/' + directory + '/submit_gpu.sh')]
        if len(matching) != 1 or matching[0].get('status') != 'submitted':
            raise ValueError('parent_first_return_submission_unverified')
    destination = Path(output_archive)
    if destination.exists():
        raise FileExistsError('parent_first_return_archive_exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.parent-first-return-', dir=destination.parent) as temp:
        root = Path(temp)
        (root/'selection').mkdir()
        shutil.copyfile(result_root/'result.json', root/'selection'/'result.json')
        for row in selected:
            shutil.copyfile(row['structure_path'], root/'selection'/Path(row['structure_path']).name)
        (root/'dft').mkdir()
        shutil.copyfile(manifest_path, root/'dft'/'dft_manifest.json')
        for entry in entries:
            directory = root/'dft'/entry['directory']
            directory.mkdir()
            shutil.copyfile(task_root/entry['directory']/'task.json', directory/'task.json')
        shutil.copyfile(receipt_path, root/'dft'/'submission_receipt.json')
        files = [path for path in root.rglob('*') if path.is_file()]
        create_bundle(root, files, destination, kind='results', binding={
            'input_manifest_sha256':checked['binding']['input_manifest_sha256'],
            'dft_input_sha256':receipt['archive_sha256']})
    return {'status':'prepared', 'archive':str(destination), 'sha256':_sha(destination),
            'selected_count':len(selected), 'submitted_jobs':len(entries)}

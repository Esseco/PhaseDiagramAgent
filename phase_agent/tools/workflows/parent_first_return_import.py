"""Import only the bound parent-first selection and DFT result archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def import_parent_first_return(state, manager, batch_id, provenance_archive, evidence_dir,
                               *, dft_result_archives=()):
    """Prepare a state and ledger update; the Agent owns the atomic commit."""
    from phase_agent.tools.remote.transfer_bundle import import_bundle, validate_bundle
    from phase_agent.tools.workflows.parent_first_dft_registration import (
        register_parent_first_dft_tasks, stage_parent_first_structures,
    )

    batch = (state.get('parent_first_recovery_batches') or {}).get(batch_id) or {}
    input_dir = Path(batch.get('input_dir') or '')
    input_manifest = input_dir / 'input_manifest.json'
    if not batch or not input_manifest.is_file():
        raise ValueError('parent_first_import_batch_invalid')
    provenance = Path(provenance_archive)
    validate_bundle(provenance)
    with zipfile.ZipFile(provenance) as archive:
        transfer = json.loads(archive.read('transfer_manifest.json'))
        binding = transfer.get('batch_binding') or {}
        if binding.get('input_manifest_sha256') != _sha(input_manifest):
            raise ValueError('parent_first_import_input_binding_invalid')
        names = {row['path'] for row in transfer['files']}
        if not {'selection/result.json', 'dft/dft_manifest.json', 'dft/submission_receipt.json'}.issubset(names):
            raise ValueError('parent_first_import_provenance_incomplete')
        receipt = json.loads(archive.read('dft/submission_receipt.json'))
        if receipt.get('archive_sha256') != binding.get('dft_input_sha256'):
            raise ValueError('parent_first_import_dispatch_binding_invalid')
        dft_manifest = json.loads(archive.read('dft/dft_manifest.json'))
        entries = dft_manifest.get('tasks') or []
        expected_names = {'selection/result.json', 'dft/dft_manifest.json', 'dft/submission_receipt.json'}
        selected = json.loads(archive.read('selection/result.json')).get('selected') or []
        expected_names.update('selection/' + str(row.get('path') or '') for row in selected)
        jobs = receipt.get('jobs') or {}
        for entry in entries:
            folder = str(entry.get('directory') or '')
            if not folder or PurePosixPath(folder).name != folder:
                raise ValueError('parent_first_import_task_path_invalid')
            expected_names.add('dft/' + folder + '/task.json')
            matches = [job for path, job in jobs.items() if path.endswith('/' + folder + '/submit_gpu.sh')]
            if (len(matches) != 1 or matches[0].get('status') != 'submitted'
                    or not str(matches[0].get('job_id') or '').isdigit()):
                raise ValueError('parent_first_import_submission_invalid')
        if names != expected_names:
            raise ValueError('parent_first_import_unexpected_provenance')
    evidence = Path(evidence_dir)
    import_bundle(provenance, evidence)
    result_dir = evidence / 'dft_results' / 'results'
    allowed = {row['directory'] for row in entries}
    for result_archive in dft_result_archives:
        validate_bundle(result_archive)
        with zipfile.ZipFile(result_archive) as archive:
            transfer = json.loads(archive.read('transfer_manifest.json'))
            if (transfer.get('batch_binding') or {}).get('input_sha256') != receipt['archive_sha256']:
                raise ValueError('parent_first_import_result_binding_invalid')
            for row in transfer['files']:
                parts = PurePosixPath(row['path']).parts
                runtime = (len(parts) == 2 and parts[0] == 'results'
                           and parts[1].startswith('gpu-job-') and parts[1].endswith('.runtime.json'))
                if not runtime and (len(parts) != 3 or parts[0] != 'results' or parts[1] not in allowed):
                    raise ValueError('parent_first_import_result_scope_invalid')
        import_bundle(result_archive, evidence / 'dft_results')
    registered = register_parent_first_dft_tasks(state, input_dir, evidence/'selection', evidence/'dft',
                                                  results_dir=result_dir)
    shadow = stage_parent_first_structures(manager, registered['tasks'])
    from phase_agent.evidence.parent_first_recovery import advance_parent_first_recovery

    updated = advance_parent_first_recovery(registered['state'])
    return {'status':registered['status'], 'state':updated, 'manager':shadow,
            'tasks':registered['tasks'], 'provenance_sha256':_sha(provenance),
            'dft_input_sha256':receipt['archive_sha256'], 'submitted':False}

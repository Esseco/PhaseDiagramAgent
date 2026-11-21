"""Recover registered parent-first evidence through the Agent transaction."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile


def recover_available_parent_first(state, *, manager, state_path):
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery
    from phase_agent.tools.remote.transfer_bundle import checksum
    from phase_agent.tools.workflows.parent_first_return_import import import_parent_first_return

    current = state
    for batch_id, batch in (state.get('parent_first_recovery_batches') or {}).items():
        if batch.get('status') not in {'prepared','submitted','running','waiting_dft_results'}:
            continue
        directory = Path((batch.get('handoff') or {}).get('return_directory') or '')
        provenance = directory/'parent-first-provenance.results.zip'
        if not provenance.is_file():
            continue
        with zipfile.ZipFile(provenance) as archive:
            transfer = json.loads(archive.read('transfer_manifest.json'))
            expected = (transfer.get('batch_binding') or {}).get('dft_input_sha256')
        if not expected:
            raise ValueError('parent_first_recovery_provenance_unbound')
        result_archives = []
        for path in directory.glob('*.results.zip'):
            if path == provenance:
                continue
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read('transfer_manifest.json'))
                if (manifest.get('batch_binding') or {}).get('input_sha256') == expected:
                    result_archives.append(path)
        input_dir = Path(batch['input_dir']).resolve()
        epoch = next((p for p in input_dir.parents if p.name.startswith('epoch') and p.name[5:].isdigit()), None)
        if epoch is None or not isinstance(batch_id, str) or len(batch_id) != 64 or any(c not in '0123456789abcdef' for c in batch_id):
            raise ValueError('parent_first_recovery_evidence_location_invalid')
        state_file = Path(state_path).resolve()
        ledger_file = Path(getattr(manager, '_ledger_path', '')).resolve()
        state_hash = checksum(state_file)
        if json.loads(state_file.read_text(encoding='utf8')) != current:
            raise ValueError('parent_first_recovery_state_changed')
        if not ledger_file.is_file() or json.loads(ledger_file.read_text(encoding='utf8')) != manager.data:
            raise ValueError('parent_first_recovery_ledger_changed')
        ledger_hash = checksum(ledger_file)
        evidence = epoch/'evidence'/'verified_results'/('parent-first-'+batch_id[:16])
        imported = import_parent_first_return(current, manager, batch_id, provenance, evidence,
                                               dft_result_archives=result_archives)
        from filelock import FileLock
        from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
        from phase_agent.analysis.feedback.versioned_products import validate_bundle

        analysis_root = Path(current.get('analysis_output_directory') or '')
        cycle = imported['state'].get('extrapolation_recovery_cycle') or {}
        if not current.get('analysis_output_directory') or not analysis_root.is_dir():
            raise ValueError('parent_first_recovery_publication_root_missing')
        lock_dir = analysis_metadata_directory(analysis_root)
        lock_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_dir/'publication.lock'), timeout=30):
            latest = validate_bundle(analysis_root/'latest'/'phase_diagram')
            identity = latest.get('identity') or {}
            if (identity.get('model_version') != cycle.get('baseline_model_version')
                    or identity.get('hull_version') != cycle.get('baseline_hull_version')):
                raise ValueError('parent_first_recovery_latest_not_baseline')
            if imported['state'] != current or imported['manager'].data != manager.data:
                result = commit_result_recovery(state_file, imported['state'], {ledger_file:imported['manager'].data},
                    expected_hashes={str(state_file):state_hash, str(ledger_file):ledger_hash},
                    evidence={'operation':'parent_first_dft_registration','batch_id':batch_id,
                              'provenance_sha256':imported['provenance_sha256'],
                              'dft_input_sha256':imported['dft_input_sha256']})
                current = result['state']
                fresh = PhaseDataManager.load(ledger_file)
                manager.data = fresh.data
                for attr in ('_ledger_path','_ledger_identity','_ledger_data_digest'):
                    setattr(manager, attr, getattr(fresh, attr))
    return current

"""Use the existing dispatcher for DFT jobs created after HPC DIRECT selection."""
from __future__ import annotations

import json
from pathlib import Path


def prepare_parent_first_dft_handoff(input_dir, result_dir, operation_dir):
    """Prepare only; the existing dispatcher owns submission and replay checks."""
    from phase_agent.tools.remote.submission_handoff import prepare_handoff
    from phase_agent.tools.workflows.parent_first_dft import prepare_parent_first_dft

    operation = Path(operation_dir)
    settings = json.loads((Path(input_dir) / 'input_manifest.json').read_text(encoding='utf8')).get('dft_settings')
    directory = operation / 'inputs' / 'batch'
    prepared = prepare_parent_first_dft(input_dir, result_dir, directory, settings)
    scripts = [directory / row['directory'] / 'submit_gpu.sh' for row in prepared['tasks']]
    handoff = prepare_handoff(operation, scripts)
    if not handoff or handoff.get('job_count') != prepared['task_count']:
        raise ValueError('parent_first_dft_handoff_invalid')
    return {'status':'prepared', 'submitted':False, 'dft_manifest':str(directory/'dft_manifest.json'),
            'task_count':prepared['task_count'], 'handoff':handoff}

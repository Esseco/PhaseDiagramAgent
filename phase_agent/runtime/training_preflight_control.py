"""Agent-mediated, user-authorized regeneration of training preflight inputs."""

from copy import deepcopy
from pathlib import Path
import uuid

from filelock import FileLock


def regenerate_preflight(control, *, task_id, input_structure_sha256, model_version,
                         expected_job_key, independent_test_fraction, seed, user_instruction):
    if (not isinstance(user_instruction, str) or not user_instruction.strip()
        or not isinstance(task_id, str) or not task_id
        or not isinstance(model_version, str) or not model_version
        or not isinstance(input_structure_sha256, str)
        or len(input_structure_sha256) != 64
        or any(ch not in '0123456789abcdef' for ch in input_structure_sha256)
        or not isinstance(expected_job_key, str) or not expected_job_key
        or type(seed) is not int):
        raise ValueError('training_preflight_invalid_authorization_or_identity')
    from phase_agent.persistence.state.read_versions import read_identity
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.tools.workflows.lifecycle_support import _save_runner_state
    from phase_agent.configuration.runtime.build_effective_run_config import build_effective_run_config
    from phase_agent.runtime.project_context import current_config_session
    from phase_agent.analysis.convergence.training_provenance import model_with_training_provenance
    from phase_agent.science.training.training_preflight_selection import selected_records, split_independent_holdout
    from phase_agent.tools.local.regenerate_finetune_inputs import plan_finetune_regeneration, regenerate_finetune_inputs

    state_path = Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(state_path.parent / 'langgraph_lifecycle.sqlite') + '.lock', timeout=10):
        version = read_identity(state_path, strong=True)
        state = read_json(state_path, {}) or {}
        old = (state.get('remote_finetune_jobs') or {}).get(expected_job_key)
        if (not old or old.get('status') not in {'inputs_prepared', 'awaiting_remote_training'}
            or old.get('submitted') or old.get('activated')):
            raise ValueError('training_preflight_old_job_not_unsubmitted')
        active = state.get('active_model') or {}
        if old.get('original_model_version') != (active.get('version') or state.get('active_model_version')):
            raise ValueError('training_preflight_active_model_mismatch')
        policy = {'include_extrapolated_dft': [{'task_id': task_id,
                  'input_structure_sha256': input_structure_sha256, 'model_version': model_version,
                  'user_instruction': user_instruction}], 'independent_test_fraction': independent_test_fraction,
                  'seed': seed, 'version': 'user_scoped_holdout_v1'}
        current = deepcopy(state)
        current['training_preflight_policy'] = policy
        records = selected_records(current, policy)
        config = build_effective_run_config(current_config_session(control.chat_handler),
                                             control.chat_handler.workflow_kwargs.get('run_config') or {})
        training, test, holdout = split_independent_holdout(
            records, model_with_training_provenance(current, active), config.get('mlip_finetune') or {}, policy)
        from phase_agent.science.training.training_preflight_selection import group_equivalent_training_records
        from phase_agent.science.training.prepare_main_model import grouped_folds
        grouped, grouping = group_equivalent_training_records(training, config.get('mlip_finetune') or {})
        cv_settings = deepcopy(config.get('mlip_finetune') or {})
        cv_settings['group_keys'] = ['independent_group_id']
        grouped_folds(grouped, cv_settings)
        holdout['cv_grouping'] = grouping
        plan = plan_finetune_regeneration(current, config)
        if not plan['exists']:
            raise ValueError('training_preflight_old_inputs_missing')
        if plan['job_key'] != expected_job_key:
            raise ValueError('training_preflight_old_job_changed')
        result = regenerate_finetune_inputs(current, config, plan)
        new_state = result['state']
        new_state.pop('training_preflight_policy', None)
        history = new_state['finetune_regeneration_history'][-1]
        backup = Path(history['backup_directory'])
        directory = Path(history['directory'])
        try:
            _save_runner_state(new_state, str(state_path), expected_identity=version)
        except Exception as error:
            # A later save hook may fail after the state write succeeded. Never
            # roll files back unless the durable identity is still unchanged.
            if read_identity(state_path, strong=True) != version:
                raise RuntimeError('training_preflight_save_outcome_uncertain; inspect state and backup') from error
            if directory.exists():
                directory.rename(directory.with_name(directory.name + '.failed-save-' + uuid.uuid4().hex[:12]))
            if backup.is_dir():
                backup.rename(directory)
            raise
        return {'status':'prepared', 'old_job_key':expected_job_key, 'backup_directory':str(backup),
                'training_directory':str(directory), 'included_task_id':task_id,
                'training_count':len(training), 'test_count':len(test), 'holdout':holdout,
                'handoffs':result.get('handoffs') or [], 'submitted':False, 'activated':False}

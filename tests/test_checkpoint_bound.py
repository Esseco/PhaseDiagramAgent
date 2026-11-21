from phase_agent.agents.strategic.codex_submission_rejection import _checkpoint_project_bound


def test_compact_checkpoint_project_binding_requires_both_paths_and_versions(tmp_path):
    state_path = tmp_path / 'state.json'
    action = {'tool': 'generate_branches', 'scientific_decision': {'config_version': 'c', 'model_version': 'm'}}
    fields = {
        'branch_energy_pool_ledger_path': {'inline': str(tmp_path / 'ledgers' / 'branch_energy_pools.json')},
        'output_index_path': {'inline': str(tmp_path / 'analysis' / 'output_index.md')},
        'confirmed_config_version': {'inline': 'c'}, 'active_model_version': {'inline': 'm'},
    }
    frame = {'loaded_state': {'__phase_state_fields_v1__': fields}}
    assert _checkpoint_project_bound(frame, state_path, action)
    fields['active_model_version']['inline'] = 'other'
    assert not _checkpoint_project_bound(frame, state_path, action)
    fields['active_model_version']['inline'] = 'm'
    fields['output_index_path']['inline'] = str(tmp_path.parent / 'analysis' / 'output_index.md')
    assert not _checkpoint_project_bound(frame, state_path, action)
    frame['runtime_state_path'] = str(tmp_path / 'wrong.json')
    assert not _checkpoint_project_bound(frame, state_path, action)

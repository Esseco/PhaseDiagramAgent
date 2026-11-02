from phase_agent.tools.workflows.initial_tool_proposal import _post_dft_allowed_tools


def test_post_dft_preserves_registered_retry_without_opening_calculation_tools():
    allowed = ['select_dft_candidates', 'restart_failed_task', 'run_calculation_stage', 'prepare_local_batch_files']
    assert _post_dft_allowed_tools(allowed) == ['select_dft_candidates', 'restart_failed_task']


def test_post_dft_does_not_invent_retry_permission():
    assert _post_dft_allowed_tools(['select_dft_candidates']) == ['select_dft_candidates']


def test_first_search_retains_existing_preparation_exception():
    assert _post_dft_allowed_tools(['prepare_local_batch_files'], awaiting_first_search=True) == ['prepare_local_batch_files']

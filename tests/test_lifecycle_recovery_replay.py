from pathlib import Path


def test_superseded_reservations_do_not_fence_clean_current_request(tmp_path):
    from phase_agent.graphs.project.runner import _business_state_is_clean_for_stale_checkpoint

    state_path = Path(tmp_path) / "state.json"
    state_path.write_text(
        '{"confirmed_config_version":"config-new",'
        '"execution_recovery_report":{"status":"clear","unsettled":[]},'
        '"pending_execution_policies":{},"pending_execution_recoveries":{},'
        '"budget_reservations":{"old":{"config_version":"config-old",'
        '"status":"reserved"}},"tasks":[]}',
        encoding="utf-8",
    )

    assert _business_state_is_clean_for_stale_checkpoint(state_path)

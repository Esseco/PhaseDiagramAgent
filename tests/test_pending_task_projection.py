from phase_agent.tools.state.task_waiting import active_pending_tasks


def test_pending_projection_matches_result_wait_boundary():
    tasks = [{"task_id": status, "status": status} for status in
             ("pending", "running", "submitted", "unknown", "completed", "failed")]
    assert [t["task_id"] for t in active_pending_tasks(tasks)] == [
        "pending", "running", "submitted", "unknown"
    ]


def test_pending_projection_preserves_explicit_waivers():
    tasks = [
        {"status": "unknown", "stage": "dft_single_point", "recovery_wait_waived": True},
        {"status": "submitted", "model_refresh_id": "refresh", "refresh_wait_waived": True},
        {"status": "unknown", "stage": "relax_and_feature", "recovery_wait_waived": True},
    ]
    assert active_pending_tasks(tasks) == [tasks[2]]

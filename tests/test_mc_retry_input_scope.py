from phase_agent.tools.local.prepare_mc_upload_batches import prepare_mc_upload_batches


def test_frozen_retry_input_scope_rejects_extra_pending_tasks(tmp_path):
    state = {
        "dedup_gate": {"status": "ready"},
        "tasks": [
            {"task_id": "a", "stage": "deep_search", "status": "pending"},
            {"task_id": "b", "stage": "deep_search", "status": "pending"},
        ],
    }
    context = {
        "effective_config": {
            "upload_batches_directory": str(tmp_path),
            "supercomputer": {"worker": {"command": ["python"]}},
        },
        "event_state": state,
    }
    result = prepare_mc_upload_batches(
        action={"parameters": {"expected_task_ids": ["a"]}}, context=context
    )
    assert result["status"] == "not_configured" and result["reason"] == "mc_input_scope_changed"
    assert result["state"] == state


def test_frozen_retry_input_scope_rejects_duplicate_ids(tmp_path):
    state = {
        "dedup_gate": {"status": "ready"},
        "tasks": [{"task_id": "a", "stage": "deep_search", "status": "pending"}],
    }
    context = {
        "effective_config": {
            "upload_batches_directory": str(tmp_path),
            "supercomputer": {"worker": {"command": ["python"]}},
        },
        "event_state": state,
    }
    result = prepare_mc_upload_batches(
        action={"parameters": {"expected_task_ids": ["a", "a"]}}, context=context
    )
    assert result["status"] == "not_configured" and result["reason"] == "mc_input_scope_changed"


def test_preparing_existing_mc_inputs_does_not_create_an_extra_task():
    from phase_agent.policies.validate_tool_action import validate_tool_action

    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": "c",
            "config": {
                "agent": {"allowed_tools": ["prepare_local_batch_files"]},
                "budgets": {
                    "enforcement": "advisory",
                    "stage_limits": {"deep_search": {"max_tasks": 300}},
                },
                "frozen_parameters": [],
            },
        },
    }
    state = {"budget_usage": {"stages": {"deep_search": {"tasks": 884}}}}
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_mlip_mc",
        "stage": "deep_search",
        "task_key": "prepare",
        "target_ids": [],
        "parameters": {"mode": "mc_inputs", "expected_task_ids": ["a"]},
        "budget": 0,
    }
    result = validate_tool_action(
        action, state, session, {"prepare_local_batch_files": {"handler": lambda **_: None}}
    )
    assert "stage_task_limit" not in result["errors"]

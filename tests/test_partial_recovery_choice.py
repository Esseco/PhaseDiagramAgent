from phase_agent.runtime.chat_state_presentation import brief_chat_state
from phase_agent.runtime.partial_recovery import partial_recovery_facts
from phase_agent.runtime.result_recovery_status import result_recovery_facts


def _dft_state():
    tasks = [
        {
            "task_id": f"D{i}",
            "task_key": f"key-D{i}",
            "stage": "dft_single_point",
            "status": "completed" if i < 12 else "failed",
            "model_version": "m1",
            "config_version": "c1",
            "upload_operation_id": "op1",
            "search_group_index": 1 if i < 11 else 3,
        }
        for i in range(13)
    ]
    return {
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
        "tasks": tasks,
    }


def test_dft_partial_return_exposes_12_of_13_and_single_fill_scope():
    state = _dft_state()
    rows = partial_recovery_facts(state)
    assert len(rows) == 1
    assert rows[0]["coverage"] == "12/13"
    assert rows[0]["failed_task_ids"] == ["D12"]
    assert rows[0]["retryable_task_ids"] == ["D12"]
    text = brief_chat_state(state)
    assert "12/13" in text
    assert "只补全失败项" in text
    assert "不会重跑整批" in text


def test_mc_partial_return_uses_the_same_choice_without_batch_resubmission():
    state = {
        "active_model_version": "m1",
        "tasks": [
            {
                "task_id": "M1",
                "task_key": "key-M1",
                "stage": "deep_search",
                "status": "completed",
                "model_version": "m1",
                "relax_mc_bundle_id": "b1",
            },
            {
                "task_id": "M2",
                "task_key": "key-M2",
                "stage": "deep_search",
                "status": "completed",
                "model_version": "m1",
                "relax_mc_bundle_id": "b1",
            },
            {
                "task_id": "M3",
                "task_key": "key-M3",
                "stage": "deep_search",
                "status": "failed",
                "model_version": "m1",
                "relax_mc_bundle_id": "b1",
            },
        ],
        "relax_mc_batches": {
            "b1": {
                "model_version": "m1",
                "status": "partial",
                "import_checksum": "verified",
                "result_summary": {
                    "status": "partial",
                    "task_count": 3,
                    "failed_task_ids": ["M3"],
                },
            }
        },
    }
    rows = partial_recovery_facts(state)
    assert len(rows) == 1
    assert rows[0]["kind"] == "mc" and rows[0]["coverage"] == "2/3"
    facts = result_recovery_facts(state)[0]
    assert "成功 2/3" in facts["label"]
    assert "只补全失败项" in facts["label"]
    assert "不会整批重跑" in facts["label"]
    assert "2/3" in brief_chat_state(state)

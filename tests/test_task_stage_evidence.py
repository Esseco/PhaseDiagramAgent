from analysis_layer.state.summarize_task_stages import summarize_task_stages


def test_task_order_does_not_change_evidence():
    tasks = [{"task_id": "a", "stage": "deep_search", "status": "completed"},
             {"task_id": "b", "stage": "deep_search", "status": "running"},
             {"task_id": "c", "stage": "dft_single_point", "status": "failed"}]
    first = summarize_task_stages({"tasks": tasks, "processed_task_ids": ["a"]})
    assert first == summarize_task_stages({"tasks": list(reversed(tasks)), "processed_task_ids": ["a"]})
    assert first["stages"][0]["all_completed_and_accepted"] is False


def test_completed_is_not_accepted_without_recovery():
    state = {"tasks": [{"task_id": "a", "stage": "deep_search", "status": "completed"}]}
    row = summarize_task_stages(state)["stages"][0]
    assert row["completed_unaccepted_count"] == 1
    assert row["all_completed_and_accepted"] is False
    state["processed_task_ids"] = ["a"]
    assert summarize_task_stages(state)["stages"][0]["all_completed_and_accepted"] is True


def test_unknown_status_is_not_completion():
    row = summarize_task_stages({"tasks": [{}]})["stages"][0]
    assert row["status_counts"] == {"unknown": 1}
    assert row["all_completed_and_accepted"] is False

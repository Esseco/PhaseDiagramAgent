from phase_agent.analysis.state.summarize_task_rounds import summarize_task_rounds


def test_shared_relax_parent_keeps_mc_segments_distinct():
    base = {"model_version": "m1", "search_group_index": 1, "parent_relax_round": 1,
            "stage": "deep_search", "status": "completed"}
    tasks = [{**base, "task_id": "a", "segment_index": 0},
             {**base, "task_id": "b", "segment_index": 1}]
    result = summarize_task_rounds({"tasks": tasks, "processed_task_ids": ["a"]})
    assert len(result["groups"]) == 2
    assert result == summarize_task_rounds({"tasks": tasks[::-1], "processed_task_ids": ["a"]})
    assert all(row["lineage"]["parent_relax_round"] == 1 for row in result["groups"])


def test_missing_lineage_is_not_invented():
    row = summarize_task_rounds({"tasks": [{"stage": "dft_single_point"}]})["groups"][0]
    assert row["lineage"]["parent_relax_round"] is None
    assert "model_version" in row["missing_lineage_fields"]

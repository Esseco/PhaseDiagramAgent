from scientific_layer.mc.second_round_state import second_round_completed
from scientific_layer.qbc.post_mc_candidates import post_mc_candidates
from run.open_webui_api import format_workflow_reply


def test_complete_wave_requires_all_tasks():
    state = {"mc_second_round_allocations": [{"model_version": "m", "task_ids": ["t"]}],
             "tasks": [{"task_id": "t", "stage": "deep_search", "model_version": "m",
                        "segment_index": 1, "status": "completed"}]}
    assert second_round_completed(state, "m")
    state["tasks"][0]["status"] = "pending"
    assert not second_round_completed(state, "m")


def test_candidates_use_recovered_structure(tmp_path):
    path = tmp_path / "final.vasp"
    path.write_text("structure")
    state = {"phase_diagrams": {"mlip": {"model_version": "m", "status": "completed",
        "version": "h", "entries": [{"structure_path": str(path),
            "phase_identification_status": "identified", "phase": "O3", "ehull_unit": "eV/atom", "ehull": 0.01}]}},
        "tasks": [{"stage": "deep_search", "status": "completed", "model_version": "m",
                   "branch_id": "b", "structure_id": "s", "segment_index": 1,
                   "outputs": {"structure_path": str(path)}}]}
    rows = post_mc_candidates(state, "m")
    assert rows[0]["structure_path"] == str(path)
    assert rows[0]["predicted_Ehull"] == 0.01
    assert rows[0]["qbc"] == {"status": "not_configured"}
    assert post_mc_candidates(state, "other") == []


def test_nested_reason_is_preserved():
    text = format_workflow_reply({"status": "not_configured", "events": [
        {"reason": "第二轮完成，缺少候选证据"}]}, "state.json")
    assert "缺少候选证据" in text and "执行接口未配置" not in text

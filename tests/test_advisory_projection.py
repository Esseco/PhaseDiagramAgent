from copy import deepcopy
import json

from phase_agent.evidence.advisory_projection import project_advisory_payload


def test_large_advisory_inventory_is_compact_and_explicit_without_modifying_source():
    tasks = [
        {
            "task_id": str(i),
            "stage": "relax_and_feature",
            "status": "completed",
            "outputs": {"forces": [[0.1, 0.2, 0.3]] * 200},
        }
        for i in range(200)
    ]
    candidates = [{"structure_id": str(i), "H": i} for i in range(200)]
    raw = {
        "state": {"tasks": tasks, "human_constraints": ["no retry"]},
        "decision_context": {
            "model_refresh": {"plan": {"candidates": candidates, "checksum": "c"}}
        },
    }
    before = deepcopy(raw)
    projected = project_advisory_payload(raw, {"sha256": "state"})
    assert raw == before
    assert projected["state"]["tasks"]["count"] == 200
    assert projected["state"]["tasks"]["stage_status_counts"][0]["count"] == 200
    assert projected["decision_context"]["model_refresh"]["plan"]["checksum"] == "c"
    assert projected["decision_context"]["model_refresh"]["plan"]["candidates"]["count"] == 200
    assert projected["state"]["human_constraints"] == ["no retry"]
    assert len(json.dumps(projected)) < len(json.dumps(raw)) / 50


def test_small_inputs_are_not_modified_and_detached_evidence_is_bound():
    small = {"state": {"tasks": [{"task_id": "a"}]}}
    assert project_advisory_payload(small) == small
    a = {"state": {"tasks": [{"task_id": str(i)} for i in range(129)]}}
    b = deepcopy(a)
    b["state"]["tasks"][-1]["task_id"] = "changed"
    assert (
        project_advisory_payload(a)["state"]["advisory_source_identity"]
        != (project_advisory_payload(b)["state"]["advisory_source_identity"])
    )


def test_team_snapshot_bound_paging_returns_all_original_tasks(tmp_path):
    import pytest
    from phase_agent.tools.step_runner.file_protocol import write_json
    from tests.test_codex_native_team import make_control, team

    control, _ = make_control(tmp_path)
    rows = [
        {
            "task_id": f"t-{i}",
            "stage": "relax_and_feature",
            "status": "completed",
            "outputs": {"forces": [[i, 0, 0]]},
        }
        for i in range(130)
    ]
    write_json(control.state_path, {"tasks": rows})
    t = team(control)
    view = t.start(question="review", consult=["exploration"])
    request = view["requests"]["exploration"]
    inventory = request["evidence_snapshot"]["facts"]["state"]["tasks"]
    assert inventory["format"] == "deferred_evidence_v1" and inventory["count"] == len(rows)
    source = inventory["source_ref"]
    snap = view["snapshot"]["snapshot_id"]
    pages = []
    offset = 0
    while offset is not None:
        result = t.evidence(view["team_id"], 1, snap, source, offset=offset, limit=50)
        pages.extend(result["items"])
        offset = result["next_offset"]
    assert pages == rows
    result = t.evidence(view["team_id"], 1, snap, source, task_ids=["t-7"])
    assert result["items"] == [rows[7]]
    with pytest.raises(ValueError, match="missing_or_duplicated"):
        t.evidence(view["team_id"], 1, snap, source, task_ids=["missing"])
    with pytest.raises(ValueError, match="snapshot_changed"):
        t.evidence(view["team_id"], 1, "wrong", source)
    rows[0]["status"] = "failed"
    write_json(control.state_path, {"tasks": rows})
    with pytest.raises(ValueError, match="stale"):
        t.evidence(view["team_id"], 1, snap, source)


def test_one_large_task_is_also_deferred():
    raw = {"state": {"tasks": [{"task_id": "large", "outputs": {"log": "x" * 300000}}]}}
    projected = project_advisory_payload(raw, {"sha256": "source"})
    assert projected["state"]["tasks"]["count"] == 1
    assert "state.tasks" in projected["decision_context"]["deferred_evidence"]["fields"]

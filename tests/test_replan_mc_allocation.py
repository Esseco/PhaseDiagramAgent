from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.tools.local import replan_mc_allocation as module
from phase_agent.tools.local import regenerate_mc_inputs as regeneration
from tests.test_regenerate_mc_inputs import _state


def fixture(tmp_path, monkeypatch):
    state = _state(tmp_path)
    state["current_branch_hull_version"] = "hull"
    state["branch_hull_batches"] = {
        "hull": {"model_version": "model", "records": [{"branch_id": "B-1"}, {"branch_id": "B-2"}]}
    }
    manager = SimpleNamespace(data={"branches": {"B-1": {}, "B-2": {}}})
    old = state["action_records"][0]["final_action"]["parameters"]["budget_preview"]
    second = {**old["allocations"][0], "task_key": "new-key", "branch_id": "B-2"}
    preview = {
        **deepcopy(old),
        "allocations": [old["allocations"][0], second],
        "selected_branch_count": 2,
        "requested_steps": 60,
        "estimated_relative_cost": 100,
        "allocation_checksum": "new",
    }

    def estimate(candidates, pool, clean, config, **kwargs):
        assert clean["tasks"] == []
        assert clean["tiered_mc_state"]["segments"] == []
        assert len(candidates) == 2
        assert kwargs["step_limit"] == float("inf")
        return deepcopy(preview)

    monkeypatch.setattr(module, "estimate_branch_mc_budget", estimate)
    config = {
        "round_strategy": {"maximum_mc_budget": 100},
        "budgets": {"stage_limits": {"deep_search": {"max_cost": 200, "max_tasks": 10}}},
    }
    return state, manager, config


def test_replanning_covers_current_relax_candidates_without_changing_state(tmp_path, monkeypatch):
    state, manager, config = fixture(tmp_path, monkeypatch)
    original = deepcopy(state)
    preview = module.replan_mc_allocation(state, manager, config)
    assert preview["selected_branch_count"] == 2
    assert preview["step_limit"] == 60
    assert state == original


@pytest.mark.parametrize("limit", ["steps", "tasks", "cost"])
def test_full_plan_is_not_silently_truncated_to_fit_limits(tmp_path, monkeypatch, limit):
    state, manager, config = fixture(tmp_path, monkeypatch)
    if limit == "steps":
        config["round_strategy"]["maximum_mc_budget"] = 30
    else:
        config["budgets"]["stage_limits"]["deep_search"]["max_" + limit] = 1
    with pytest.raises(ValueError, match="上限"):
        module.replan_mc_allocation(state, manager, config)


def test_expanded_replacement_removes_old_scope_and_records_new_scope(tmp_path, monkeypatch):
    state, manager, config = fixture(tmp_path, monkeypatch)
    preview = module.replan_mc_allocation(state, manager, config)
    plan = regeneration.plan_mc_regeneration(state, tmp_path, replacement_preview=preview)
    assert len(plan["old_task_ids"]) == 1
    assert plan["allocation_count"] == 2

    def prepare(*, action, context):
        return {
            "status": "prepared",
            "task_count": 2,
            "batch_count": 1,
            "state": context["event_state"],
        }

    monkeypatch.setattr(regeneration, "prepare_mc_upload_batches", prepare)
    result = regeneration.regenerate_mc_inputs(
        state,
        approved_plan=plan,
        upload_root=tmp_path,
        config=config,
        manager=manager,
        phase_references={},
        config_version="v1",
        replacement_preview=preview,
    )
    assert len(result["state"]["tasks"]) == 2
    assert result["state"]["action_records"] == state["action_records"]
    current, source = regeneration._approved_mc_preview(result["state"])
    assert current == preview and source == "user_requested_replanning"


def test_submitted_or_completed_mc_cannot_be_replanned(tmp_path, monkeypatch):
    state, manager, config = fixture(tmp_path, monkeypatch)
    state["tasks"][0]["status"] = "completed"
    with pytest.raises(ValueError, match="尚未提交"):
        module.replan_mc_allocation(state, manager, config)


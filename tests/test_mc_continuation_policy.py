from copy import deepcopy
import pytest
from phase_agent.science.mc.continuation_policy import effective_mc_policy
from phase_agent.science.mc.schedule_tiered_mc import schedule_tiered_mc
from phase_agent.science.mc.second_round_state import first_round_source


def inputs(count=12, improved=2):
    candidates = [
        {
            "branch_id": f"B{i}",
            "x": 0.5,
            "P": "O3" if i % 2 else "P2",
            "relaxed_ehull": 0.01 + i * 0.001,
            "structure_path": f"initial{i}.vasp",
        }
        for i in range(count)
    ]
    state = {
        "segments": [
            {
                "branch_id": f"B{i}",
                "task_key": f"initial:{i}",
                "segment_index": 0,
                "model_version": "m1",
                "status": "completed",
                "energy_improvement": 0.01 if i < improved else 0,
                "min_improvement": 0.001,
                "tier_index": 0,
                "result_path": f"result{i}.vasp",
                "actual_cost": 500,
            }
            for i in range(count)
        ],
        "processed_task_keys": [],
    }
    config = {
        "mc_policy": {
            "tiers": [
                {"max_mc_steps": 50, "patience_steps": 5, "min_improvement": 0.001},
                {"max_mc_steps": 100, "patience_steps": 10, "min_improvement": 0.001},
                {"max_mc_steps": 200, "patience_steps": 20, "min_improvement": 0.001},
            ],
            "max_segments_per_branch": 2,
            "max_cumulative_cost_per_branch": 200,
            "random_exploration_fraction": 0.1,
        },
        "budgets": {"enforcement": "advisory"},
    }
    return candidates, state, config


def run(candidates, state, config):
    return schedule_tiered_mc(
        candidates,
        state,
        policy=effective_mc_policy(config),
        total_budget=1,
        seed=42,
        model_version="m1",
        hull_reference_version="h1",
        phase_diagram_version="p1",
        scope_by_model=True,
    )


def test_two_improved_candidates_are_filled_to_ten_and_cost_reference_does_not_block():
    candidates, state, config = inputs()
    result = run(candidates, state, config)
    assert len(result["actions"]) == 10
    assert sum(row["improvement_class"] == "strong" for row in result["actions"]) == 2
    assert len({row["task_key"] for row in result["actions"]}) == 10
    assert len({row["branch_id"] for row in result["actions"]}) == 10
    assert all(row["segment_index"] == 1 for row in result["actions"])


def test_valid_pool_below_ten_rolls_back_without_padding_duplicates():
    candidates, state, config = inputs(5)
    result = run(candidates, state, config)
    assert result["status"] == "continuation_pool_below_minimum"
    assert result["actions"] == [] and result["state"] == state


def test_two_additions_are_possible_but_third_is_blocked_and_model_reset_is_scoped():
    candidates, state, config = inputs(10, 10)
    initial_source = first_round_source({"tiered_mc_state": state}, "m1")
    first = run(candidates, state, config)
    for row in first["state"]["segments"]:
        row.update(
            status="completed", energy_improvement=0.01, result_path=f"{row['task_key']}.vasp"
        )
    next_source = first_round_source({"tiered_mc_state": first["state"]}, "m1")
    assert next_source and next_source["checksum"] != initial_source["checksum"]
    assert len(next_source["branch_ids"]) == 10 and next_source["completed_count"] == 10
    second = run(candidates, first["state"], config)
    assert len(second["actions"]) == 10 and all(
        row["segment_index"] == 2 for row in second["actions"]
    )
    for row in second["state"]["segments"]:
        row.update(
            status="completed", energy_improvement=0.01, result_path=f"{row['task_key']}.vasp"
        )
    assert first_round_source({"tiered_mc_state": second["state"]}, "m1") is None
    assert run(candidates, second["state"], config)["actions"] == []
    fresh = schedule_tiered_mc(
        candidates,
        second["state"],
        policy=effective_mc_policy(config),
        total_budget=1,
        seed=42,
        model_version="m2",
        hull_reference_version="h2",
        scope_by_model=True,
    )
    assert len(fresh["actions"]) == 10 and all(
        row["segment_index"] == 0 for row in fresh["actions"]
    )


def test_weak_feedback_does_not_force_continuation_and_pending_work_is_not_repeated():
    candidates, state, config = inputs(10, 0)
    result = run(candidates, state, config)
    assert result["actions"] == []
    state["segments"][0]["status"] = "running"
    assert run(candidates, state, config)["status"] == "tasks_in_progress"


def test_explicit_hard_budget_keeps_cost_guard():
    candidates, state, config = inputs()
    config["budgets"]["enforcement"] = "hard"
    assert run(candidates, state, config)["actions"] == []


@pytest.mark.parametrize("mixed,count", [(False, 10), (True, 10), (False, 20)])
def test_sequential_preview_dispatch_and_third_segment_use_same_real_task_plan(
    tmp_path, mixed, count
):
    import json
    from types import SimpleNamespace
    from tests.test_relax_mc_handoff import _completed_relax_fixture
    from phase_agent.analysis.phase.branch_relax_hull import build_relax_hull
    from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action
    from phase_agent.tools.workflows.create_active_learning_handlers import _allocate_mc_bohb

    branches, structures, tasks, rows, entries, segments = {}, {}, [], [], [], []
    for i in range(count):
        folder = tmp_path / str(i)
        folder.mkdir()
        manager, state, config = _completed_relax_fixture(folder)

        def rename(value):
            text = json.dumps(value).replace('"B1"', f'"B{i}"')
            for j in range(3):
                text = text.replace(f'"S{j}"', f'"S{i}_{j}"')
            return json.loads(text)

        data = rename(manager.data)
        branches.update(data["branches"])
        structures.update(data["structures"])
        for row in rename(state["tasks"]):
            row["task_id"] += f":{i}"
            row["task_key"] += f":{i}"
            tasks.append(row)
        rows.extend(
            rename(state["branch_hull_batches"][state["current_branch_hull_version"]]["records"])
        )
        entries.extend(rename(state["phase_diagrams"]["mlip"]["entries"]))
        final = folder / "mc-final.vasp"
        final.write_text(f"mc {i}")
        segment = {
            "task_id": f"MC:{i}",
            "task_key": f"mc:{i}",
            "branch_id": f"B{i}",
            "stage": "deep_search",
            "model_version": "m1",
            "status": "completed",
            "segment_index": 0,
            "tier_index": 0,
            "energy_improvement": 0.002 if mixed else 0.01,
            "min_improvement": 0.001,
            "stop_reason": "max_steps",
            "result_path": str(final),
            "outputs": {"energy_per_atom": -6},
        }
        tasks.append(segment)
        segments.append(deepcopy(segment))
        entries.append(
            {
                "structure_id": f"final:{i}",
                "structure_path": str(final),
                "ehull": (0.015 if i < 9 else 0.03) if mixed else 0,
                "ehull_unit": "eV/atom",
                "phase": "O3",
                "phase_identification_status": "identified",
            }
        )
    pool = build_relax_hull(rows, model_version="m1")
    state = {
        "tasks": tasks,
        "tiered_mc_state": {"segments": segments},
        "dedup_gate": {"status": "ready"},
        "current_branch_hull_version": pool["version"],
        "branch_hull_batches": {pool["version"]: pool},
        "phase_diagrams": {
            "mlip": {
                "method": "mlip",
                "status": "completed",
                "model_version": "m1",
                "version": "p1",
                "entries": entries,
            }
        },
    }
    manager = SimpleNamespace(
        data={"branches": branches, "structures": structures}, boundary={"P": ["O3"]}
    )
    from phase_agent.tools.workflows.prepare_mc_preview import prepare_mc_preview

    partial = prepare_mc_preview(
        {
            "tool": "allocate_mc_bohb",
            "task_key": "partial",
            "target_ids": ["B0", "B1"],
            "parameters": {"mc_budget": 900, "branch_count": 10},
        },
        state,
        {"manager": manager},
        config,
    )
    assert (
        len(partial["target_ids"]) == count
        and partial["parameters"]["budget_preview"]["selected_branch_count"] == count
    )

    adaptive_state = {**deepcopy(state), "search_mode": "Adaptive"}
    adaptive_action = prepare_mc_preview(
        {
            "tool": "allocate_mc_bohb",
            "task_key": "adaptive-partial",
            "target_ids": ["B0", "B1"],
            "parameters": {"mc_budget": 900, "branch_count": 10},
        },
        adaptive_state,
        {"manager": manager},
        config,
    )
    assert "round_kind" not in adaptive_action["parameters"]
    adaptive_result = _allocate_mc_bohb(
        action=adaptive_action,
        context={
            "manager": manager,
            "phase_references": {},
            "effective_config": config,
            "event_state": adaptive_state,
            "config_version": "v1",
            "execution_mode": "automatic",
        },
    )
    assert adaptive_result["status"] == "completed" and len(adaptive_result["actions"]) == count, (
        adaptive_result.get("reason")
    )
    if not mixed:
        assert all(row["resource_tier"] == "high" for row in adaptive_result["actions"])
        assert (
            adaptive_action["parameters"]["budget_preview"]["selection"]["high_required_count"]
            == count
        )
    if mixed:
        assert sum(row["selection_group"] == "main" for row in adaptive_result["actions"]) == 9
        assert (
            sum(row["selection_group"] == "exploration" for row in adaptive_result["actions"]) == 1
        )
        assert {row["max_mc_steps"] for row in adaptive_result["actions"]} == {100, 150}

    def dispatch(current):
        action = choose_debug_next_action(
            current,
            manager,
            config,
            allowed_tools=config["agent"]["allowed_tools"],
            user_message="继续",
        )
        assert action["parameters"]["budget_preview"]["selected_branch_count"] == count
        result = _allocate_mc_bohb(
            action=action,
            context={
                "manager": manager,
                "phase_references": {},
                "effective_config": config,
                "event_state": current,
                "config_version": "v1",
                "execution_mode": "automatic",
            },
        )
        assert result["status"] == "completed" and len(result["actions"]) == count
        return action, result

    action, first = dispatch(state)
    assert [
        {
            key: row[key]
            for key in ("branch_id", "max_mc_steps", "patience_steps", "selection_group")
        }
        for row in first["actions"]
    ] == [
        {
            key: row[key]
            for key in ("branch_id", "max_mc_steps", "patience_steps", "selection_group")
        }
        for row in adaptive_result["actions"]
    ]
    duplicate = _allocate_mc_bohb(
        action=action,
        context={
            "manager": manager,
            "phase_references": {},
            "effective_config": config,
            "event_state": first["state"],
            "config_version": "v1",
            "execution_mode": "automatic",
        },
    )
    assert duplicate["reason"] == "second_mc_round_already_allocated"
    current = first["state"]
    for row in current["tasks"]:
        if row.get("stage") == "deep_search" and row.get("status") == "pending":
            final = tmp_path / (row["task_id"] + ".vasp")
            final.write_text("next")
            row.update(
                status="completed",
                energy_improvement=0.01,
                result_path=str(final),
                outputs={"energy_per_atom": -6},
            )
            current["phase_diagrams"]["mlip"]["entries"].append(
                {
                    "structure_path": str(final),
                    "ehull": 0,
                    "ehull_unit": "eV/atom",
                    "phase": "O3",
                    "phase_identification_status": "identified",
                }
            )
    _, second = dispatch(current)
    assert all(row["segment_index"] == 2 for row in second["actions"])

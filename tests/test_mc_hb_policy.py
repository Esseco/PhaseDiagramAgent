"""Offline scientific scheduling checks; no models or HPC connectors."""

from copy import deepcopy
import pytest
from phase_agent.science.mc.continuation_policy import effective_mc_policy
from phase_agent.science.mc.schedule_tiered_mc import (
    schedule_tiered_mc,
    _hb_resource_tier,
    _hb_improvement_class,
)
from phase_agent.science.mc.second_round_state import first_round_source


def candidate(i, gap=0.015):
    return {
        "branch_id": f"B{i}",
        "P": "O3" if i % 2 else "P2",
        "x": 0.25 if i % 3 else 0.75,
        "relaxed_ehull": gap,
        "structure_path": f"initial{i}.vasp",
    }


def schedule(candidates, state=None, **overrides):
    seed = overrides.pop("seed", 42)
    policy = effective_mc_policy(
        {
            "mc_policy": {
                "tiers": [{"max_mc_steps": 10, "patience_steps": 1, "min_improvement": 0.9}]
            }
        }
    )
    return schedule_tiered_mc(
        candidates,
        state,
        policy={**policy, **overrides},
        total_budget=1,
        seed=seed,
        model_version="m1",
        hull_reference_version="h1",
        scope_by_model=True,
    )


def complete(result, improvement=0.002):
    state = deepcopy(result["state"])
    for row in state["segments"]:
        row.update(
            status="completed",
            result_path=f"{row['task_key']}.vasp",
            energy_improvement=improvement,
            outputs={"energy_per_atom": -6},
        )
    return state


def test_halving_80_40_20_and_shared_feedback_resources():
    candidates = [candidate(i) for i in range(80)]
    first = schedule(candidates)
    assert len(first["actions"]) == 80
    second = schedule(candidates, complete(first))
    assert len(second["actions"]) == 40
    state = complete(second)
    source = first_round_source({"tiered_mc_state": state}, "m1")
    assert source["completed_count"] == 40
    third = schedule(candidates, state)
    assert len(third["actions"]) == 20
    for result, pair in [(first, (80, 8)), (second, (150, 15)), (third, (150, 15))]:
        assert {(row["max_mc_steps"], row["patience_steps"]) for row in result["actions"]} == {pair}
        assert all(row["min_improvement"] is None for row in result["actions"])
    assert schedule(candidates, complete(third))["actions"] == []


@pytest.mark.parametrize("gap,steps", [(0.010, 100), (0.020, 80), (0.040, 60), (0.041, 40)])
def test_four_initial_bands_and_reclassification_after_ehull_change(gap, steps):
    candidates = [candidate(i, gap) for i in range(10)]
    first = schedule(candidates, random_exploration_fraction=1)
    assert len(first["actions"]) == 10
    assert all(
        row["max_mc_steps"] == steps and row["patience_steps"] == steps // 10
        for row in first["actions"]
    )
    # Both additions reclassify against latest Ehull and segment feedback.
    for row in candidates:
        row["relaxed_ehull"] = 0.015
    second = schedule(candidates, complete(first))
    assert all(
        row["max_mc_steps"] == 150 and row["patience_steps"] == 15 for row in second["actions"]
    )


@pytest.mark.parametrize(
    "gap,tiers",
    [
        (0.010, ("high", "high", "medium", "low")),
        (0.020, ("high", "medium", "low", "exploration")),
        (0.040, ("medium", "low", "exploration", None)),
        (0.041, ("low", "exploration", None, None)),
    ],
)
@pytest.mark.parametrize(
    "improvement,column,grade",
    [
        (0.0031, 0, "strong"),
        (0.003, 1, "moderate"),
        (0.001, 1, "moderate"),
        (0.0009, 2, "small"),
        (0.0005, 2, "small"),
        (0.00049, 3, "weak"),
    ],
)
def test_four_grade_resource_matrix_and_exact_boundaries(gap, tiers, improvement, column, grade):
    policy = effective_mc_policy({})
    previous = [{"energy_improvement": improvement}]
    assert _hb_improvement_class(previous, policy) == grade
    assert _hb_resource_tier(candidate(0, gap), previous, policy) == tiers[column]


def test_protect_weak_ground_and_competitors_and_do_not_revive_eliminated_branches():
    candidates = [candidate(i) for i in range(40)]
    first = schedule(candidates)
    state = complete(first, 0.002)
    for i in (38, 39):
        candidates[i]["relaxed_ehull"] = 0 if i == 38 else 0.003
        state["segments"][i]["energy_improvement"] = 0
    second = schedule(candidates, state)
    assert len(second["actions"]) == 20
    assert {"B38", "B39"} <= {row["branch_id"] for row in second["actions"]}
    third = schedule(candidates, complete(second))
    assert len(third["actions"]) == 10
    assert {row["branch_id"] for row in third["actions"]} <= {
        row["branch_id"] for row in second["actions"]
    }


def test_minimum_rollback_active_block_and_agent_larger_batch():
    candidates = [candidate(i) for i in range(30)]
    state = complete(schedule(candidates))
    result = schedule(candidates, state, agent_branch_count=22)
    assert len(result["actions"]) == 22
    assert result["selection"]["reference_count"] == 15
    result = schedule(candidates[:5], state)
    assert result["status"] == "continuation_pool_below_minimum" and result["state"] == state
    assert not result["actions"]
    state["segments"][0]["status"] = "running"
    assert schedule(candidates, state)["status"] == "tasks_in_progress"


def test_initial_coverage_exploration_and_protection_above_count():
    candidates = [candidate(i, 0.008 if i < 80 else 0.08) for i in range(100)]
    candidates[0]["relaxed_ehull"] = 0
    result = schedule(candidates, agent_branch_count=20)
    assert len(result["actions"]) == 20
    assert sum(row["relaxed_ehull"] > 0.04 for row in result["actions"]) == 2
    assert {row["branch_id"] for row in result["actions"]} >= {"B0"}
    assert {row["branch_id"] for row in result["actions"]} <= {
        row["branch_id"] for row in candidates
    }
    assert {
        row.get("P")
        for row in candidates
        if row["branch_id"] in {a["branch_id"] for a in result["actions"]}
    } == {"O3", "P2"}
    for row in candidates[:25]:
        row["relaxed_ehull"] = 0
    assert len(schedule(candidates, agent_branch_count=20)["actions"]) >= 25


def test_initial_selection_rotates_structure_families_within_same_phase_composition():
    candidates = [
        {**candidate(i, 0.01), "P": "O3", "x": 0.5, "H": [[1 if i < 10 else 2]], "T": ["Fe"]}
        for i in range(20)
    ]
    result = schedule(candidates, agent_branch_count=2)
    assert len(result["actions"]) == 2
    ids = {row["branch_id"] for row in result["actions"]}
    assert any(row["branch_id"] in ids for row in candidates[:10])
    assert any(row["branch_id"] in ids for row in candidates[10:])


def test_both_additions_use_ninety_ten_main_and_seeded_exploration():
    candidates = [candidate(i, 0.015 if i < 30 else 0.03) for i in range(40)]
    state = complete(schedule(candidates))
    second = schedule(candidates, state)
    assert len(second["actions"]) == 20
    assert sum(row["selection_group"] == "main" for row in second["actions"]) == 18
    assert sum(row["selection_group"] == "exploration" for row in second["actions"]) == 2
    assert {(row["max_mc_steps"], row["patience_steps"]) for row in second["actions"]} == {
        (150, 15),
        (100, 10),
    }
    assert second["actions"] == schedule(list(reversed(candidates)), state)["actions"]
    third = schedule(candidates, complete(second))
    assert len(third["actions"]) == 10
    assert sum(row["selection_group"] == "main" for row in third["actions"]) == 9
    assert sum(row["selection_group"] == "exploration" for row in third["actions"]) == 1
    assert {(row["max_mc_steps"], row["patience_steps"]) for row in third["actions"]} == {
        (150, 15),
        (100, 10),
    }
    explore = lambda result: {
        row["branch_id"]
        for row in result["actions"]
        if row["selection_group"].startswith("exploration")
    }
    assert explore(second) != explore(schedule(candidates, state, seed=17))


def test_exploration_rotates_low_and_exploration_tiers():
    candidates = [candidate(i, 0.015 if i < 40 else 0.03 if i < 50 else 0.05) for i in range(60)]
    # All candidates are included solely to construct this offline completed-wave fixture.
    state = complete(schedule(candidates, random_exploration_fraction=1))
    result = schedule(candidates, state)
    explorations = [row for row in result["actions"] if row["selection_group"] == "exploration"]
    assert len(result["actions"]) == 30 and len(explorations) == 3
    assert {row["resource_tier"] for row in explorations} == {"low", "exploration"}


def test_scarce_main_increases_exploration_but_no_useful_feedback_stops():
    candidates = [candidate(i) for i in range(12)]
    state = complete(schedule(candidates), 0)
    for row in state["segments"]:
        if row["branch_id"] in {"B0", "B1"}:
            row["energy_improvement"] = 0.004
    result = schedule(candidates, state)
    assert len(result["actions"]) == 10
    assert sum(row["selection_group"] == "main" for row in result["actions"]) == 2
    assert result["selection"]["actual_exploration_count"] == 8
    assert sum(row["selection_group"] == "exploration_fill" for row in result["actions"]) == 7
    assert all(
        row["max_mc_steps"] == 50
        for row in result["actions"]
        if row["selection_group"].startswith("exploration")
    )
    exhausted = complete(schedule(candidates), 0)
    assert schedule(candidates, exhausted)["actions"] == []


def test_protected_weak_candidates_at_least_medium_can_expand_batch():
    candidates = [candidate(i, 0 if i < 25 else 0.03) for i in range(40)]
    state = complete(schedule(candidates), 0.002)
    for row in state["segments"]:
        if int(row["branch_id"][1:]) < 25:
            row["energy_improvement"] = 0
    result = schedule(candidates, state)
    protected = [row for row in result["actions"] if row["protected_candidate"]]
    assert len(protected) == 25 and len(result["actions"]) > 25
    assert all(row["resource_tier"] == "medium" and row["max_mc_steps"] == 150 for row in protected)


def test_feedback_can_reduce_second_addition_budget_and_ineligible_candidates_are_not_padded():
    candidates = [candidate(i, 0.015) for i in range(10)]
    state = complete(schedule(candidates), 0.004)
    second = schedule(candidates, state)
    assert all(row["max_mc_steps"] == 200 for row in second["actions"])
    for row in candidates:
        row["relaxed_ehull"] = 0.03
    third = schedule(candidates, complete(second, 0.002))
    assert len(third["actions"]) == 10 and all(
        row["max_mc_steps"] == 100 for row in third["actions"]
    )
    for row in candidates:
        row["relaxed_ehull"] = 0.05
    assert schedule(candidates, complete(second, 0.0006))["actions"] == []


def test_every_high_candidate_is_retained_in_both_additions_despite_count_reference():
    candidates = [candidate(i) for i in range(40)]
    first = schedule(candidates)
    second = schedule(candidates, complete(first, 0.004), agent_branch_count=10)
    third = schedule(candidates, complete(second, 0.004), agent_branch_count=10)
    for result in (second, third):
        assert len(result["actions"]) == 40
        assert {row["branch_id"] for row in result["actions"]} == {
            row["branch_id"] for row in candidates
        }
        assert all(
            row["resource_tier"] == "high" and row["max_mc_steps"] == 200
            for row in result["actions"]
        )
        assert result["selection"]["reference_count"] == 20
        assert result["selection"]["high_required_count"] == 40
        assert result["selection"]["mandatory_count"] == 40
    assert schedule(candidates, complete(third, 0.004))["actions"] == []


def test_high_and_protected_union_expands_for_exploration_without_double_count():
    candidates = [candidate(i, 0 if i < 2 else 0.015 if i < 30 else 0.03) for i in range(40)]
    state = complete(schedule(candidates), 0.002)
    for row in state["segments"]:
        if int(row["branch_id"][1:]) < 30:
            row["energy_improvement"] = 0.004
    result = schedule(candidates, state, agent_branch_count=10)
    assert len(result["actions"]) == 34
    assert result["selection"]["high_required_count"] == 30
    assert result["selection"]["protected_count"] == 2
    assert result["selection"]["mandatory_count"] == 30
    assert result["selection"]["actual_exploration_count"] == 4
    assert {f"B{i}" for i in range(30)} <= {row["branch_id"] for row in result["actions"]}
    assert len({row["task_key"] for row in result["actions"]}) == 34


def test_high_and_weak_protected_are_both_mandatory_and_medium_only_fills_remaining_slots():
    candidates = [candidate(i, 0 if i < 4 else 0.015 if i < 32 else 0.03) for i in range(40)]
    state = complete(schedule(candidates), 0.002)
    for row in state["segments"]:
        index = int(row["branch_id"][1:])
        row["energy_improvement"] = 0 if index < 4 else 0.004 if index < 28 else 0.002
    result = schedule(candidates, state, agent_branch_count=10)
    assert result["selection"]["mandatory_count"] == 28
    assert result["selection"]["high_required_count"] == 24
    assert {f"B{i}" for i in range(28)} <= {row["branch_id"] for row in result["actions"]}
    assert len(result["actions"]) == 32
    assert result["selection"]["actual_exploration_count"] == 4
    assert all(
        row["resource_tier"] == "medium" for row in result["actions"] if row["protected_candidate"]
    )


@pytest.mark.parametrize("mode", ["Sequential", "Adaptive"])
def test_mc_dry_run_calls_no_handlers_and_creates_no_tasks(mode, tmp_path, monkeypatch):
    import subprocess
    from tests.test_relax_mc_handoff import _completed_relax_fixture
    from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.tools.workflows.run_event_loop import run_event_loop

    calls = []

    def forbidden(*args, **kwargs):
        calls.append("side_effect")
        raise AssertionError("Dry-run called a backend or subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    manager, state, config = _completed_relax_fixture(tmp_path)
    state["search_mode"] = mode
    before = deepcopy(state)
    action = choose_debug_next_action(
        state, manager, config, allowed_tools=config["agent"]["allowed_tools"], user_message="继续"
    )
    plan = {
        "config_version": "v1",
        "model_version": "m1",
        "reason": "MC dry-run",
        "actions": [{"id": "mc", "kind": "MC", "depends_on": [], "action": action}],
    }
    registry = create_tool_registry({name: forbidden for name in create_tool_registry()})
    result = run_event_loop(
        state,
        {"status": "confirmed", "confirmed_snapshot": {"config_version": "v1", "config": config}},
        registry=registry,
        context={"manager": manager},
        execution_mode="dry_run",
        max_steps=1,
        agent_client=lambda request: deepcopy(
            plan if request.get("mode") == "adaptive_plan" else action
        ),
        state_path=tmp_path / "dry-state.json",
    )
    assert result["status"] == "planned_only", result.get("reason")
    assert calls == [] and state == before
    assert result["state"]["tasks"] == before["tasks"]
    assert not result["state"].get("budget_reservations")
    assert not result["state"].get("slurm_batches")
    import json

    (tmp_path / f"{mode}-dry-run.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "status": result["status"],
                "handler_or_subprocess_calls": len(calls),
                "new_tasks": len(result["state"]["tasks"]) - len(before["tasks"]),
                "budget_reservations": len(result["state"].get("budget_reservations") or {}),
                "slurm_batches": len(result["state"].get("slurm_batches") or []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("configured", [None, 0.5])
def test_mc_worker_leaves_library_default_stopping_threshold_untouched(
    tmp_path, monkeypatch, configured
):
    import sys
    from types import SimpleNamespace
    from phase_agent.science.mlip.mace_worker import run_mace_worker

    captured = {}
    sentinel = object()

    class FakeMC:
        def __init__(self, min_improvement=sentinel, **kwargs):
            captured.update(kwargs)
            captured["threshold"] = min_improvement

        def run(self, **kwargs):
            return None

    monkeypatch.setitem(
        sys.modules, "Process_AL_MC", SimpleNamespace(LayeredOxide_MCOrderingClass=FakeMC)
    )
    run_mace_worker(
        {
            "operation": "mc",
            "structure_path": "fake.vasp",
            "output_directory": str(tmp_path),
            "segment_budget": 100,
            "stopping_threshold_source": "library_default",
            "parameters": {"patience_steps": 10, "patience": 99, "min_improvement": configured},
        }
    )
    assert captured["threshold"] is sentinel
    assert captured["max_steps"] == 100 and captured["patience"] == 10


def test_prepared_mc_payload_omits_inherited_threshold_and_is_not_submitted(tmp_path):
    import json
    from pathlib import Path
    from tests.test_relax_mc_handoff import _completed_relax_fixture
    from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action
    from phase_agent.tools.workflows.create_active_learning_handlers import _allocate_mc_bohb

    manager, state, config = _completed_relax_fixture(tmp_path)
    config["mlip"]["mc_parameters"]["min_improvement"] = 0.5
    action = choose_debug_next_action(
        state, manager, config, allowed_tools=config["agent"]["allowed_tools"], user_message="继续"
    )
    result = _allocate_mc_bohb(
        action=action,
        context={
            "manager": manager,
            "phase_references": {},
            "effective_config": config,
            "event_state": state,
            "config_version": "v1",
            "execution_mode": "interactive",
        },
    )
    assert result["status"] == "completed"
    assert result["mc_upload"]["submitted"] is False
    directory = result["mc_upload"]["batches"][0]["directory"]
    payload = json.loads(next(Path(directory).glob("*/task.json")).read_text(encoding="utf-8"))[
        "worker_job"
    ]
    assert payload["stopping_threshold_source"] == "library_default"
    assert "min_improvement" not in payload["parameters"]
    assert payload["segment_budget"] == 100 and payload["parameters"]["patience_steps"] == 10

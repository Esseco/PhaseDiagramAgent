from copy import deepcopy
import importlib

import pytest


def block_scientific_copy(monkeypatch):
    budget = importlib.import_module("phase_agent.tools.budget.reserve_budget")
    original = budget.deepcopy
    calls = []

    def checked(value, *args, **kwargs):
        calls.append(set(value) if isinstance(value, dict) else type(value).__name__)
        assert not (isinstance(value, dict) and "scientific_history_probe" in value), (
            "per-task budget reservation copied the complete scientific history"
        )
        return original(value, *args, **kwargs)

    monkeypatch.setattr(budget, "deepcopy", checked)
    return calls


def test_real_refresh_approval_does_not_copy_scientific_history_per_task(tmp_path, monkeypatch):
    from tests.test_model_refresh_workflow import setup
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    state, session, context, registry = setup(tmp_path)
    state["scientific_history_probe"] = {"large": list(range(100))}
    original = deepcopy(state)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    calls = block_scientific_copy(monkeypatch)
    batch = importlib.import_module("phase_agent.tools.remote.batch_runner")
    original_batch_copy = batch.deepcopy

    def bounded_copy(value):
        assert not (isinstance(value, dict) and "scientific_history_probe" in value)
        return original_batch_copy(value)

    monkeypatch.setattr(batch, "deepcopy", bounded_copy)
    result = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert result["status"] == "prepared", result
    assert result["state"]["model_refresh"]["status"] == "waiting_results"
    assert len(result["state"]["tasks"]) == 1
    assert state == original
    assert calls == [], "private reservations must not copy even the complete budget ledger"


def test_real_relax_mc_prepare_does_not_copy_scientific_history_per_task(tmp_path, monkeypatch):
    from tests.test_relax_mc_batch import fixture
    from phase_agent.tools.relax_mc_batch.plan import attach_preview
    from phase_agent.tools.relax_mc_batch.prepare import prepare_batch

    manager, state, config, context, action = fixture(tmp_path / "p", count=3)
    state["scientific_history_probe"] = {"large": list(range(100))}
    original = deepcopy(state)
    action = attach_preview(action, state, context, config)
    calls = block_scientific_copy(monkeypatch)
    result = prepare_batch(action=action, context=context)
    assert result["status"] == "prepared" and not result["submitted"]
    assert len(result["state"]["tasks"]) == 3
    assert state == original
    assert calls == []


def test_rejected_batch_discards_partial_private_reservations_and_creates_no_input(tmp_path):
    from tests.test_relax_mc_batch import fixture
    from phase_agent.tools.relax_mc_batch.plan import attach_preview
    from phase_agent.tools.relax_mc_batch.prepare import prepare_batch

    manager, state, config, context, action = fixture(tmp_path / "p", count=3)
    config["budgets"]["enforcement"] = "hard"
    config["budgets"]["stage_limits"]["relax_and_feature"]["max_tasks"] = 1
    action = attach_preview(action, state, context, config)
    original = deepcopy(state)
    result = prepare_batch(action=action, context=context)
    assert result["status"] == "rejected"
    assert result["state"] is state and state == original
    assert not list(tmp_path.rglob("plan.json"))
    assert not list(tmp_path.rglob("*_input"))


def test_owned_rejection_does_not_mutate_shared_ledger_or_create_empty_one():
    from phase_agent.tools.budget.reserve_budget import reserve_owned_budget

    limits = {"enforcement": "hard", "total_relative_cost": 0, "stage_limits": {"deep_search": {}}}
    state = {"history": {"large": [1] * 1000}}
    original = deepcopy(state)
    result = reserve_owned_budget(state, task_key="x", stage="deep_search", amount=1, limits=limits)
    assert result["status"] == "rejected" and state == original
    assert "budget_reservations" not in state


def test_private_budget_projection_matches_full_policy_and_keeps_history_identity():
    from phase_agent.tools.budget.reserve_budget import reserve_budget, reserve_owned_budget

    limits = {
        "enforcement": "hard",
        "total_relative_cost": 3,
        "stage_limits": {"deep_search": {"max_tasks": 2, "max_cost": 3}},
    }
    history = {"tasks": [{"completed": True}]}
    state = {"scientific_history_probe": history}
    expected = deepcopy(state)
    for key in ("a", "a", "b", "c"):
        args = dict(task_key=key, stage="deep_search", amount=1, limits=limits)
        baseline = reserve_budget(expected, **args)
        result = reserve_owned_budget(state, **args)
        assert result["status"] == baseline["status"]
        if baseline["status"] != "rejected":
            expected = baseline["state"]
        assert state == expected
        assert result["state"] is state
        assert state["scientific_history_probe"] is history


def test_default_budget_api_still_detaches_input():
    from phase_agent.tools.budget.reserve_budget import reserve_budget

    state = {"history": {"values": [1]}}
    result = reserve_budget(
        state,
        task_key="x",
        stage="deep_search",
        amount=1,
        limits={
            "enforcement": "advisory",
            "total_relative_cost": 2,
            "stage_limits": {"deep_search": {}},
        },
    )
    result["state"]["history"]["values"].append(2)
    assert state == {"history": {"values": [1]}}


def test_refresh_accepts_held_old_runner_without_retry(tmp_path, monkeypatch):
    from tests.test_model_refresh_workflow import setup
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
    from phase_agent.tools.local.prepare_model_refresh import RefreshUploadRunner

    current = RefreshUploadRunner.prepare
    calls = []

    def old(self, state):
        calls.append(True)
        return current(self, state)

    monkeypatch.setattr(RefreshUploadRunner, "prepare", old)
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    result = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert result["status"] == "prepared" and len(calls) == 2
    assert len(result["state"]["tasks"]) == 1
    handoff = result["state"]["slurm_batches"][0]["handoff"]
    from pathlib import Path

    assert all(
        (Path(handoff["folder"]) / name).is_file()
        for name in ("inputs.zip", "submit_all.sh", "dispatch_tasks.py")
    )

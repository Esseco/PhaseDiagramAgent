from copy import deepcopy
from types import SimpleNamespace

import phase_agent.tools.workflows.recover_saved_results as recovery


def test_returns_recovered_with_original_config_without_migration(
    tmp_path, monkeypatch
):
    result = tmp_path / "result.json"
    result.write_text("{}")
    state = {
        "confirmed_config_version": "old",
        "confirmed_config": {
            "system": {"system_id": "Co"},
            "budgets": {"max_tasks": 300},
        },
        "tasks": [
            {
                "task_id": "r1",
                "stage": "relax_and_feature",
                "status": "pending",
                "result_path": str(result),
            }
        ],
    }
    original = deepcopy(state)
    incoming = {
        "task_id": "r1",
        "task_key": "k1",
        "stage": "relax_and_feature",
        "status": "completed",
    }
    collector = SimpleNamespace(
        collect_results_with_report=lambda s: {
            "results": [incoming],
            "report": {"recovered_count": 1},
        }
    )
    seen = {}

    def feedback(current, results, **kwargs):
        seen["feedback"] = kwargs
        current["feedback_processed_task_ids"] = [row["task_id"] for row in results]
        return {"state": current, "rejected": []}

    def hull(current, *, config, path):
        seen["config"] = config
        return current

    monkeypatch.setattr(recovery, "apply_scientific_feedback", feedback)
    monkeypatch.setattr(recovery, "update_local_mlip_hull_pool", hull)
    monkeypatch.setattr(
        recovery, "_save_runner_state", lambda s, p: seen.update(saved=s)
    )
    recovered = recovery.recover_saved_results(
        state,
        manager=object(),
        result_collector=collector,
        config_session={"confirmed_snapshot": {"config_version": "new"}},
        state_path=tmp_path / "state.json",
    )
    assert state == original
    assert recovered["tasks"][0]["status"] == "completed"
    assert recovered["confirmed_config_version"] == "old"
    assert seen["config"]["config_version"] == "old"
    assert seen["config"]["budgets"] == {"max_tasks": 300}
    assert seen["saved"] == recovered
    assert recovered["processed_task_ids"] == ["r1"]
    collector.collect_results_with_report = lambda s: (_ for _ in ()).throw(
        AssertionError("duplicate collection")
    )
    assert (
        recovery.recover_saved_results(
            recovered, manager=object(), result_collector=collector
        )
        == recovered
    )


def test_invalid_return_exposes_report_without_settlement(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("{}")
    state = {
        "confirmed_config_version": "old",
        "confirmed_config": {"run": {}},
        "tasks": [
            {
                "status": "pending",
                "stage": "relax_and_feature",
                "result_path": str(result),
            }
        ],
    }
    collector = SimpleNamespace(
        collect_results_with_report=lambda s: {
            "results": [],
            "report": {
                "invalid_count": 1,
                "details_preview": [{"reason": "checksum_mismatch"}],
            },
        }
    )
    recovered = recovery.recover_saved_results(
        state, manager=object(), result_collector=collector
    )
    assert recovered["tasks"] == state["tasks"]
    assert recovered["result_collection"]["invalid_count"] == 1
    assert "result_collection" not in state


def test_dialogue_reads_returns_before_decision(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    write_json(path, {"tasks": [{"status": "pending"}]})
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            conversation_id="c",
            messages=[{"role": "user", "content": "回收完毕"}],
        )
    )
    monkeypatch.setattr(
        recovery,
        "recover_saved_results",
        lambda s, **kw: {"tasks": [{"status": "completed"}]},
    )
    assert read_project({}, runtime)["facts"]["tasks"][0]["status"] == "completed"


def test_migration_block_still_observes_returns(monkeypatch):
    from phase_agent.tools.workflows.lifecycle_initialization import (
        _initialize_workflow,
    )

    state = {
        "confirmed_config_version": "old",
        "confirmed_config": {"run": {}},
        "tasks": [{"status": "pending"}],
    }
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {"config_version": "new", "config": {"run": {}}},
    }
    monkeypatch.setattr(
        recovery,
        "recover_saved_results",
        lambda s, **kw: {**s, "tasks": [{"status": "completed"}]},
    )
    result = _initialize_workflow(None, {}, {}, session, state=state)
    assert result["reason"] == "approval_required"
    assert result["state"]["tasks"][0]["status"] == "completed"
    assert result["state"]["confirmed_config_version"] == "old"
    assert result["submitted"] is False


def test_batch_settlement_is_isolated_and_idempotent():
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
    from phase_agent.tools.budget.settle_budget import settle_budget

    state = {
        "budget_reservations": {
            f"k{i}": {"reserved_cost": 2, "stage": "relax_and_feature"}
            for i in range(20)
        },
        "reserved_relative_cost": 40,
    }
    original = deepcopy(state)
    results = [
        {
            "task_id": f"t{i}",
            "task_key": f"k{i}",
            "status": "completed",
            "actual_cost": 1,
        }
        for i in range(20)
    ]
    current = reconcile_task_results(state, results)["state"]
    assert state == original
    assert current["budget_usage"]["total_relative_cost"] == 20
    assert current["budget_usage"]["stages"]["relax_and_feature"]["tasks"] == 20
    assert current["reserved_relative_cost"] == 0
    assert reconcile_task_results(current, results)["state"] == current
    public = settle_budget(
        state, task_key="k0", settlement_id="s", task_status="completed", actual_cost=1
    )
    assert public["state"]["budget_usage"]["total_relative_cost"] == 1
    assert state == original


def test_interrupted_feedback_resumes_without_double_settlement(tmp_path, monkeypatch):
    import pytest

    result = tmp_path / "result.json"
    result.write_text("{}")
    incoming = {
        "task_id": "t",
        "task_key": "k",
        "stage": "relax_and_feature",
        "status": "completed",
        "actual_cost": 1,
    }
    state = {
        "confirmed_config_version": "old",
        "confirmed_config": {"run": {}},
        "tasks": [{**incoming, "status": "pending", "result_path": str(result)}],
        "budget_reservations": {"k": {"reserved_cost": 2}},
        "reserved_relative_cost": 2,
    }
    saved = []
    monkeypatch.setattr(
        recovery, "_save_runner_state", lambda s, p: saved.append(deepcopy(s))
    )
    monkeypatch.setattr(
        recovery,
        "apply_scientific_feedback",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    collector = SimpleNamespace(collect_results=lambda s: [incoming])
    with pytest.raises(RuntimeError, match="interrupted"):
        recovery.recover_saved_results(
            state, manager=object(), result_collector=collector
        )
    checkpoint = saved[-1]
    assert checkpoint["tasks"][0]["status"] == "completed"
    assert checkpoint["budget_usage"]["total_relative_cost"] == 1
    collector.collect_results = lambda s: []

    def feedback(s, rows, **kw):
        assert [r["task_id"] for r in rows] == ["t"]
        s["feedback_processed_task_ids"] = ["t"]
        return {"state": s, "rejected": []}

    monkeypatch.setattr(recovery, "apply_scientific_feedback", feedback)
    monkeypatch.setattr(recovery, "update_local_mlip_hull_pool", lambda s, **kw: s)
    resumed = recovery.recover_saved_results(
        checkpoint, manager=object(), result_collector=collector
    )
    assert resumed["budget_usage"]["total_relative_cost"] == 1
    assert resumed["feedback_processed_task_ids"] == ["t"]

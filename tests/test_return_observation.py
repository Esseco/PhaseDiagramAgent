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


def test_continue_with_active_dft_tasks_is_read_only_wait(tmp_path):
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "active_model_version": "m",
            "confirmed_config_version": "c",
            "execution_recovery_report": {"status": "clear", "unsettled": []},
            "tasks": [
                {
                    "task_id": "dft-1",
                    "stage": "dft_single_point",
                    "status": "pending",
                    "model_version": "m",
                    "config_version": "c",
                }
            ],
            "action_records": [
                {
                    "status": "prepared",
                    "agent_proposal": {
                        "raw_action": {"tool": "select_dft_candidates"}
                    },
                }
            ],
        },
    )
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            conversation_id="continue-wait",
            messages=[{"role": "user", "content": "继续"}],
        )
    )
    result = read_project({}, runtime)
    assert result["operation"] == "done"
    assert "DFT" in result["reply"] and "等待超算完成" in result["reply"]


def test_dialogue_requires_explicit_recovery_approval(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import write_json

    result = tmp_path / "result.json"
    result.write_text("{}")
    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "tasks": [
                {
                    "task_id": "returned",
                    "stage": "deep_search",
                    "status": "pending",
                    "result_path": str(result),
                }
            ]
        },
    )
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    calls = []

    def recover(state, **kwargs):
        calls.append(True)
        return {"tasks": [{"task_id": "returned", "status": "completed"}], "result_collection": {"recovered_count": 1}}

    monkeypatch.setattr(recovery, "recover_saved_results", recover)
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            conversation_id="c",
            messages=[{"role": "user", "content": "继续"}],
        )
    )
    prompt = read_project({}, runtime)
    assert "同意回收" in prompt["reply"] and not calls

    runtime.context.messages = [{"role": "user", "content": "同意回收"}]
    approved = read_project({}, runtime)
    assert calls == [True]
    assert approved["operation"] == "dispatch"
    assert approved["message"] == "继续"
    assert approved["recovery_just_approved"] is True


def test_recovery_gate_precedes_saved_plan(tmp_path, monkeypatch):
    """A stale saved proposal must not hide an unapproved recovery gate."""
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import write_json

    result = tmp_path / "result.json"
    result.write_text("{}")
    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "tasks": [
                {
                    "task_id": "returned",
                    "stage": "deep_search",
                    "status": "pending",
                    "result_path": str(result),
                }
            ],
            "pending_execution_policies": {
                "stale-plan": {
                    "agent_proposal": {
                        "raw_action": {"tool": "prepare_local_batch_files"}
                    }
                }
            },
        },
    )
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    monkeypatch.setattr(
        recovery,
        "recover_saved_results",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must wait for explicit approval")
        ),
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            conversation_id="c-gated",
            messages=[{"role": "user", "content": "继续"}],
        )
    )
    reply = read_project({}, runtime)
    assert "同意回收" in reply["reply"]
    assert "输入准备" not in reply["reply"]
    assert handler.last_turn_outcome == "answered"


def test_approved_recovery_replaces_stale_plan_with_new_planning_turn(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json

    result = tmp_path / "result.json"
    result.write_text("{}")
    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "tasks": [
                {
                    "task_id": "returned",
                    "stage": "deep_search",
                    "status": "pending",
                    "result_path": str(result),
                }
            ],
            "pending_execution_policies": {
                "stale-plan": {
                    "agent_proposal": {
                        "raw_action": {"tool": "prepare_local_batch_files"}
                    }
                }
            },
            "action_records": [{"record_id": "stale-plan", "status": "awaiting_approval"}],
        },
    )
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )

    def recover(state, **kwargs):
        return {
            **state,
            "tasks": [{"task_id": "returned", "stage": "deep_search", "status": "completed"}],
            "result_collection": {"recovered_count": 1},
        }

    monkeypatch.setattr(recovery, "recover_saved_results", recover)
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            conversation_id="c-approved",
            messages=[{"role": "user", "content": "同意回收"}],
        )
    )
    result_state = read_project({}, runtime)
    assert result_state["operation"] == "dispatch"
    assert result_state["recovery_just_approved"] is True
    saved = read_json(path, {})
    assert not saved.get("pending_execution_policies")
    assert saved["action_records"][0]["status"] == "superseded"


def test_acknowledged_return_is_not_recollected_on_every_continue(tmp_path, monkeypatch):
    import phase_agent.tools.workflows.recover_saved_results as recovery_module
    from phase_agent.tools.workflows.recover_saved_results import (
        recovery_candidate,
        recovery_candidate_fingerprint,
    )

    archive = tmp_path / "results.zip"
    archive.write_bytes(b"return")
    state = {
        "relax_mc_batches": {
            "batch": {
                "expected_results_archive": str(archive),
                "status": "prepared",
            }
        }
    }
    state["recovery_acknowledgement"] = {
        "fingerprint": recovery_candidate_fingerprint(state)
    }
    assert recovery_candidate(state)["pending"] is False
    state["relax_mc_batches"]["batch"]["expected_results_archive"] = str(
        tmp_path / "new-results.zip"
    )
    monkeypatch.setattr(recovery_module, "has_registered_return", lambda _state: True)
    assert recovery_candidate(state)["pending"] is True


def test_prepared_slurm_inputs_do_not_reopen_acknowledged_returns(tmp_path):
    from phase_agent.tools.workflows.recover_saved_results import (
        recovery_candidate,
        recovery_candidate_fingerprint,
    )

    archive = tmp_path / "results.zip"
    archive.write_bytes(b"return")
    state = {
        "relax_mc_batches": {
            "batch": {
                "expected_results_archive": str(archive),
                "status": "completed",
                "import_checksum": "old",
            }
        },
        "slurm_batches": [],
        "tasks": [],
        "execution_recovery_report": {"status": "clear", "unsettled": []},
        "recovery_acknowledgement": {
            "scope": "explicit_user_recovery_approval",
        },
    }
    state["recovery_acknowledgement"]["fingerprint"] = recovery_candidate_fingerprint(state)
    state["slurm_batches"].append(
        {
            "batch_id": "prepared-only",
            "status": "prepared",
            "results_directory": str(tmp_path / "empty-results"),
        }
    )
    assert recovery_candidate(state)["pending"] is False


def test_post_recovery_gate_supersedes_stale_plan_before_saved_control(tmp_path):
    from phase_agent.graphs.dialogue.nodes import route_request
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json

    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "post_recovery_next_step_pending": True,
            "pending_execution_policies": {
                "stale": {
                    "agent_proposal": {
                        "raw_action": {
                            "tool": "prepare_local_batch_files",
                            "parameters": {"mode": "relax_mc_inputs"},
                        }
                    }
                }
            },
            "action_records": [{"record_id": "stale", "status": "awaiting_approval"}],
        },
    )
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(handler=handler, conversation_id="c", messages=[])
    )

    result = route_request({"message": "开始", "facts": read_json(path, {})}, runtime)

    assert result["operation"] == "decide"
    saved = read_json(path, {})
    assert saved["pending_execution_policies"] == {}
    assert saved["action_records"][0]["status"] == "superseded"


def test_scientific_proposal_layer_blocks_relax_mc_after_recovery():
    from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

    result = select_initial_proposal(
        current={"post_recovery_next_step_pending": True},
        mode="interactive",
        config={"agent": {"allowed_tools": ["prepare_local_batch_files"]}},
        context={},
        decision_state={},
        registry={"prepare_local_batch_files": {"handler": object()}},
        agent_client=None,
        invocation_id="test",
        propose=None,
        revise=None,
        prepare_debug=None,
        is_verified_second=None,
        mc_block=None,
        model_failed=lambda _: False,
        record_id_factory=lambda *_: "record",
    )

    assert result["status"] == "not_configured"
    assert "未生成 Relax/MC 方案" in result["reason"]


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

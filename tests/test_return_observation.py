from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import phase_agent.tools.workflows.recover_saved_results as recovery


def test_returns_recovered_with_original_config_without_migration(tmp_path, monkeypatch):
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
    monkeypatch.setattr(recovery, "_save_runner_state", lambda s, p: seen.update(saved=s))
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
        recovery.recover_saved_results(recovered, manager=object(), result_collector=collector)
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
    recovered = recovery.recover_saved_results(state, manager=object(), result_collector=collector)
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
                    "agent_proposal": {"raw_action": {"tool": "select_dft_candidates"}},
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
        return {
            "tasks": [{"task_id": "returned", "status": "completed"}],
            "result_collection": {"recovered_count": 1},
        }

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
                    "agent_proposal": {"raw_action": {"tool": "prepare_local_batch_files"}}
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
                    "agent_proposal": {"raw_action": {"tool": "prepare_local_batch_files"}}
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
    state["recovery_acknowledgement"] = {"fingerprint": recovery_candidate_fingerprint(state)}
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


def test_new_dft_return_is_not_hidden_by_legacy_acknowledgement(tmp_path):
    from phase_agent.tools.workflows.recover_saved_results import (
        recovery_candidate,
        recovery_candidate_fingerprint,
    )

    input_dir = tmp_path / "DFTSP_Batch01_input"
    output_dir = tmp_path / "DFTSP_Batch01_output"
    input_dir.mkdir()
    output_dir.mkdir()
    state = {
        "slurm_batches": [
            {
                "batch_id": "remote-000001",
                "status": "prepared",
                "task_ids": ["DFT-1"],
                "handoff": {
                    "layout_version": 4,
                    "folder": str(input_dir),
                    "return_directory": str(output_dir),
                    "import_directory": str(tmp_path / "import"),
                    "upload_archive": str(input_dir / "inputs.zip"),
                },
            }
        ],
        "tasks": [
            {
                "task_id": "DFT-1",
                "stage": "dft_single_point",
                "status": "pending",
                "result_path": str(tmp_path / "missing-result.json"),
            }
        ],
        "execution_recovery_report": {"status": "clear", "unsettled": []},
        "recovery_acknowledgement": {"scope": "explicit_user_recovery_approval"},
    }
    state["recovery_acknowledgement"]["fingerprint"] = recovery_candidate_fingerprint(state)
    archive = output_dir / "calculations__G0001__results.results.zip"
    archive.write_bytes(b"new DFT return")

    candidate = recovery_candidate(state)

    assert candidate["returned"] is True
    assert candidate["pending"] is True


def test_preparing_and_submitting_training_does_not_reopen_old_returns(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import recover_saved_results as module

    monkeypatch.setattr(module, "has_registered_return", lambda _: True)
    state = {"tasks": [], "remote_finetune_jobs": {}}
    state["recovery_acknowledgement"] = {
        "scope": "explicit_user_recovery_approval",
        "fingerprint": module.recovery_candidate_fingerprint(state),
    }
    job = {
        "status": "inputs_prepared",
        "directory": str(tmp_path / "training"),
        "activated": False,
        "handoff": {"return_directory": str(tmp_path / "Finetune_Batch01_output")},
    }
    state["remote_finetune_jobs"]["new-training"] = job
    for status in ("inputs_prepared", "submitted", "running"):
        job["status"] = status
        assert module.recovery_candidate(state)["pending"] is False
    output = tmp_path / "Finetune_Batch01_output"
    output.mkdir()
    (output / "training.results.zip").write_bytes(b"new training output")
    assert module.recovery_candidate(state)["pending"] is True


def test_legacy_training_metadata_ack_is_accepted_until_actual_output_changes(
    tmp_path, monkeypatch
):
    from phase_agent.tools.workflows import recover_saved_results as module

    monkeypatch.setattr(module, "has_registered_return", lambda _: True)
    state = {
        "remote_finetune_jobs": {
            "training": {"directory": str(tmp_path / "training"), "status": "inputs_prepared"}
        }
    }
    state["recovery_acknowledgement"] = {
        "scope": "explicit_user_recovery_approval",
        "fingerprint": module.recovery_candidate_fingerprint(state, legacy_training_metadata=True),
    }
    assert module.recovery_candidate(state)["pending"] is False
    output = tmp_path / "training" / "results"
    output.mkdir(parents=True)
    (output / "training.finished.json").write_text("{}", encoding="utf-8")
    assert module.recovery_candidate(state)["pending"] is True


def test_recovery_candidate_reuses_read_only_cache_until_state_or_return_changes(
    tmp_path, monkeypatch
):
    from phase_agent.tools.workflows import recover_saved_results as module

    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    return_dir = tmp_path / "return"
    return_dir.mkdir()
    state = {
        "tasks": [],
        "slurm_batches": [{"handoff": {"return_directory": str(return_dir)}}],
    }
    calls = {"returns": 0, "fingerprint": 0}

    def count_returns(current):
        calls["returns"] += 1
        return False

    def count_fingerprint(current):
        calls["fingerprint"] += 1
        return "fingerprint"

    monkeypatch.setattr(module, "has_registered_return", count_returns)
    monkeypatch.setattr(module, "recovery_candidate_fingerprint", count_fingerprint)
    original_stat = Path.stat
    directory_stat = return_dir.stat()
    # Windows can retain a directory timestamp while a new archive is visible.
    monkeypatch.setattr(
        Path,
        "stat",
        lambda path, *args, **kwargs: (
            directory_stat if path == return_dir else original_stat(path, *args, **kwargs)
        ),
    )

    first = module.recovery_candidate(state, state_path=state_path)
    second = module.recovery_candidate(state, state_path=state_path)

    assert first == second
    assert calls == {"returns": 1, "fingerprint": 1}
    (return_dir / "new.results.zip").write_bytes(b"new")
    module.recovery_candidate(state, state_path=state_path)
    assert calls == {"returns": 2, "fingerprint": 2}


def test_recovery_cache_detects_a_new_pending_task_result_file(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import recover_saved_results as module

    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    result = tmp_path / "result.json"
    state = {
        "tasks": [
            {
                "task_id": "pending",
                "stage": "dft_single_point",
                "status": "pending",
                "result_path": str(result),
            }
        ]
    }
    calls = []

    def returns(_):
        calls.append(True)
        return result.is_file()

    monkeypatch.setattr(module, "has_registered_return", returns)
    monkeypatch.setattr(module, "recovery_candidate_fingerprint", lambda _: "fingerprint")
    assert not module.recovery_candidate(state, state_path=state_path)["returned"]
    result.write_text("{}", encoding="utf-8")
    assert module.recovery_candidate(state, state_path=state_path)["returned"]
    assert len(calls) == 2


def test_registered_return_scan_ignores_historical_batches_when_tasks_are_pending(
    tmp_path, monkeypatch
):
    from phase_agent.tools.remote import transfer_bundle as module

    old_dir = tmp_path / "old-output"
    current_dir = tmp_path / "current-output"
    old_dir.mkdir()
    current_dir.mkdir()
    seen = []

    def archives(root):
        seen.append(str(root))
        return [Path(root) / "result.results.zip"]

    monkeypatch.setattr(module, "_registered_archives", archives)
    monkeypatch.setattr(module, "_archive_receipt", lambda root, archive: Path(root) / "receipt")
    monkeypatch.setattr(module, "checksum", lambda archive: "digest")
    state = {
        "tasks": [{"task_id": "current", "stage": "dft_single_point", "status": "pending"}],
        "slurm_batches": [
            {"task_ids": ["old"], "results_directory": str(old_dir)},
            {"task_ids": ["current"], "results_directory": str(current_dir)},
        ],
    }

    assert module.has_registered_return(state) is True
    assert str(old_dir) not in seen
    assert str(current_dir) in seen


def test_recovery_intent_uses_bounded_semantic_classifier_for_natural_language():
    from phase_agent.graphs.dialogue.saved_control import semantic_recovery_intent

    seen = []

    def client(payload):
        seen.append(payload)
        return {"recovery_intent": "approve"}

    handler = SimpleNamespace(workflow_kwargs={"agent_client": client})
    intent = semantic_recovery_intent(handler, {"tasks": []}, "已回传，可以回收")

    assert intent == "approve"
    assert seen and seen[0]["bounded_control"] is True
    assert "execution" not in seen[0]["output_contracts"]


def test_standalone_recovery_imperative_is_explicit_approval():
    from phase_agent.graphs.dialogue.saved_control import semantic_recovery_intent

    handler = SimpleNamespace(
        workflow_kwargs={"agent_client": lambda _payload: (_ for _ in ()).throw(AssertionError())}
    )

    assert semantic_recovery_intent(handler, {"tasks": []}, "回收") == "approve"


def test_no_pending_recovery_does_not_add_an_intent_model_call():
    from phase_agent.graphs.dialogue.saved_control import semantic_recovery_intent

    calls = []
    handler = SimpleNamespace(workflow_kwargs={"agent_client": lambda p: calls.append(p)})
    assert (
        semantic_recovery_intent(handler, {"tasks": []}, "请解释当前状态", pending=False)
        == "unknown"
    )
    assert calls == []
    assert semantic_recovery_intent(handler, {"tasks": []}, "同意回收", pending=False) == "approve"
    assert calls == []


def test_natural_recovery_approval_enters_recovery_without_keyword_matching(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "tasks": [
                {
                    "task_id": "returned",
                    "stage": "dft_single_point",
                    "status": "pending",
                    "result_path": str(tmp_path / "result.json"),
                }
            ]
        },
    )
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        recovery,
        "recover_saved_results",
        lambda state, **kwargs: (
            calls.append(True)
            or {
                **state,
                "tasks": [
                    {"task_id": "returned", "stage": "dft_single_point", "status": "completed"}
                ],
            }
        ),
    )
    handler = SimpleNamespace(
        state_path=path,
        conversation_id=None,
        config_delegate=None,
        workflow_kwargs={"agent_client": lambda payload: {"recovery_intent": "approve"}},
        last_turn_outcome=None,
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler,
            conversation_id="semantic-recovery",
            messages=[{"role": "user", "content": "已回传，可以回收"}],
        )
    )

    result = read_project({}, runtime)

    assert calls == [True]
    assert result["operation"] == "dispatch"
    assert result["recovery_just_approved"] is True


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
            f"k{i}": {"reserved_cost": 2, "stage": "relax_and_feature"} for i in range(20)
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
    monkeypatch.setattr(recovery, "_save_runner_state", lambda s, p: saved.append(deepcopy(s)))
    monkeypatch.setattr(
        recovery,
        "apply_scientific_feedback",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    collector = SimpleNamespace(collect_results=lambda s: [incoming])
    with pytest.raises(RuntimeError, match="interrupted"):
        recovery.recover_saved_results(state, manager=object(), result_collector=collector)
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

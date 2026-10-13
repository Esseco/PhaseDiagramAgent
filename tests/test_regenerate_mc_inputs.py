import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.tools.local import regenerate_mc_inputs as module
from phase_agent.tools.remote.build_upload_batch_directory import build_upload_batch_directory
from phase_agent.runtime.agent_api import RunWorkflowChatHandler
from phase_agent.tools.budget.stratify_mc_actions import (
    interleave_mc_strata,
    summarize_mc_interception,
)


def _state(root):
    key = "tiered-mc:one"
    task_id = "MC-" + hashlib.sha256(key.encode()).hexdigest()[:12]
    mc_root = root / "MLIP-round-0001_model" / "MC-search"
    old_batch = mc_root / "MC-sampling-0001_remote-000002"
    old_input = old_batch / f"00000-{task_id}" / "task.json"
    source = root / "source.vasp"
    source.write_text("source", encoding="utf-8")
    child = {
        "task_key": key,
        "branch_id": "B-1",
        "structure_id": "S-1",
        "structure_path": str(source),
        "stage": "deep_search",
        "model_version": "model",
        "max_mc_steps": 30,
        "patience_steps": 4,
        "min_improvement": 0.001,
        "seed": 42,
        "planned_relative_cost": 50.0,
    }
    old = {
        **child,
        "task_id": task_id,
        "status": "pending",
        "input_path": str(old_input),
        "result_path": str(old_input.with_name("result.json")),
        "batch_id": "remote-000002",
    }
    return {
        "tasks": [old],
        "pending_tasks": [old],
        "slurm_batches": [
            {
                "batch_id": "remote-000001",
                "task_ids": ["RELAX-1"],
                "calculation_group": "Relax-screening",
                "status": "prepared",
            },
            {
                "batch_id": "remote-000002",
                "task_ids": [task_id],
                "calculation_group": "MC-search",
                "upload_directory": str(old_batch),
                "status": "prepared",
                "job_id": None,
            },
        ],
        "budget_reservations": {key: {"stage": "deep_search", "status": "reserved"}},
        "tiered_mc_state": {"segments": [child]},
        "action_records": [
            {
                "status": "completed",
                "human_feedback": {"decision": "approve"},
                "final_action": {
                    "tool": "allocate_mc_bohb",
                    "task_key": "approved-plan",
                    "parameters": {
                        "budget_preview": {
                            "allocations": [child],
                            "selected_branch_count": 1,
                            "allocation_checksum": "abc",
                            "estimated_relative_cost": 50.0,
                        }
                    },
                },
            }
        ],
    }


def test_missing_directory_regenerates_approved_plan(tmp_path, monkeypatch):
    state = _state(tmp_path)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["exists"] is False
    assert plan["mc_sampling_start"] == 1
    assert plan["remote_batch_start"] == 2
    assert plan["mc_batch_size"] == 10
    assert plan["estimated_batch_count"] == 1
    captured = {}

    def prepare(*, action, context):
        captured.update(context["event_state"])
        return {
            "status": "prepared",
            "task_count": 1,
            "batch_count": 1,
            "state": context["event_state"],
        }

    monkeypatch.setattr(module, "prepare_mc_upload_batches", prepare)
    result = module.regenerate_mc_inputs(
        state,
        approved_plan=plan,
        upload_root=tmp_path,
        config={},
        manager=None,
        phase_references={},
        config_version="v1",
    )
    assert result["task_count"] == 1
    assert len(captured["tasks"]) == 1
    assert captured["budget_reservations"]["tiered-mc:one"]["approval_override"] == "abc"
    assert [row["batch_id"] for row in captured["slurm_batches"]] == ["remote-000001"]
    assert captured["mc_regeneration_history"][0]["old_batch_ids"] == ["remote-000002"]
    directory, layout = build_upload_batch_directory(
        tmp_path, captured, batch_id="remote-000002", stage="deep_search", model_version="model"
    )
    assert directory.name == "MC-sampling-0001_remote-000002"
    assert layout["submission_index"] == 1


def test_existing_directory_requires_confirmation(tmp_path):
    state = _state(tmp_path)
    mc_root = Path(state["tasks"][0]["input_path"]).parents[2]
    mc_root.mkdir(parents=True)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["exists"] is True
    with pytest.raises(ValueError, match="确认"):
        module.regenerate_mc_inputs(
            state,
            approved_plan=plan,
            upload_root=tmp_path,
            config={},
            manager=None,
            phase_references={},
            config_version="v1",
        )


def test_result_file_blocks_regeneration(tmp_path):
    state = _state(tmp_path)
    result = Path(state["tasks"][0]["result_path"])
    result.parent.mkdir(parents=True)
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="结果"):
        module.plan_mc_regeneration(state, tmp_path)


def test_budget_limited_order_interleaves_tier_and_phase():
    actions = [
        {"task_key": "a", "branch_id": "A", "tier": "tier-0"},
        {"task_key": "b", "branch_id": "A", "tier": "tier-0"},
        {"task_key": "c", "branch_id": "B", "tier": "tier-0"},
        {"task_key": "d", "branch_id": "C", "tier": "tier-1"},
    ]
    phase = {"A": "P2", "B": "P3", "C": "P2"}
    ordered = interleave_mc_strata(actions, phase)
    assert [row["task_key"] for row in ordered] == ["a", "c", "d", "b"]
    summary = summarize_mc_interception(
        ordered[:2],
        [{"action": row, "reasons": ["stage_cost_limit"]} for row in ordered[2:]],
        phase,
        full_plan_approved=False,
    )
    assert summary["strata"]["tier-1/P2"]["rejected"] == 1
    assert summary["rejected_count"] == 2


@pytest.mark.parametrize("directory_exists", [False, True])
def test_chat_request_proposes_and_simple_approval_executes(
    tmp_path, monkeypatch, directory_exists
):
    state = _state(tmp_path)
    if directory_exists:
        Path(state["tasks"][0]["input_path"]).parents[2].mkdir(parents=True)
    state["confirmed_config_version"] = "v1"
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    config = {
        "upload_batches_directory": str(tmp_path),
        "budgets": {"stage_limits": {"deep_search": {"max_cost": 100}}},
    }
    monkeypatch.setattr(
        "phase_agent.configuration.runtime.build_effective_run_config.build_effective_run_config",
        lambda *_: config,
    )
    calls = []
    monkeypatch.setattr(
        "phase_agent.tools.local.replan_mc_allocation.replan_mc_allocation",
        lambda *_: module._approved_mc_preview(state)[0],
    )

    def regenerate(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "prepared", "state": args[0], "task_count": 1, "batch_count": 1}

    monkeypatch.setattr(module, "regenerate_mc_inputs", regenerate)
    handler = RunWorkflowChatHandler(
        {
            "state_path": str(state_path),
            "config_session": {},
            "run_config": {},
            "manager": None,
            "phase_references": {},
        }
    )
    reply = handler([{"role": "user", "content": "重新生成当前轮MC任务"}])
    assert "尚未删除文件或生成任务" in reply
    assert ("该目录存在" if directory_exists else "该目录不存在") in reply
    assert "MC-sampling 从 0001" in reply
    assert "重新规划" in reply
    assert calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["pending_mc_regeneration"]
    confirmed = handler([{"role": "user", "content": "同意"}])
    assert "生成 1 个 MC 输入" in confirmed
    assert len(calls) == 1


def _legacy_state(root):
    state = _state(root)
    record = state["action_records"][0]
    children = record["final_action"]["parameters"].pop("budget_preview")["allocations"]
    record["final_action"].update(target_ids=["B-1"], budget=50.0)
    record["final_action"]["parameters"]["mc_budget"] = 30
    record["execution_result"] = {
        "status": "completed",
        "result": {"status": "completed", "actions": children},
    }
    return state


def test_legacy_approved_execution_allocations_are_reused_without_replanning(tmp_path, monkeypatch):
    state = _legacy_state(tmp_path)
    original = json.dumps(state, sort_keys=True)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["allocation_count"] == 1
    assert plan["total_mc_steps"] == 30
    assert plan["estimated_relative_cost"] == 50.0
    assert plan["allocation_source"] == "approved_execution_record"
    captured = {}

    def prepare(*, action, context):
        captured.update(context["event_state"])
        return {
            "status": "prepared",
            "task_count": 1,
            "batch_count": 1,
            "state": context["event_state"],
        }

    monkeypatch.setattr(module, "prepare_mc_upload_batches", prepare)
    module.regenerate_mc_inputs(
        state,
        approved_plan=plan,
        upload_root=tmp_path,
        config={},
        manager=None,
        phase_references={},
        config_version="v1",
    )
    assert captured["tasks"][0]["max_mc_steps"] == 30
    assert json.dumps(state, sort_keys=True) == original


@pytest.mark.parametrize(
    "field,value", [("max_mc_steps", 31), ("planned_relative_cost", 51.0), ("branch_id", "other")]
)
def test_legacy_allocation_cannot_exceed_or_mismatch_approved_scope(tmp_path, field, value):
    state = _legacy_state(tmp_path)
    from copy import deepcopy

    state["action_records"][0]["execution_result"] = deepcopy(
        state["action_records"][0]["execution_result"]
    )
    state["action_records"][0]["execution_result"]["result"]["actions"][0][field] = value
    with pytest.raises(ValueError):
        module.plan_mc_regeneration(state, tmp_path)


def test_absent_legacy_receipt_and_partial_new_preview_are_not_guessed(tmp_path):
    state = _legacy_state(tmp_path)
    state["action_records"][0].pop("execution_result")
    with pytest.raises(ValueError):
        module.plan_mc_regeneration(state, tmp_path)
    state = _legacy_state(tmp_path)
    state["action_records"][0]["final_action"]["parameters"]["budget_preview"] = {
        "selected_branch_count": 2
    }
    with pytest.raises(ValueError, match="不完整"):
        module.plan_mc_regeneration(state, tmp_path)


def test_numbered_mc_round_inputs_layout_and_input_archive_are_recognized(tmp_path):
    state = _legacy_state(tmp_path)
    root = tmp_path / "epoch0_model/Search-group-0001/Relax-0001_MC-round-0001"
    batch = root / "inputs/MC-sampling-0001_remote-000002"
    task_root = batch / "00000-MC-1"
    task_root.mkdir(parents=True)
    (task_root / "task.json").write_text("{}")
    (batch / "transfer_bundle.py").write_text("adapter")
    (batch / "_handoff_mlip_batch.py").write_text("adapter")
    archive = batch.with_name(batch.name + ".upload.zip")
    archive.write_bytes(b"existing input archive")
    task = state["tasks"][0]
    task["input_path"] = str(task_root / "task.json")
    task["result_path"] = str(root / "results/00000-MC-1/result.json")
    state["slurm_batches"][1]["upload_directory"] = str(batch)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["directory"] == str(root.resolve())
    assert plan["exists"]
    (root / "results").mkdir()
    returned = Path(task["result_path"])
    returned.parent.mkdir(parents=True)
    returned.write_text("actual MC return")
    with pytest.raises(ValueError, match="结果|未知"):
        module.plan_mc_regeneration(state, tmp_path)


def _mixed_return_state(tmp_path):
    state = _legacy_state(tmp_path)
    root = tmp_path / "epoch0_model/Search-group-0001/Relax-0001_MC-round-0001"
    batch = root / "inputs/MC-sampling-0001_remote-000002"
    task_root = batch / "00000-MC-1"
    task_root.mkdir(parents=True)
    (task_root / "task.json").write_bytes(b"old input")
    archive = batch.with_name(batch.name + ".upload.zip")
    archive.write_bytes(b"old upload archive")
    results = root / "results"
    (results / "00000-RELAX-old").mkdir(parents=True)
    (results / "00000-RELAX-old/result.json").write_bytes(b"returned Relax evidence")
    (results / "Relax-submission-0001_remote-000001.runtime.json").write_bytes(b"Relax timing")
    (results / "other.log").write_bytes(b"preserve any result-side evidence")
    state["tasks"][0]["input_path"] = str(task_root / "task.json")
    state["tasks"][0]["result_path"] = str(results / "00000-MC-1/result.json")
    state["slurm_batches"][1]["upload_directory"] = str(batch)
    return state, root, batch, archive


def test_mixed_relax_returns_do_not_block_input_only_regeneration(tmp_path):
    state, root, batch, archive = _mixed_return_state(tmp_path)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["directory"] == str(root.resolve())
    assert plan["input_directories"] == [str(batch.resolve())]
    assert plan["input_archives"] == [str(archive.resolve())]


def test_registered_compact_calculation_path_can_be_regenerated(tmp_path):
    state, root, batch, archive = _mixed_return_state(tmp_path)
    compact = tmp_path / "epoch0_model/G0001_Relax-0001_MC-round-0001"
    root.rename(compact)
    task = state["tasks"][0]
    task["input_path"] = str(compact / Path(task["input_path"]).relative_to(root))
    task["result_path"] = str(compact / Path(task["result_path"]).relative_to(root))
    state["slurm_batches"][1]["upload_directory"] = str(compact / batch.relative_to(root))
    state["slurm_batches"][1]["operation_directory"] = str(compact)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["directory"] == str(compact.resolve())
    state["slurm_batches"][1]["operation_directory"] = str(tmp_path / "unregistered")
    with pytest.raises(ValueError, match="上传根目录"):
        module.plan_mc_regeneration(state, tmp_path)


def test_regeneration_backs_up_and_refreshes_complete_input_archive(tmp_path, monkeypatch):
    from phase_agent.tools.remote.submission_storage import pack_storage_directory

    state, root, batch, archive = _mixed_return_state(tmp_path)
    pack_storage_directory(root / "inputs", "inputs")
    complete = root / "inputs/all.inputs.zip"
    original = complete.read_bytes()
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert str(complete) in plan["input_archives"]

    def prepare(*, action, context):
        assert not complete.exists()
        batch.mkdir(parents=True)
        (batch / "task.json").write_bytes(b"new input")
        return {
            "status": "prepared",
            "state": context["event_state"],
            "task_count": 1,
            "batch_count": 1,
        }

    monkeypatch.setattr(module, "prepare_mc_upload_batches", prepare)
    result = module.regenerate_mc_inputs(
        state,
        approved_plan=plan,
        upload_root=tmp_path,
        config={},
        manager=None,
        phase_references={},
        config_version="v1",
        allow_delete=True,
    )
    backup = Path(result["state"]["mc_regeneration_history"][-1]["input_backup_directory"])
    assert (backup / "inputs/all.inputs.zip").read_bytes() == original
    assert complete.is_file() and complete.read_bytes() != original
    import zipfile

    with zipfile.ZipFile(complete) as zipped:
        assert zipped.read(batch.name + "/task.json") == b"new input"


def test_regeneration_archives_inputs_and_preserves_all_returned_files(tmp_path, monkeypatch):
    state, root, batch, archive = _mixed_return_state(tmp_path)
    results = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in (root / "results").rglob("*")
        if p.is_file()
    }
    plan = module.plan_mc_regeneration(state, tmp_path)

    def prepare(*, action, context):
        assert not batch.exists() and not archive.exists()
        batch.mkdir(parents=True)
        (batch / "task.json").write_bytes(b"new input")
        return {
            "status": "prepared",
            "state": context["event_state"],
            "task_count": 1,
            "batch_count": 1,
        }

    monkeypatch.setattr(module, "prepare_mc_upload_batches", prepare)
    outcome = module.regenerate_mc_inputs(
        state,
        approved_plan=plan,
        upload_root=tmp_path,
        config={},
        manager=None,
        phase_references={},
        config_version="v1",
        allow_delete=True,
    )
    history = outcome["state"]["mc_regeneration_history"][-1]
    backup = Path(history["input_backup_directory"])
    assert (backup / batch.relative_to(root) / "00000-MC-1/task.json").read_bytes() == b"old input"
    assert (backup / archive.relative_to(root)).read_bytes() == b"old upload archive"
    assert (batch / "task.json").read_bytes() == b"new input"
    assert results == {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in (root / "results").rglob("*")
        if p.is_file()
    }


def test_failed_regeneration_restores_old_inputs_and_keeps_returned_results(tmp_path, monkeypatch):
    state, root, batch, archive = _mixed_return_state(tmp_path)
    before = json.dumps(state, sort_keys=True)
    plan = module.plan_mc_regeneration(state, tmp_path)

    def fail(*, action, context):
        batch.mkdir(parents=True)
        (batch / "partial.json").write_bytes(b"failed new input")
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(module, "prepare_mc_upload_batches", fail)
    with pytest.raises(RuntimeError, match="preparation failed"):
        module.regenerate_mc_inputs(
            state,
            approved_plan=plan,
            upload_root=tmp_path,
            config={},
            manager=None,
            phase_references={},
            config_version="v1",
            allow_delete=True,
        )
    assert (batch / "00000-MC-1/task.json").read_bytes() == b"old input"
    assert archive.read_bytes() == b"old upload archive"
    assert (root / "results/00000-RELAX-old/result.json").read_bytes() == b"returned Relax evidence"
    assert json.dumps(state, sort_keys=True) == before
    assert list((root.parent / ".mc_input_history").rglob("partial.json"))


def test_actual_mc_result_archive_and_task_local_result_block_regeneration(tmp_path):
    state, root, batch, archive = _mixed_return_state(tmp_path)
    returned = root / "results" / (batch.name + ".results.zip")
    returned.write_bytes(b"actual MC archive")
    with pytest.raises(ValueError, match="结果"):
        module.plan_mc_regeneration(state, tmp_path)
    returned.unlink()
    task_result = Path(state["tasks"][0]["input_path"]).with_name("result.json")
    task_result.write_text("actual task-local result")
    with pytest.raises(ValueError, match="结果|未知"):
        module.plan_mc_regeneration(state, tmp_path)


def test_current_wave_partial_preparation_failure_ignores_old_history(tmp_path):
    state = _state(tmp_path)
    state["mc_regeneration_history"] = [
        {"replacement_preview": {"allocations": [{"task_key": "obsolete-wave"}]}}
    ]
    record = state["action_records"][0]
    record["status"] = "not_configured"
    record["execution_result"] = {"result": {"actions": [state["tasks"][0]]}}
    directory = Path(state["tasks"][0]["input_path"]).parent.parent
    directory.mkdir(parents=True)
    (directory / "_gpu_accounting.py").write_text("# timing")
    (directory / "_gpu_transfer_bundle.py").write_text("# bundle")
    assert module.plan_mc_regeneration(state, tmp_path)["allocation_count"] == 1
    state["budget_reservations"][state["tasks"][0]["task_key"]]["status"] = "released"
    with pytest.raises(ValueError, match="预算"):
        module.plan_mc_regeneration(state, tmp_path)


def test_replacement_rebinds_second_wave_receipt(tmp_path, monkeypatch):
    from copy import deepcopy

    state = _state(tmp_path)
    old_id = state["tasks"][0]["task_id"]
    receipt = {"task_ids": [old_id], "allocation_checksum": "abc", "source_checksum": "first-wave"}
    state["mc_second_round_allocations"] = [receipt]
    state["mc_second_round_allocation"] = receipt
    preview = deepcopy(state["action_records"][0]["final_action"]["parameters"]["budget_preview"])
    preview["allocations"][0]["task_key"] = "tiered-mc:new"
    preview["allocation_checksum"] = "new-checksum"
    plan = module.plan_mc_regeneration(state, tmp_path, replacement_preview=preview)
    monkeypatch.setattr(
        module,
        "prepare_mc_upload_batches",
        lambda **kw: {"status": "prepared", "task_count": 1, "state": kw["context"]["event_state"]},
    )
    result = module.regenerate_mc_inputs(
        state,
        approved_plan=plan,
        upload_root=tmp_path,
        config={},
        manager=None,
        phase_references={},
        config_version="new",
        replacement_preview=preview,
    )
    current = result["state"]
    new_id = current["tasks"][0]["task_id"]
    assert new_id != old_id
    assert current["mc_second_round_allocations"][0]["task_ids"] == [new_id]
    assert current["mc_second_round_allocation"]["allocation_checksum"] == "new-checksum"
    assert current["mc_second_round_allocation"]["source_checksum"] == "first-wave"
    assert state["tasks"][0]["task_id"] == old_id

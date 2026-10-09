import hashlib
import json
from pathlib import Path

import pytest

from execution_layer.local import regenerate_mc_inputs as module
from execution_layer.remote.build_upload_batch_directory import build_upload_batch_directory
from run.agent_api import RunWorkflowChatHandler
from execution_layer.budget.stratify_mc_actions import (
    interleave_mc_strata, summarize_mc_interception,
)


def _state(root):
    key = "tiered-mc:one"
    task_id = "MC-" + hashlib.sha256(key.encode()).hexdigest()[:12]
    mc_root = root / "MLIP-round-0001_model" / "MC-search"
    old_batch = mc_root / "MC-sampling-0001_remote-000002"
    old_input = old_batch / f"00000-{task_id}" / "task.json"
    source = root / "source.vasp"
    source.write_text("source", encoding="utf-8")
    child = {"task_key": key, "branch_id": "B-1", "structure_id": "S-1",
             "structure_path": str(source), "stage": "deep_search", "model_version": "model",
             "max_mc_steps": 30, "patience_steps": 4, "min_improvement": 0.001,
             "seed": 42, "planned_relative_cost": 50.0}
    old = {**child, "task_id": task_id, "status": "pending", "input_path": str(old_input),
           "result_path": str(old_input.with_name("result.json")), "batch_id": "remote-000002"}
    return {"tasks": [old], "pending_tasks": [old], "slurm_batches": [
                {"batch_id": "remote-000001", "task_ids": ["RELAX-1"],
                 "calculation_group": "Relax-screening", "status": "prepared"},
                {"batch_id": "remote-000002", "task_ids": [task_id],
                 "calculation_group": "MC-search", "upload_directory": str(old_batch),
                 "status": "prepared", "job_id": None}],
            "budget_reservations": {key: {"stage": "deep_search", "status": "reserved"}},
            "tiered_mc_state": {"segments": [child]},
            "action_records": [{"status": "completed", "human_feedback": {"decision": "approve"},
                "final_action": {"tool": "allocate_mc_bohb", "task_key": "approved-plan",
                    "parameters": {"budget_preview": {"allocations": [child],
                        "selected_branch_count": 1, "allocation_checksum": "abc",
                        "estimated_relative_cost": 50.0}}}}]}


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
        return {"status": "prepared", "task_count": 1, "batch_count": 1,
                "state": context["event_state"]}

    monkeypatch.setattr(module, "prepare_mc_upload_batches", prepare)
    result = module.regenerate_mc_inputs(state, approved_plan=plan, upload_root=tmp_path,
        config={}, manager=None, phase_references={}, config_version="v1")
    assert result["task_count"] == 1
    assert len(captured["tasks"]) == 1
    assert captured["budget_reservations"]["tiered-mc:one"]["approval_override"] == "abc"
    assert [row["batch_id"] for row in captured["slurm_batches"]] == ["remote-000001"]
    assert captured["mc_regeneration_history"][0]["old_batch_ids"] == ["remote-000002"]
    directory, layout = build_upload_batch_directory(tmp_path, captured,
        batch_id="remote-000002", stage="deep_search", model_version="model")
    assert directory.name == "MC-sampling-0001_remote-000002"
    assert layout["submission_index"] == 1


def test_existing_directory_requires_confirmation(tmp_path):
    state = _state(tmp_path)
    mc_root = Path(state["tasks"][0]["input_path"]).parents[2]
    mc_root.mkdir(parents=True)
    plan = module.plan_mc_regeneration(state, tmp_path)
    assert plan["exists"] is True
    with pytest.raises(ValueError, match="确认"):
        module.regenerate_mc_inputs(state, approved_plan=plan, upload_root=tmp_path,
            config={}, manager=None, phase_references={}, config_version="v1")


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
    summary = summarize_mc_interception(ordered[:2],
        [{"action": row, "reasons": ["stage_cost_limit"]} for row in ordered[2:]],
        phase, full_plan_approved=False)
    assert summary["strata"]["tier-1/P2"]["rejected"] == 1
    assert summary["rejected_count"] == 2


@pytest.mark.parametrize("directory_exists", [False, True])
def test_chat_request_only_proposes_and_dedicated_confirmation_executes(
        tmp_path, monkeypatch, directory_exists):
    state = _state(tmp_path)
    if directory_exists:
        Path(state["tasks"][0]["input_path"]).parents[2].mkdir(parents=True)
    state["confirmed_config_version"] = "v1"
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    config = {"upload_batches_directory": str(tmp_path),
              "budgets": {"stage_limits": {"deep_search": {"max_cost": 10}}}}
    monkeypatch.setattr(
        "config_layer.runtime.build_effective_run_config.build_effective_run_config",
        lambda *_: config)
    calls = []

    def regenerate(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "prepared", "state": args[0], "task_count": 1, "batch_count": 1}

    monkeypatch.setattr(module, "regenerate_mc_inputs", regenerate)
    handler = RunWorkflowChatHandler({"state_path": str(state_path), "config_session": {},
        "run_config": {}, "manager": None, "phase_references": {}})
    reply = handler([{"role": "user", "content": "重新生成当前轮MC任务"}])
    assert "尚未删除文件或生成任务" in reply
    assert ("该目录存在" if directory_exists else "该目录不存在") in reply
    assert "MC-sampling 从 0001" in reply
    assert "高于当前 MC 阶段成本上限" in reply
    assert calls == []
    assert json.loads(state_path.read_text(encoding="utf-8"))["pending_mc_regeneration"]
    handler([{"role": "user", "content": "同意"}])
    assert calls == []
    confirmed = handler([{"role": "user", "content": "确认重新生成当前轮MC任务"}])
    assert "重新生成 1 个 MC 输入" in confirmed
    assert len(calls) == 1

from copy import deepcopy
from pathlib import Path
import pytest
from execution_layer.local.regenerate_finetune_inputs import (
    is_finetune_regeneration_request, plan_finetune_regeneration, regenerate_finetune_inputs)


def fixture(tmp_path):
    from tests.test_epoch_output_layout import add_result
    from config_layer.defaults.default_mace_committee_config import default_mace_committee_config
    from execution_layer.workflows.prepare_remote_finetune import prepare_remote_finetune
    state = {}
    for index in range(10):
        add_result(state, task_id=f"T{index}")
    settings = default_mace_committee_config()
    settings["labels"]["include_stress"] = False
    config = {"mlip_finetune": settings, "upload_batches_directory": str(tmp_path),
              "mlip": {"version": "m1", "model_path": "/remote/m1.model"},
              "python_environments": {"remote_mlip": "mace"}}
    prepared = prepare_remote_finetune(state, config)
    settings["training"]["patience"] = 19
    blocked = prepare_remote_finetune(prepared["state"], config)
    assert blocked["status"] == "confirmation_required"
    return blocked["state"], config


def test_context_plan_read_only_and_same_round_replacement(tmp_path):
    state, config = fixture(tmp_path)
    original = deepcopy(state)
    assert is_finetune_regeneration_request("重新生成原轮输入", state)
    assert not is_finetune_regeneration_request("重新生成当前轮MC", state)
    plan = plan_finetune_regeneration(state, config)
    assert state == original
    old = Path(plan["directory"])
    result = regenerate_finetune_inputs(state, config, plan)
    assert result["status"] == "awaiting_remote_training"
    assert old.is_dir() and old.name == "MLIP-finetune-round-0001"
    assert not (old.parent / "MLIP-finetune-round-0002").exists()
    assert list(old.parent.glob("*.inputs-backup-*"))
    assert state == original


def test_outputs_and_submission_block_replacement(tmp_path):
    state, config = fixture(tmp_path)
    directory = Path(state["finetune_input_conflict"]["directory"])
    (directory / "results").mkdir(exist_ok=True)
    (directory / "results" / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="结果"):
        plan_finetune_regeneration(state, config)
    state["remote_finetune_jobs"][state["finetune_input_conflict"]["job_key"]]["submitted"] = True
    with pytest.raises(ValueError, match="提交"):
        plan_finetune_regeneration(state, config)


def test_changed_data_invalidates_confirmation(tmp_path):
    state, config = fixture(tmp_path)
    plan = plan_finetune_regeneration(state, config)
    state.setdefault("new_dft_records", []).append({"data_id": "changed"})
    with pytest.raises(ValueError, match="已改变"):
        regenerate_finetune_inputs(state, config, plan)
    assert Path(plan["directory"]).is_dir()


def test_failed_generation_restores_original(tmp_path, monkeypatch):
    state, config = fixture(tmp_path)
    plan = plan_finetune_regeneration(state, config)
    original = (Path(plan["directory"]) / "inputs/GPU.sh").read_bytes()
    def fail(*args, **kwargs):
        raise ValueError("test failure")
    monkeypatch.setattr("execution_layer.workflows.prepare_remote_finetune.prepare_remote_finetune", fail)
    with pytest.raises(ValueError, match="test failure"):
        regenerate_finetune_inputs(state, config, plan)
    assert (Path(plan["directory"]) / "inputs/GPU.sh").read_bytes() == original


def test_chat_routes_implicit_request_to_training_plan_not_branch(tmp_path, monkeypatch):
    import json
    from run.agent_api import RunWorkflowChatHandler
    state, config = fixture(tmp_path)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("config_layer.runtime.build_effective_run_config.build_effective_run_config",
                        lambda *args: config)
    handler = RunWorkflowChatHandler({"state_path": str(state_path), "config_session": {}, "run_config": {}},
        workflow=lambda **kwargs: pytest.fail("planning must not advance workflow"))
    reply = handler([{"role": "user", "content": "重新生成原轮输入"}])
    assert "微调原轮输入" in reply
    assert "MLIP-finetune-round-0001" in reply
    assert "确认重新生成原轮微调输入" in reply
    assert "候选：" not in reply
    assert json.loads(state_path.read_text(encoding="utf-8"))["pending_finetune_regeneration"]
    reply = handler([{"role": "user", "content": "确认重新生成原轮微调输入"}])
    assert "原轮编号不变" in reply
    assert not list(tmp_path.rglob("MLIP-finetune-round-0002"))


def test_missing_directory_can_be_regenerated_without_new_number(tmp_path):
    state, config = fixture(tmp_path)
    directory = Path(state["finetune_input_conflict"]["directory"])
    directory.rename(directory.with_name(directory.name + ".manually-removed"))
    plan = plan_finetune_regeneration(state, config)
    assert not plan["exists"]
    result = regenerate_finetune_inputs(state, config, plan)
    assert result["status"] == "awaiting_remote_training"
    assert directory.is_dir()

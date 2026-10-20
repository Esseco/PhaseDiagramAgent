from copy import deepcopy

import pytest

from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.runtime.status_presentation import format_progress
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.chat_state_presentation import format_epoch_reply
from phase_agent.runtime.response_preferences import response_detail
from tests.test_dft_comparison_csv import add_result


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    "status,tool",
    [
        ("paused", None),
        ("failed", None),
        ("rejected_by_user", None),
        ("awaiting_remote_training", None),
        ("completed", "generate_branches"),
        ("completed", "check_convergence"),
        ("completed", None),
    ],
)
def test_history_alone_never_becomes_this_turn_scientific_summary(tmp_path, status, tool, verbose):
    state = {"active_model_version": "new"}
    add_result(state, version="old")
    original = deepcopy(state)
    result = {"status": status, "state": state, "reason": "本轮原因"}
    if tool:
        result["final_action"] = {"tool": tool, "task_key": "current"}
    with response_detail("detailed" if verbose else "brief"):
        raw = format_workflow_reply(result, tmp_path / "state.json", verbose=verbose)
        text = format_epoch_reply(raw if verbose else format_turn_reply(raw, state), state)
    assert "磁矩：" not in text and "本轮 DFT：" not in text
    assert state == original


def test_actual_dft_recovery_keeps_metrics_and_only_matching_round_magnetism(tmp_path):
    state = {"active_model_version": "new"}
    add_result(state, version="old", task_id="old", operation="old")
    add_result(state, version="new", task_id="current", operation="new")
    original = deepcopy(state)
    text = format_workflow_reply(
        {"status": "completed", "recovered_count": 1, "state": state},
        tmp_path / "state.json",
        verbose=True,
    )
    assert "本轮 DFT：回收 1/1" in text
    assert "磁矩：已回收 1 个结果" in text
    assert "能量 MAE/RMSE" in text and "受力" in text
    assert state == original


@pytest.mark.parametrize("detail", ["brief", "detailed"])
def test_training_status_uses_registered_paired_handoff(tmp_path, detail):
    folder = tmp_path / "epoch0_Finetune_Batch01_input"
    output = tmp_path / "epoch0_Finetune_Batch01_output"
    folder.mkdir()
    for name in ("submit_all.sh", "dispatch_tasks.py", "inputs.zip"):
        (folder / name).write_text("existing artifact")
    job = {
        "status": "inputs_prepared",
        "original_model_version": "m1",
        "directory": str(tmp_path / "training"),
        "handoff": {
            "folder": str(folder),
            "return_directory": str(output),
            "layout_version": 4,
            "job_count": 1,
            "upload_archive": str(folder / "inputs.zip"),
            "submit_script": str(folder / "submit_all.sh"),
        },
    }
    state = {"active_model_version": "m1", "remote_finetune_jobs": {"job": job}}
    original = deepcopy(state)
    with response_detail(detail):
        for raw in (
            format_progress(state),
            format_workflow_reply(
                {"status": "awaiting_remote_training", "state": state}, tmp_path / "state.json"
            ),
        ):
            text = format_epoch_reply(
                raw if detail == "detailed" else format_turn_reply(raw, state), state
            )
            assert str(folder) in text and str(output) in text
            assert "sbatch submit_all.sh" in text
            assert "GPU.sh" not in text and "return-results" not in text
            assert "results/transfer" not in text
    assert state == original


def test_complete_reply_chain_preserves_nonidentical_blocker():
    state = {
        "pending_execution_policies": {
            "p": {"agent_proposal": {"raw_action": {"tool": "generate_branches"}}}
        }
    }
    text = format_epoch_reply(
        format_turn_reply("当前：执行记录与占位不一致，本轮未执行。", state), state
    )
    assert "执行记录与占位不一致" in text and "本轮未执行" in text


@pytest.mark.parametrize("detail", ["brief", "detailed"])
def test_current_region_approval_retains_boundaries_without_count_targets(tmp_path, detail):
    from tests.test_region_sampling import params

    fields = params()
    fields["max_det_H"] = 12
    fields["regions"][0].update(na_min=0.25, na_max=0.75, min_det_H=2, max_det_H=6)
    state = {"confirmed_config": {"storage": {"workspace_root": str(tmp_path)}}}
    result = {
        "status": "awaiting_approval",
        "state": state,
        "agent_proposal": {"raw_action": {"tool": "generate_branches", "parameters": fields}},
    }
    before = deepcopy(result)
    with response_detail(detail):
        text = format_epoch_reply(format_turn_reply(format_workflow_reply(result, None), {}), {})
    assert "0.25–0.75" in text and "det(H) 2–6" in text and "det(H) ≤ 12" in text
    assert "新增数量生成后统计" in text and "优先级 1" in text
    assert result == before


def test_invalid_review_is_not_preserved_as_valid_evidence():
    from phase_agent.decisions.agent.post_dft_review import request_review_repair
    from tests.test_post_dft_review import review
    from tests.test_region_sampling import params

    action = {
        "tool": "generate_branches",
        "parameters": params(),
        "post_dft_review": {**review(), "choice": "finetune"},
    }
    with pytest.raises(ValueError) as caught:
        request_review_repair(lambda *_: pytest.fail("must not call model"), {}, action)
    assert not getattr(caught.value, "post_dft_review", None)


def test_reply_chain_is_watched_and_held_old_function_is_reported(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from phase_agent.runtime import loaded_code

    names = {
        "phase_agent.runtime." + n
        for n in (
            "workflow_reply_presentation",
            "post_dft_presentation",
            "status_presentation",
            "concise_dialogue",
            "proposal_presentation",
            "chat_state_presentation",
            "studio_reply_presentation",
            "waiting_files_presentation",
            "training_reply",
        )
    }
    assert names <= set(loaded_code.WATCHED_MODULES)
    name = "phase_agent.runtime.workflow_reply_presentation"
    path = tmp_path.joinpath(*name.split(".")).with_suffix(".py")
    path.parent.mkdir(parents=True)
    path.write_text("def format_workflow_reply():\n return 'old'\n")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    held = module.format_workflow_reply
    path.write_text("def format_workflow_reply():\n return 'updated version'\n")
    report = loaded_code.loaded_code_report(source_root=tmp_path)
    row = next(r for r in report["modules"] if r["module"] == name)
    assert row["status"] == "source_changed"
    assert "format_workflow_reply" in row["changed_functions"]
    assert held() == "old" and module.format_workflow_reply is held


def test_loaded_old_diagnostic_detects_its_own_source_change(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from phase_agent.runtime import loaded_code

    current = Path(loaded_code.__file__).read_text(encoding="utf-8")
    old = current.replace('        "watched_module_count": len(module_names),\n', "")
    assert old != current
    name = "phase_agent.runtime.loaded_code"
    path = tmp_path.joinpath(*name.split(".")).with_suffix(".py")
    path.parent.mkdir(parents=True)
    path.write_text(old, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    held = module.loaded_code_report
    path.write_text(current, encoding="utf-8")
    report = held(source_root=tmp_path)
    row = next(r for r in report["modules"] if r["module"] == name)
    assert row["status"] == "source_changed"
    assert "loaded_code_report" in row["changed_functions"]
    assert module.loaded_code_report is held

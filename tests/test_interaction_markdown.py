from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
import json

import pytest

from phase_agent.runtime.interaction_reports import save_interaction, short_reply
from phase_agent.runtime.turn_process import record_turn


def report_path(reply):
    return Path(reply.split("\n交互记录地址：", 1)[1])


def test_brief_chat_saves_readable_branch_and_dft_tables_without_state_changes(tmp_path):
    branch = tmp_path / "workflow_state/branch_plans/branch_plan_1.json"
    branch.parent.mkdir(parents=True)
    branch.write_text(
        json.dumps(
            {
                "actual": {
                    "new_branch_count": 2,
                    "registered_structure_count": 3,
                    "branches": [
                        {"phase": "O3", "Na_fraction": 0.5, "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                        {"phase": "P3", "Na_fraction": 1, "H": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]},
                    ],
                }
            }
        )
    )
    dft = tmp_path / "workflow_state/dft_plans/dft_selection_1.json"
    dft.parent.mkdir(parents=True)
    dft.write_text(
        json.dumps(
            {
                "status": "awaiting_approval",
                "decisions": [
                    {
                        "candidate_id": "S-1",
                        "phase": "O3",
                        "Na_fraction": 0.5,
                        "action": "DFT_SINGLE_POINT",
                        "supercell_det": 2,
                    }
                ],
            }
        )
    )
    full = (
        f"当前状态：待确认。\n结果：选点方案已保存。\n下一步：核对后回复同意。\n预算：1个DFT。\nBranch：`{branch}`\nDFT：`{dft}`\n"
        + "详细依据。" * 1000
    )
    state = {"active_model_version": "m1"}
    before = deepcopy(state)
    reply = save_interaction(tmp_path, "继续", full, full, state)
    path = report_path(reply)
    body = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "interaction_records/unassigned"
    assert reply.splitlines()[-1] == f"交互记录地址：{path}"
    assert "[本轮交互记录]" not in reply
    assert len(reply.splitlines()) <= 5 and "核对后回复同意" in reply
    assert "Branch 选点分布" in body and "DFT 选点分布" in body
    assert "| --- |" in body and "| S-1 | O3 | 0.5 | 未记录 | 2 | DFT单点 |" in body
    assert body.count("详细依据。") == 1000
    assert state == before
    assert not list(path.parent.glob("*.tmp"))


def test_only_current_turn_artifacts_are_embedded(tmp_path):
    outside = tmp_path.parent / "do-not-read.json"
    reply = save_interaction(tmp_path, "状态", f"旧文件：`{outside}`", "当前状态：待回传。", {})
    assert "Branch 选点分布" not in report_path(reply).read_text(encoding="utf-8")


def test_error_origins_and_original_stack_are_readable_and_credentials_redacted(tmp_path):
    state = {
        "relax_mc_batches": {
            "b": {
                "remote_diagnostics": {
                    "issue_groups": [
                        {
                            "origin": "remote_calculation",
                            "error": "model missing",
                            "task_ids": ["t"],
                            "traceback": "FileNotFoundError: /cluster/com_1.model",
                        }
                    ],
                    "controller": {"error": "squeue failed"},
                    "evidence_path": "return_diagnostics.json",
                },
                "return_diagnostic": {
                    "origin": "local_import",
                    "error": "completion marker missing",
                },
            }
        }
    }
    reply = save_interaction(
        tmp_path,
        '{"api_key":"secret-value"}',
        "Bearer secret-bearer",
        "当前状态：回收受阻。",
        state,
    )
    text = report_path(reply).read_text(encoding="utf-8")
    assert "超算计算" in text and "本地导入" in text
    assert "FileNotFoundError" in text and "squeue failed" in text
    assert "secret-value" not in text and "secret-bearer" not in text


def test_exception_still_writes_markdown_without_replacing_the_original_error(tmp_path):
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    with pytest.raises(ValueError, match="invalid returned archive"):
        with record_turn(config, "已回传", "fake-thread"):
            raise ValueError("invalid returned archive")
    reports = list((tmp_path / "interaction_records").rglob("*.md"))
    assert len(reports) == 1
    assert "invalid returned archive" in reports[0].read_text(encoding="utf-8")


def test_studio_turn_returns_short_receipt_and_preserves_full_original_answer(
    tmp_path, monkeypatch
):
    import phase_agent.runtime.studio_runtime as runtime
    from phase_agent.runtime.loaded_code import assert_loaded_code_current

    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(config))
    monkeypatch.setattr("phase_agent.runtime.loaded_code.assert_loaded_code_current", lambda: None)
    answer = (
        "当前状态：已完成回收。\n结果：回收2项。\n下一步：检查相图。\n" + "错误和完整依据。" * 200
    )
    handler = Mock(return_value=answer)
    monkeypatch.setattr(runtime, "_handler", handler)
    reply = runtime.send_project_message("已回传", "thread")
    assert len(reply.splitlines()) <= 6
    assert "错误和完整依据。" * 200 in report_path(reply).read_text(encoding="utf-8")
    assert handler.call_count == 1


def test_approval_summary_keeps_confirmation_and_budget():
    reply = short_reply(
        "项目 · epoch1 · 待确认\n结果：DFT选点方案待确认。\n下一步：阅读分布后回复同意。\n预算：3个GPU作业。\n数百项细节"
    )
    assert "同意" in reply and "3个GPU" in reply and "数百项" not in reply


def test_control_chat_entry_also_writes_markdown_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.runtime.agent_web_service import create_control_server

    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text("{}")
    monkeypatch.setattr(
        "phase_agent.runtime.local_service_tokens.local_service_tokens",
        lambda: ("a" * 24, "b" * 24),
    )
    monkeypatch.setattr(
        "phase_agent.runtime.agent_api.create_server",
        lambda chat, **kwargs: SimpleNamespace(chat=chat),
    )
    handler = Mock(
        return_value="结果：已经核对。\n下一步：查看详细记录。\n" + "完整回收明细。" * 200
    )
    server = create_control_server(handler, runtime, port=0)
    reply = server.chat([{"role": "user", "content": "检查回传"}], conversation_id="api")
    assert len(reply.splitlines()) <= 5
    assert "完整回收明细。" * 200 in report_path(reply).read_text(encoding="utf-8")
    assert len(list((tmp_path / "interaction_records").rglob("*.md"))) == 1
    assert handler.call_count == 1

"""Recovery metrics survive the same brief reply path as follow-up approvals."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from phase_agent.runtime.chat_state_presentation import format_epoch_reply
from phase_agent.runtime.concise_dialogue import concise_body
from phase_agent.runtime.response_preferences import response_detail
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from tests.test_dft_comparison_csv import add_result


def next_proposal(tool):
    return {
        "recommended_action": tool,
        "expected_purpose": "根据已回收结果准备下一步",
        "raw_action": {
            "tool": tool,
            "parameters": {"prepare_inputs_only": True} if tool == "update_mlip" else {},
        },
    }


def render(result, tmp_path, detail):
    with response_detail(detail):
        state = result.get("state") or {}
        text = format_workflow_reply(result, tmp_path / "state.json")
        return format_epoch_reply(format_turn_reply(concise_body(text, state), state), state)


@pytest.mark.parametrize("detail", ["brief", "detailed"])
@pytest.mark.parametrize("tool", ["update_mlip", "generate_branches", "select_dft_candidates"])
def test_recovery_followup_approval_keeps_round_metrics(tmp_path, detail, tool):
    state = {"active_model_version": "m1"}
    add_result(state, energy_error=0.02, force_error=0.03)
    # This older, much worse round must not change this round's statistics.
    add_result(state, task_id="old", operation="000", energy_error=10, force_error=8)
    result = {
        "status": "awaiting_approval",
        "recovered_count": 1,
        "state": state,
        "agent_proposal": next_proposal(tool),
    }
    state["pending_execution_policies"] = {"next": {"agent_proposal": result["agent_proposal"]}}
    before = deepcopy(result)
    text = render(result, tmp_path, detail)
    assert "本轮 DFT：回收 1/1，合格配对 1" in text
    assert "能量 MAE/RMSE 10/10 meV/atom" in text
    assert "受力 MAE/RMSE 30/30 meV/Å" in text
    assert "同意" in text
    assert text.count("本轮 DFT：") == 1
    assert result == before


def test_verified_recovery_wraps_analysis_only_once(tmp_path):
    state = {}
    add_result(state)
    result = {
        "status": "completed",
        "state": state,
        "recovered_count": 0,
        "verified_recovery": {"dft_single_point": {"completed": 1, "failed": 0}},
    }
    text = render(result, tmp_path, "detailed")
    assert text.count("本轮 DFT：回收 1/1") == 1
    assert "本次已核对回传并更新记录" in text


def test_dialogue_present_passes_verified_recovery_to_brief_approval(tmp_path):
    from phase_agent.graphs.dialogue.nodes import present

    state = {}
    add_result(state, energy_error=0.02, force_error=0.03)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    saved_bytes = state_path.read_bytes()
    original = deepcopy(state)
    handler = SimpleNamespace(state_path=state_path, conversation_id=None)
    runtime = SimpleNamespace(context=SimpleNamespace(handler=handler))
    with response_detail("brief"):
        reply = present(
            {
                "result": {
                    "status": "awaiting_approval",
                    "state": state,
                    "agent_proposal": next_proposal("update_mlip"),
                    "recovered_count": 0,
                },
                "verified_recovery": {"dft_single_point": {"completed": 1, "failed": 0}},
            },
            runtime,
        )["reply"]
        text = format_epoch_reply(format_turn_reply(concise_body(reply, state), state), state)
    assert "本轮 DFT：回收 1/1" in text
    assert "能量 MAE/RMSE 10/10 meV/atom" in text
    assert "受力 MAE/RMSE 30/30 meV/Å" in text
    assert state_path.read_bytes() == saved_bytes
    assert state == original


def test_real_zero_error_is_distinct_from_missing_prediction(tmp_path):
    state = {}
    add_result(state, energy_error=0, force_error=0)
    text = render({"status": "completed", "state": state, "recovered_count": 1}, tmp_path, "brief")
    assert "能量 MAE/RMSE 0/0 meV/atom" in text
    assert "受力 MAE/RMSE 0/0 meV/Å" in text
    assert "暂不可用" not in text


def test_recovery_missing_prediction_reports_unavailable_not_silence_or_zero(tmp_path):
    state = {}
    add_result(state, evaluator=False)
    text = render({"status": "completed", "state": state, "recovered_count": 1}, tmp_path, "brief")
    assert "本轮 DFT：回收 1/1，合格配对 0" in text
    assert "能量 MAE/RMSE 暂不可用" in text
    assert "受力 MAE/RMSE 暂不可用" in text
    assert "配对不完整" in text
    assert "0/0 meV" not in text


def test_partial_pairing_reports_only_valid_pairs_with_coverage_warning(tmp_path):
    state = {}
    add_result(state, energy_error=0.02, force_error=0.03)
    add_result(state, task_id="missing", evaluator=False)
    text = render({"status": "completed", "state": state, "recovered_count": 2}, tmp_path, "brief")
    assert "回收 2/2，合格配对 1" in text
    assert "10/10 meV/atom" in text and "30/30 meV/Å" in text
    assert "配对不完整" in text


@pytest.mark.parametrize("tool", ["update_mlip", "generate_branches", "select_dft_candidates"])
def test_unrelated_proposal_does_not_repeat_old_round(tmp_path, tool):
    state = {}
    add_result(state)
    text = render(
        {"status": "awaiting_approval", "state": state, "agent_proposal": next_proposal(tool)},
        tmp_path,
        "brief",
    )
    assert "本轮 DFT：" not in text and "能量 MAE/RMSE" not in text

"""Final chat summaries keep the completed result, recommendation and approval scope."""

from copy import deepcopy
import pytest

from phase_agent.runtime.concise_dialogue import approval_reply, concise_body
from phase_agent.runtime.interaction_reports import short_reply


def test_completed_result_and_specific_next_purpose_survive_together():
    text = (
        "项目 · epoch1 · DFT已回收\n"
        "结果：本轮DFT回收12个，3个失败，已入账。\n"
        "本次：能量和力误差偏高，建议微调模型\n"
        "下一步：回复同意或拒绝。\n"
        "审批范围：只准备训练输入，提交超算另行操作。"
    )
    visible = short_reply(concise_body(text, {}))
    assert "回收12个" in visible and "建议微调模型" in visible
    assert "只准备训练输入，提交超算另行操作" in visible


def test_next_step_does_not_hide_handoff_paths_or_truncate_approval_scope():
    folder = "E:/project/" + "long/" * 45 + "a" * 32 + "/training_input"
    text = (
        "结果：训练输入已准备完成。\n"
        f"上传整个文件夹：`{folder}`\n"
        "提交：`bash GPU.sh`\n"
        "回传：`E:/project/training_output`\n"
        "下一步：上传并提交，完成后回传结果。\n"
        "确认范围：" + "合成范围说明；" * 40 + "只准备输入，不提交超算。"
    )
    visible = short_reply(concise_body(text, {}))
    assert folder in visible and "bash GPU.sh" in visible
    assert "E:/project/training_output" in visible
    assert "只准备输入，不提交超算。" in visible


@pytest.mark.parametrize(
    "tool,parameters,recommendation",
    [
        ("generate_branches", {}, "生成候选结构"),
        ("prepare_local_batch_files", {"mode": "relax_mc_inputs"}, "准备 Relax+MC 输入"),
        ("select_dft_candidates", {}, "补充 DFT 决策"),
        ("update_mlip", {"prepare_inputs_only": True, "action": "RETRAIN_MLIP"}, "微调"),
        ("check_convergence", {}, "检查收敛"),
        ("pause_search", {}, "暂停搜索"),
    ],
)
def test_approval_card_names_the_recommendation_and_why(monkeypatch, tool, parameters, recommendation):
    if tool == "prepare_local_batch_files":
        # Scope verification is separate; format a proposal with an available overview.
        monkeypatch.setattr(
            "phase_agent.tools.workflows.generation_artifacts.proposal_artifacts",
            lambda *_: {"text": "E:/plan/branch-overview.txt"},
        )
    proposal = {
        "recommended_action": tool,
        "raw_action": {
            "tool": tool,
            "parameters": parameters,
            "scientific_decision": {"rationale": "已回收证据支持本步建议"},
        },
    }
    text = short_reply(approval_reply(proposal, {}))
    assert "下一步：建议" in text and recommendation in text
    assert "已回收证据支持本步建议" in text
    assert "是否同意" in text and "确认范围：" in text


def test_missing_relax_scope_keeps_its_concrete_blocker():
    proposal = {
        "recommended_action": "prepare_local_batch_files",
        "raw_action": {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_mc_inputs"}},
    }
    visible = short_reply(approval_reply(proposal, {}))
    assert "先补齐 branch 生成概况和范围核对" in visible
    assert "是否同意" not in visible


def test_existing_post_dft_review_keeps_the_selected_route_reason():
    proposal = {
        "recommended_action": "update_mlip",
        "raw_action": {
            "tool": "update_mlip",
            "parameters": {"prepare_inputs_only": True, "action": "RETRAIN_MLIP"},
            "post_dft_review": {"choice": "finetune", "finetune_assessment": "当前合格数据足够，原模型误差支持微调"},
        },
    }
    assert "当前合格数据足够，原模型误差支持微调" in short_reply(approval_reply(proposal, {}))


def test_approval_in_lifecycle_events_uses_the_same_concrete_card(tmp_path):
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    proposal = {
        "recommended_action": "update_mlip",
        "raw_action": {
            "tool": "update_mlip",
            "parameters": {"prepare_inputs_only": True, "action": "RETRAIN_MLIP"},
        },
    }
    result = {
        "status": "awaiting_approval",
        "state": {"pending_execution_policies": {"current": {"agent_proposal": proposal}}},
        "events": [
            {"agent_proposal": {"recommended_action": "pause_search"}},
            {"agent_proposal": proposal},
        ],
    }
    visible = short_reply(format_workflow_reply(result, tmp_path / "state.json"))
    assert "下一步：建议微调模型" in visible
    assert "是否同意" in visible and "确认范围：" in visible


def test_chat_recovery_presentation_with_a_fake_planner(tmp_path, monkeypatch):
    """Presentation only; test_recovered_dft_production_entry retains the actual planner."""
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json
    from tests.test_dft_comparison_csv import add_result
    from tests.test_unified_react_dialogue import make_handler

    calls, planned = [], []

    def model(request):
        calls.append(request["mode"])
        assert request["mode"] == "recovery_intent"
        return {"recovery_intent": "approve"}

    handler = make_handler(tmp_path, model)
    snapshot = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
    facts = {
        "active_model_version": "m1",
        "confirmed_config_version": snapshot["config_version"],
        "confirmed_config": snapshot["config"],
    }
    for index in range(12):
        add_result(facts, task_id=f"dft-{index}")
    for index in range(3):
        failed = deepcopy(facts["tasks"][0])
        failed.update(task_id=f"failed-{index}", status="failed")
        facts["tasks"].append(failed)
    for task in facts["tasks"]:
        task["config_version"] = snapshot["config_version"]
    write_json(handler.state_path, facts)

    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recovery_candidate",
        lambda state, **_: {"pending": not state.get("recovery_acknowledgement"), "fingerprint": "return-1"},
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recover_saved_results",
        lambda state, **_: state,
    )

    def plan(invocation, feedback, message):
        planned.append((invocation, feedback))
        state = read_json(handler.state_path, {})
        proposal = {
            "recommended_action": "update_mlip",
            "expected_purpose": "改善当前模型的能量与力预测",
            "raw_action": {
                "tool": "update_mlip", "task_key": "next-training",
                "parameters": {"prepare_inputs_only": True, "action": "RETRAIN_MLIP"},
                "scientific_decision": {"rationale": "合成回收误差支持微调，失败项按部分覆盖保留"},
            },
        }
        state["pending_execution_policies"] = {
            "next-training": {"status": "awaiting_approval", "agent_proposal": proposal}
        }
        write_json(handler.state_path, state)
        return {"status": "awaiting_approval", "state": state, "agent_proposal": proposal}

    handler._run = plan
    reply = handler(
        [{"role": "user", "content": "这些回传现在可以收进来，并告诉我推荐的下一步"}],
        conversation_id="recovery-next",
    )
    visible = short_reply(reply)
    assert calls == ["recovery_intent"]
    assert len(planned) == 1 and planned[0][1] is None
    assert "12/15" in visible and "3 个失败" in visible
    assert "下一步：建议微调模型" in visible
    assert "合成回收误差支持微调" in visible
    assert "是否同意" in visible and "完成本批输入准备后停止" in visible
    assert "尚未符合" not in visible
    saved = read_json(handler.state_path, {})
    assert len(saved["pending_execution_policies"]) == 1
    assert not saved.get("budget_reservations")
    assert not saved.get("remote_finetune_jobs")

"""A returned training result needs one analysis turn and one explicit switch approval."""

from copy import deepcopy
from types import SimpleNamespace
import pytest


def returned_state():
    return {
        "active_model_version": "base",
        "execution_recovery_report": {"status": "clear", "unsettled": []},
        "remote_finetune_jobs": {
            "j": {
                "status": "results_received",
                "original_model_version": "base",
                "activated": False,
                "directory": "nonexistent",
                "returned_results": {"kfold_metrics": {"structures": 24}},
            }
        },
    }


def test_consent_to_unanalyzed_return_is_not_an_empty_scientific_approval(tmp_path):
    from phase_agent.graphs.dialogue.saved_control import unavailable_review_reply

    handler = SimpleNamespace(config_delegate=None, state_path=tmp_path / "state.json")
    state = returned_state()
    assert unavailable_review_reply(handler, "同意", state) is None
    assert "没有可批准" in unavailable_review_reply(handler, "拒绝", state)
    assert "没有可批准" in unavailable_review_reply(handler, "同意", {})


def test_status_keeps_the_next_analysis_step_in_the_visible_reply():
    from phase_agent.runtime.status_presentation import format_progress
    from phase_agent.runtime.interaction_reports import short_reply

    state = returned_state()
    before = deepcopy(state)
    visible = short_reply(format_progress(state))
    assert "尚未分析" in visible and "下一步：" in visible
    assert "报告" in visible and "epoch" in visible and "已回收" in visible
    assert "准备验证作业" not in visible
    assert state == before


def test_full_studio_rendering_keeps_the_specific_epoch_approval():
    from phase_agent.runtime.training_reply import concise_training_reply
    from phase_agent.runtime.studio_reply_presentation import format_turn_reply
    from phase_agent.runtime.concise_dialogue import concise_body
    from phase_agent.runtime.interaction_reports import short_reply

    state = {
        "active_model_version": "old",
        "candidate_models": {
            "new": {"agent_review": {"choice": "activate", "reason": "训练已核验"}}
        },
    }
    handoff = {
        "stage": "awaiting_activation_approval",
        "candidate_model_version": "new",
        "training_report_path": "E:/project/results/finetuning_results.md",
        "next_epoch": "epoch2",
    }
    visible = short_reply(
        concise_body(format_turn_reply(concise_training_reply(handoff, state), state), state)
    )
    assert "是否进入 epoch2" in visible and "同意" in visible and "拒绝" in visible
    assert "finetuning_results.md" in visible and "本次批准仅切换模型" in visible


def test_repeated_return_notice_displays_the_saved_plan_without_reanalysis(tmp_path, monkeypatch):
    from phase_agent.graphs.dialogue.nodes import read_project
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.runtime import review_requests, plan_queries
    from phase_agent.graphs.dialogue import support
    import phase_agent.tools.workflows.recover_saved_results as recovery

    state = returned_state()
    path = tmp_path / "state.json"
    write_json(path, state)
    handler = SimpleNamespace(
        state_path=path, conversation_id=None, config_delegate=None, workflow_kwargs={}
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            handler=handler, conversation_id="t", messages=[{"role": "user", "content": "已回收"}]
        )
    )
    monkeypatch.setattr(recovery, "recovery_candidate", lambda *a, **kw: {"pending": False})
    monkeypatch.setattr(
        recovery, "recover_saved_results", lambda *a, **kw: pytest.fail("must reuse saved review")
    )
    monkeypatch.setattr(
        review_requests, "pending_reviews", lambda s: [{"review_kind": "model_activation"}]
    )
    monkeypatch.setattr(plan_queries, "pending_plan_reply", lambda s: "已保存的模型切换方案")
    bound = []
    monkeypatch.setattr(support, "bind_presented_proposal", lambda h, s: bound.append(True))
    result = read_project({}, runtime)
    assert result["operation"] == "done" and result["reply"] == "已保存的模型切换方案"
    assert bound == [True]

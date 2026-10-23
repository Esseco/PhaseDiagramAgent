from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_unified_react_dialogue import make_handler
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.workflows.propose_followup import propose_after_generation
from phase_agent.runtime.studio_reply_presentation import format_turn_reply


@pytest.mark.parametrize("stage", ["relax_and_feature", "deep_search", "dft_single_point"])
def test_return_notification_reaches_analysis_and_saved_proposal_in_one_turn(
    tmp_path, monkeypatch, stage
):
    calls, executed = [], []

    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            assert payload["decision_context"]["conversation_controls"]
            return {"kind": "advance", "answer": "核对回传并分析下一步"}
        assert not payload["decision_context"].get("conversation_controls")
        # Summary-first transport exposes counts before loading full scientific evidence.
        assert payload["state"]["task_counts"]["completed"] == 1
        return {
            "tool": "check_convergence",
            "parameters": {},
            "budget": 0,
            "reason": "根据已回收数据检查",
        }

    handler = make_handler(
        tmp_path,
        model,
        {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}},
    )
    write_json(
        handler.state_path,
        {"tasks": [{"task_id": "returned", "stage": stage, "status": "pending"}]},
    )

    def observe(state, **kwargs):
        current = deepcopy(state)
        current["tasks"][0]["status"] = "completed"
        write_json(handler.state_path, current)
        return current

    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recover_saved_results", observe
    )
    reply = handler(
        [{"role": "user", "content": "已经回收好了，接下来生成下一步文件"}], conversation_id="t"
    )
    assert "本次已核对回传" in reply
    assert read_json(handler.state_path, {}).get("pending_execution_policies")
    assert len(calls) == 3 and not executed
    handler([{"role": "user", "content": "同意"}], conversation_id="t")
    assert len(executed) == 1 and len(calls) == 3


def test_natural_approval_reuses_frozen_plan_and_survives_restart(tmp_path):
    calls, executed = [], []

    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "检查"}
        binding = payload["conversation_facts"]["presented_plan_binding"]
        return {
            "kind": "review",
            "plan_id": binding["plan_id"],
            "decision": "approve",
            "_llm_usage": {"calls": 1, "input_tokens": 100, "output_tokens": 10},
        }

    tools = {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}}
    handler = make_handler(tmp_path, model, tools)
    handler([{"role": "user", "content": "分析并提出下一步"}], conversation_id="t")
    original = deepcopy(read_json(handler.state_path, {})["pending_execution_policies"])
    restarted = make_handler(tmp_path, model, tools)
    reply = restarted([{"role": "user", "content": "批准这个方案，生成文件"}], conversation_id="t")
    assert len(executed) == 1 and len(calls) == 2
    state = read_json(handler.state_path, {})
    assert not state.get("pending_execution_policies")
    assert (
        state["action_records"][-1]["agent_proposal"]
        == next(iter(original.values()))["agent_proposal"]
    )
    assert "再次" not in reply


def test_approval_must_reference_the_presented_plan(tmp_path):
    calls, executed = [], []

    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "检查"}
        return {"kind": "review", "plan_id": "invented", "decision": "approve"}

    handler = make_handler(tmp_path, model, {"check_convergence": lambda **kw: executed.append(kw)})
    handler([{"role": "user", "content": "提出方案"}], conversation_id="t")
    reply = handler([{"role": "user", "content": "批准该方案"}], conversation_id="t")
    assert "不一致" in reply and not executed


def test_scientific_change_still_blocks_approval(tmp_path):
    calls, executed = [], []

    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "检查"}
        binding = payload["conversation_facts"]["presented_plan_binding"]
        return {"kind": "review", "plan_id": binding["plan_id"], "decision": "approve"}

    handler = make_handler(tmp_path, model, {"check_convergence": lambda **kw: executed.append(kw)})
    handler([{"role": "user", "content": "提出方案"}], conversation_id="t")
    state = read_json(handler.state_path, {})
    state["active_model_version"] = "changed"
    write_json(handler.state_path, state)
    reply = handler([{"role": "user", "content": "批准这个方案"}], conversation_id="t")
    assert "已变化" in reply and not executed


def test_unverified_recovery_report_does_not_claim_completion(tmp_path):
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler

    calls = []

    def workflow(**kwargs):
        calls.append(kwargs)
        return {
            "status": "answered",
            "state": read_json(tmp_path / "state.json", {}),
            "answer": "尚未发现匹配的结果包，请核对当前批次 results。",
        }

    handler = RunWorkflowChatHandler(
        {"state_path": str(tmp_path / "state.json")}, workflow=workflow
    )
    write_json(
        handler.state_path,
        {"tasks": [{"task_id": "one", "stage": "deep_search", "status": "pending"}]},
    )
    reply = handler([{"role": "user", "content": "我已经回收了，往下做"}], conversation_id="t")
    assert len(calls) == 1 and calls[0]["human_feedback"] is None
    assert "本次已核对回传" not in reply
    assert read_json(handler.state_path, {})["tasks"][0]["status"] == "pending"


def test_legacy_binding_migration_requires_exact_unchanged_state(tmp_path):
    from phase_agent.graphs.dialogue.support import _save_binding, _load_binding
    from phase_agent.tools.step_runner.file_protocol import content_id

    handler = SimpleNamespace(state_path=tmp_path / "state.json", conversation_id="t")
    state = {"budget_usage": {"llm": {"calls": 10}}}
    write_json(handler.state_path, state)
    summary = build_status_summary(state)
    old = content_id(
        {key: value for key, value in summary.items() if key != "summary_id"}, "summary"
    )
    _save_binding(handler, {"plan_id": "p", "state_version": old, "proposal_hash": "h"})
    assert _load_binding(handler)["state_version"] == summary["summary_id"]
    _save_binding(handler, {"plan_id": "p", "state_version": old, "proposal_hash": "h"})
    write_json(handler.state_path, {**state, "active_model_version": "new"})
    assert _load_binding(handler)["state_version"] == old


def test_token_accounting_does_not_make_a_scientific_plan_stale():
    original = {
        "budget_usage": {"total_relative_cost": 10, "stages": {"deep_search": {"cost": 10}}}
    }
    after = deepcopy(original)
    after["budget_usage"]["llm"] = {"calls": 2, "input_tokens": 100}
    assert build_status_summary(original)["summary_id"] == build_status_summary(after)["summary_id"]
    after["budget_usage"]["total_relative_cost"] += 1
    assert build_status_summary(original)["summary_id"] != build_status_summary(after)["summary_id"]


def test_session_continue_proposes_without_a_second_user_prompt():
    from phase_agent.graphs.dialogue.history import restore_session

    handler = SimpleNamespace(history_decision=None)
    assert restore_session(handler, "继续", {}) is None
    assert handler.history_decision == "continue"


@pytest.mark.parametrize("tool", ["select_candidates", "adjust_strategy", "reevaluate_candidates"])
def test_local_milestone_prepares_one_followup_without_approval_or_execution(tool):
    result = {
        "status": "completed",
        "state": {},
        "steps_executed": 1,
        "events": [{"action": {"tool": tool}, "execution": {"status": "completed", "result": {}}}],
    }
    frame = {
        "execution_mode": "interactive",
        "agent_client": lambda _: None,
        "config_session": {},
        "effective_registry": {},
        "effective_config": {},
    }
    calls = []

    def propose(state, session, **kwargs):
        calls.append(kwargs)
        assert kwargs["human_feedback"] is None and kwargs["replay_record"] is None
        return {
            "status": "awaiting_approval",
            "state": {"pending_execution_policies": {"p": {}}},
            "events": [],
        }

    following = propose_after_generation(frame, result, event_loop=propose)
    assert len(calls) == 1 and following["steps_executed"] == 1
    assert following["completed_action"]["tool"] == tool


def test_pending_external_results_do_not_trigger_new_followup():
    result = {
        "status": "completed",
        "state": {"pending_tasks": [{"task_id": "mc"}]},
        "events": [{"action": {"tool": "adjust_strategy"}, "execution": {"status": "completed"}}],
    }
    frame = {"execution_mode": "interactive", "agent_client": lambda _: None}

    def forbidden(*args, **kwargs):
        raise AssertionError("new proposal while still waiting")

    assert propose_after_generation(frame, result, event_loop=forbidden) == result


def test_presentation_does_not_add_an_extra_continue_round(tmp_path):
    reply = format_turn_reply("已核对当前数据。", {})
    assert "回复‘继续’" not in reply and "是否现在" not in reply
    state = {"tasks": [{"task_id": "mc", "status": "pending", "stage": "deep_search"}]}
    incomplete = format_turn_reply("MC 输入已准备。", state)
    assert "尚不能提交" in incomplete
    for name in ("inputs.zip", "submit_all.sh", "dispatch_tasks.py"):
        (tmp_path / name).write_text("synthetic handoff")
    state["slurm_batches"] = [
        {
            "task_ids": ["mc"],
            "handoff": {
                "folder": str(tmp_path),
                "upload_archive": str(tmp_path / "inputs.zip"),
                "return_directory": str(tmp_path / "results"),
            },
        }
    ]
    waiting = format_turn_reply("MC 输入已准备。", state)
    assert "回传后说“已回传”" in waiting and "回复‘继续’" not in waiting


def test_mc_plan_simple_approval_survives_restart_and_updates_changed_input_plan(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from tests.test_regenerate_mc_inputs import _state
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.tools.local import regenerate_mc_inputs as module

    state = _state(tmp_path)
    state["confirmed_config_version"] = "v1"
    input_path = Path(state["tasks"][0]["input_path"])
    input_path.parent.mkdir(parents=True)
    input_path.write_text("old")
    state_path = tmp_path / "state.json"
    write_json(state_path, state)
    config = {
        "upload_batches_directory": str(tmp_path),
        "budgets": {"stage_limits": {"deep_search": {"max_cost": 100}}},
    }
    monkeypatch.setattr(
        "phase_agent.configuration.runtime.build_effective_run_config.build_effective_run_config",
        lambda *_: config,
    )
    monkeypatch.setattr(
        "phase_agent.tools.local.replan_mc_allocation.replan_mc_allocation",
        lambda *_: module._approved_mc_preview(state)[0],
    )
    calls = []

    def regenerate(original, **kwargs):
        calls.append(kwargs)
        return {"state": original, "task_count": 1, "batch_count": 1}

    monkeypatch.setattr(module, "regenerate_mc_inputs", regenerate)
    kwargs = {
        "state_path": str(state_path),
        "config_session": {},
        "run_config": {},
        "manager": None,
        "phase_references": {},
    }
    handler = RunWorkflowChatHandler(kwargs)
    reply = handler([{"role": "user", "content": "重新生成MC"}], conversation_id="t")
    assert "回复“同意”" in reply and not calls
    restarted = RunWorkflowChatHandler(kwargs)
    input_path.write_text("changed")
    refreshed = restarted([{"role": "user", "content": "同意"}], conversation_id="t")
    assert "已更新方案" in refreshed and not calls
    regenerated = restarted([{"role": "user", "content": "同意"}], conversation_id="t")
    assert "生成 1 个 MC 输入" in regenerated and len(calls) == 1

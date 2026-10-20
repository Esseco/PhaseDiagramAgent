from copy import deepcopy

import pytest

from phase_agent.runtime.chat_state_presentation import format_epoch_reply
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
from phase_agent.tools.workflows.propose_followup import propose_after_generation


def refreshed_state():
    return {
        "active_model_version": "new",
        "active_model": {"version": "new"},
        "model_refresh": {
            "status": "waiting_results",
            "wave": 1,
            "new_model_version": "new",
            "task_ids": ["t1"],
            "plan": {"checksum": "p", "total_unique": 2},
        },
        "tasks": [
            {
                "task_id": "t1",
                "model_version": "new",
                "status": "completed",
                "stage": "relax_and_feature",
                "model_refresh_id": "p",
                "input_structure_sha256": "s1",
                "outputs": {"single_point_completed": True},
            }
        ],
        "phase_diagrams": {
            "mlip": {
                "status": "completed",
                "model_version": "new",
                "entries": [{"record_id": "r1"}],
            }
        },
        "phase_records": [{"record_id": "r1", "source_task_id": "t1", "source_version": "new"}],
    }


def completion():
    state = refreshed_state()
    state["model_refresh"].update(status="completed", refreshed=1, deferred=1, partial=True)
    return {
        "status": "completed",
        "state": state,
        "steps_executed": 1,
        "events": [
            {
                "status": "completed",
                "completed_refresh": {
                    "model_version": "new",
                    "refreshed": 1,
                    "deferred": 1,
                    "partial": True,
                    "task_count": 1,
                    "phase_diagram_entry_count": 1,
                },
            }
        ],
    }


def frame():
    return {
        "execution_mode": "interactive",
        "agent_client": lambda _: None,
        "config_session": {},
        "effective_registry": {},
        "effective_config": {},
        "invocation_id": "refresh",
        "context": {"user_message": "已回传"},
    }


def proposal():
    return {
        "recommended_action": "generate_branches",
        "raw_action": {"tool": "generate_branches", "parameters": {}},
        "reason": "依据刷新后的相图，补充低覆盖区域候选。",
    }


def test_refresh_gate_emits_compact_completion_evidence():
    original = refreshed_state()
    response = run_model_refresh_gate(original, {}, registry={}, context={})
    assert response["status"] == "completed"
    assert response["completed_refresh"] == completion()["events"][0]["completed_refresh"]
    assert original["model_refresh"]["status"] == "waiting_results"


def test_refresh_milestone_requests_one_next_proposal_without_carrying_approval():
    result = completion()
    f = frame()
    f["human_feedback"] = {"decision": "approve"}
    f["replay_record"] = {"final_action": {"tool": "prepare_local_batch_files"}}
    calls = []

    def propose(state, session, **kwargs):
        calls.append(kwargs)
        assert kwargs["human_feedback"] is None and kwargs["replay_record"] is None
        assert kwargs["execution_mode"] == "interactive" and kwargs["max_steps"] == 1
        assert "禁止执行或提交下一项任务" in kwargs["context"]["user_message"]
        assert "刷新" in kwargs["context"]["user_message"]
        assert "生成当前模型新branch" in kwargs["context"]["user_message"]
        updated = deepcopy(state)
        updated["pending_execution_policies"] = {"next": {"agent_proposal": proposal()}}
        return {
            "status": "awaiting_approval",
            "state": updated,
            "events": [],
            "agent_proposal": proposal(),
        }

    combined = propose_after_generation(f, result, event_loop=propose)
    assert len(calls) == 1
    assert combined["status"] == "awaiting_approval"
    assert combined["steps_executed"] == 1
    assert combined["completed_refresh"]["refreshed"] == 1
    assert combined["state"]["tasks"] == result["state"]["tasks"]


@pytest.mark.parametrize(
    "change", ["no_event", "other_model", "pending", "waiting", "already_completed"]
)
def test_historical_refresh_or_blocked_state_does_not_request_followup(change):
    result = completion()
    if change == "no_event":
        result["events"] = []
    elif change == "other_model":
        result["state"]["active_model_version"] = "other"
    elif change == "pending":
        result["state"]["pending_execution_policies"] = {"p": {}}
    elif change == "waiting":
        result["state"]["tasks"].append({"status": "pending", "stage": "deep_search"})
    else:
        result["events"] = [{"status": "completed", "action": {"tool": "check_convergence"}}]

    def forbidden(*args, **kwargs):
        raise AssertionError("no new proposal allowed")

    assert propose_after_generation(frame(), result, event_loop=forbidden) == result


@pytest.mark.parametrize("verbose", [False, True])
def test_completed_refresh_reply_survives_all_studio_renderers(verbose):
    result = completion()
    result["verified_recovery"] = {"relax_and_feature": {"completed": 1, "failed": 0}}
    reply = format_workflow_reply(result, "state.json", verbose=verbose)
    text = format_epoch_reply(format_turn_reply(reply, result["state"]), result["state"])
    assert "已刷新 1 个独立结构" in text
    assert "相图" in text and "1 条" in text
    assert "1 个结构未刷新" in text and "收敛" in text
    assert "下一步：" in text and "方案" in text
    assert "尚无待批准方案" in text
    assert "`workflow`" not in text
    assert "本次：已检查当前项目状态" not in text


@pytest.mark.parametrize("verbose", [False, True])
def test_next_approval_keeps_completion_and_exact_scope(verbose):
    result = completion()
    result.update(status="awaiting_approval", agent_proposal=proposal())
    result["state"]["pending_execution_policies"] = {"next": {"agent_proposal": proposal()}}
    reply = format_workflow_reply(result, "state.json", verbose=verbose)
    text = format_epoch_reply(format_turn_reply(reply, result["state"]), result["state"])
    assert "已刷新 1 个独立结构" in text and "1 个结构未刷新" in text
    assert "同意" in text and ("确认范围" in text or "本次只生成结构" in text)
    assert "尚无待批准方案" not in text


def test_followup_failure_keeps_completed_facts_and_actionable_next_step():
    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    result = propose_after_generation(frame(), completion(), event_loop=fail)
    assert "offline" in result["followup_error"]
    text = format_workflow_reply(result, "state.json", verbose=False)
    assert "已刷新 1 个独立结构" in text
    assert "尚无待批准方案" in text and "下一步：" in text
    assert "重新请求" in text


def test_pending_refresh_does_not_claim_completion():
    state = refreshed_state()
    state["tasks"][0]["status"] = "pending"
    response = run_model_refresh_gate(state, {}, registry={}, context={})
    assert "completed_refresh" not in response


def test_real_single_action_runner_stops_at_next_approval(tmp_path):
    from tests.test_event_loop import _session
    from phase_agent.tools.workflows.lifecycle_finalization import _run_workflow_actions
    from phase_agent.tools.workflows.run_event_loop import run_action_turn

    executed, calls = [], []

    def model(payload):
        calls.append(payload)
        return {
            "tool": "check_convergence",
            "parameters": {},
            "budget": 0,
            "reason": "依据部分刷新结果检查当前覆盖与证据缺口。",
        }

    def handler(**kwargs):
        executed.append(kwargs)
        return {"status": "completed", "result": {}}

    session = _session()
    f = frame()
    f.update(
        feedback={"state": refreshed_state()},
        config_session=session,
        effective_registry={"check_convergence": {"handler": handler}},
        agent_client=model,
        max_steps=1,
        state_path=str(tmp_path / "state.json"),
        human_feedback={"decision": "approve", "comment": "同意刷新"},
        replay_record=None,
        approval_directory=None,
        initial_long_term_advice=None,
        pre_reconciled={"reconciled": []},
    )
    response = _run_workflow_actions(f, event_loop=run_action_turn)["result"]
    assert response["status"] == "awaiting_approval", response
    assert len(calls) == 1 and not executed
    assert response["completed_refresh"]["refreshed"] == 1
    assert len(response["state"]["pending_execution_policies"]) == 1
    assert len(response["state"]["tasks"]) == 1
    assert len(response["events"]) == 2
    assert response["steps_executed"] == 1
    assert response["submitted"] is False
    assert (tmp_path / "state.json").is_file()
    pending_key = next(iter(response["state"]["pending_execution_policies"]))
    approved = run_action_turn(
        response["state"],
        session,
        registry=f["effective_registry"],
        agent_client=model,
        context={"unified_dialogue": True, "user_message": "同意"},
        execution_mode="interactive",
        human_feedback={"decision": "approve", "comment": "同意"},
        invocation_id=pending_key,
    )
    assert len(executed) == 1 and len(calls) == 1
    replay = run_action_turn(
        approved["state"],
        session,
        registry=f["effective_registry"],
        agent_client=model,
        context={"unified_dialogue": True},
        execution_mode="interactive",
        human_feedback={"decision": "approve", "comment": "同意"},
        invocation_id=pending_key,
    )
    assert replay["status"] == approved["status"]
    assert len(executed) == 1 and len(calls) == 1


def test_followup_explanation_is_preserved_when_no_approval_is_saved():
    def blocked(state, session, **kwargs):
        return {"status": "rejected", "state": state, "reason": "缺少当前模型的验证数据"}

    result = propose_after_generation(frame(), completion(), event_loop=blocked)
    text = format_workflow_reply(result, "state.json", verbose=False)
    assert "缺少当前模型的验证数据" in text
    assert "已刷新 1 个独立结构" in text and "尚无待批准方案" in text


def test_stale_completion_marker_is_not_rendered_for_another_model():
    result = completion()
    result["state"]["active_model_version"] = "other"
    text = format_workflow_reply(result, "state.json", verbose=False)
    assert "已刷新 1 个独立结构" not in text

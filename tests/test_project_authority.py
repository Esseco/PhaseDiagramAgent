from copy import deepcopy
from types import SimpleNamespace

import pytest

from phase_agent.analysis.state.project_authority import (
    confirmed_configuration,
    current_project,
    project_identity,
)
from phase_agent.analysis.state.build_state_snapshot import build_state_snapshot
from phase_agent.runtime.project_context import handler_project_context, project_context_line
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import read_json, write_json, content_id
from tests.test_unified_react_dialogue import make_handler


def facts(root=None):
    config = {
        "calculation": {"mlip_version": "initial"},
        "mlip": {"version": "initial", "model_path": "/old.model"},
        "run": {"stage_task_limit": 10000},
        "system": {"boundary": {"P": ["O3"], "TM_ratio": {"Fe": 1}}},
    }
    if root:
        config["storage"] = {"workspace_root": str(root)}
    return {
        "confirmed_config": config,
        "confirmed_config_version": "config-v16",
        "active_model_version": "new",
        "active_model": {"version": "new", "model_path": "/new.model"},
        "upload_layout": {"model_rounds": {"initial": 1, "new": 2}},
        "candidate_models": {
            "initial": {"status": "activated"},
            "newer": {"status": "validated_candidate"},
        },
    }


def test_active_initial_candidate_and_effective_settings_are_separated(tmp_path):
    state = facts(tmp_path)
    before = deepcopy(state)
    card = current_project(state)
    assert card["active_model"]["version"] == "new"
    assert card["active_model"]["model_path"] == "/new.model"
    assert card["initial_model_version"] == "initial"
    assert card["epoch"] == "epoch1"
    assert card["candidate_models"] == [
        {"version": "newer", "status": "validated_candidate", "activated": False}
    ]
    assert card["configuration"]["effective_settings"]["calculation"]["mlip_version"] == "new"
    assert state == before


def test_unconfirmed_draft_is_not_effective_and_previous_snapshot_remains_effective():
    draft = {"run": {"stage_task_limit": 50}}
    assert confirmed_configuration({}, {"config": draft, "status": "draft"}) == {}
    state = facts()
    session = {
        "status": "draft",
        "draft_revision": 17,
        "config": deepcopy(state["confirmed_config"]),
        "confirmed_snapshot": {"config_version": "config-v16", "config": state["confirmed_config"]},
    }
    session["config"]["run"]["stage_task_limit"] = 50
    card = current_project(state, session)
    assert card["configuration"]["effective_settings"]["run"]["stage_task_limit"] == 10000
    assert card["draft"]["effective"] is False
    assert card["draft"]["status"] == "pending_confirmation"
    assert "run.stage_task_limit" in card["draft"]["changed_paths"]
    assert confirmed_configuration({}, session)["run"]["stage_task_limit"] == 10000


def test_runtime_state_wins_over_conflicting_session_snapshot():
    state = facts()
    card = current_project(
        state,
        {
            "confirmed_snapshot": {
                "config_version": "wrong",
                "config": {"run": {"stage_task_limit": 5}},
            }
        },
    )
    assert card["configuration"]["version"] == "config-v16"
    assert card["consistency_issues"]
    assert card["configuration"]["effective_settings"]["run"]["stage_task_limit"] == 10000


def test_card_reads_latest_session_even_if_handler_cache_is_old(tmp_path):
    state = facts(tmp_path)
    path = tmp_path / "session.json"
    write_json(path, {"status": "draft", "draft_revision": 22, "config": state["confirmed_config"]})
    handler = SimpleNamespace(
        state_path=tmp_path / "workflow_state/state.json",
        workflow_kwargs={"config_session_path": str(path), "config_session": {"draft_revision": 1}},
    )
    card = handler_project_context(handler, state)
    assert card["draft"]["revision"] == 22
    assert card["project_path"] == str(tmp_path)
    assert "设置修改待确认" in project_context_line(card)
    assert "v22" not in project_context_line(card)


def test_same_label_config_or_model_descriptor_changes_invalidate_approval_summary():
    state = facts()
    original = build_status_summary(state)["summary_id"]
    for key in ("config", "model"):
        changed = deepcopy(state)
        if key == "config":
            changed["confirmed_config"]["run"]["stage_task_limit"] = 50
        else:
            changed["active_model"]["model_path"] = "/different.model"
        assert build_status_summary(changed)["summary_id"] != original
    state["budget_usage"] = {"llm": {"calls": 2, "input_tokens": 200}}
    assert build_status_summary(state)["summary_id"] == original


def test_snapshot_supplies_current_card_to_scientific_decision():
    state = facts()
    snapshot = build_state_snapshot(state, snapshot_index=2)
    assert snapshot["current_project"]["configuration"]["version"] == "config-v16"
    assert snapshot["current_project"]["active_model"]["version"] == "new"


def test_outer_question_uses_current_model_even_when_previous_reply_is_old(tmp_path):
    captured = []

    def model(payload):
        captured.append(payload)
        return {"kind": "answer", "answer": "使用已激活的新模型。"}

    handler = make_handler(tmp_path, model)
    state = facts(tmp_path)
    # Keep the valid configuration required by the fixture, with a different active model.
    state["confirmed_config"] = deepcopy(
        handler.workflow_kwargs["config_session"]["confirmed_snapshot"]["config"]
    )
    state["confirmed_config_version"] = handler.workflow_kwargs["config_session"][
        "confirmed_snapshot"
    ]["config_version"]
    write_json(handler.state_path, state)
    handler(
        [
            {"role": "assistant", "content": "当前还是旧模型。"},
            {"role": "user", "content": "现在用哪个模型？"},
        ],
        conversation_id="t",
    )
    assert captured[0]["state"]["current_project"]["active_model"]["version"] == "new"
    assert (
        captured[0]["conversation_facts"]["project_settings"]["calculation"]["mlip_version"]
        == "new"
    )
    assert captured[0]["decision_context"]["current_settings_authority"]


@pytest.mark.parametrize("change", ["model_path", "same_version_setting"])
def test_approval_rejects_changed_content_without_replanning_or_execution(tmp_path, change):
    calls, executed = [], []

    def model(payload):
        calls.append(payload)
        return {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "检查"}

    handler = make_handler(tmp_path, model, {"check_convergence": lambda **kw: executed.append(kw)})
    handler([{"role": "user", "content": "提出方案"}], conversation_id="t")
    state = read_json(handler.state_path, {})
    saved = deepcopy(state["pending_execution_policies"])
    assert next(iter(saved.values()))["project_identity"] == project_identity(state)
    if change == "model_path":
        state["active_model"] = {
            "version": state.get("active_model_version"),
            "model_path": "/changed.model",
        }
    else:
        state["confirmed_config"]["run"]["stage_task_limit"] = 17
    write_json(handler.state_path, state)
    reply = handler([{"role": "user", "content": "同意"}], conversation_id="t")
    assert "变化" in reply
    assert not executed and len(calls) == 1
    assert read_json(handler.state_path, {})["pending_execution_policies"] == saved


def test_previous_scientific_binding_migrates_only_when_current_facts_match(tmp_path):
    from phase_agent.graphs.dialogue.support import _save_binding, _load_binding

    state = facts(tmp_path)
    path = tmp_path / "state.json"
    write_json(path, state)
    summary = build_status_summary(state)
    previous = {k: v for k, v in summary.items() if k not in {"summary_id", "project_identity"}}
    previous["budget_usage"].pop("llm", None)
    previous["budget_usage"].setdefault("total_relative_cost", 0.0)
    previous["budget_usage"].setdefault("stages", {})
    handler = SimpleNamespace(state_path=path, conversation_id="t")
    binding = {
        "plan_id": "p",
        "state_version": content_id(previous, "summary"),
        "proposal_hash": "hash",
    }
    _save_binding(handler, binding)
    assert _load_binding(handler)["state_version"] == summary["summary_id"]
    _save_binding(handler, binding)
    state["active_model_version"] = "changed"
    write_json(path, state)
    assert _load_binding(handler)["state_version"] == binding["state_version"]


def test_header_is_rebuilt_from_model_activation_result_and_pending_draft(tmp_path):
    from phase_agent.runtime.chat_state_presentation import format_epoch_reply

    state = facts(tmp_path)
    card = current_project(state, {"status": "draft", "draft_revision": 19, "config": {}})
    text = format_epoch_reply("切换完成。", state, project_card=card)
    assert text.startswith(tmp_path.name + " · epoch1 · 当前模型 new")
    assert "设置修改待确认" in text
    assert "config-v16" not in text


def test_legacy_cached_snapshot_gets_current_card_without_state_mutation():
    from phase_agent.tools.state.state_manager import agent_state_summary

    state = facts()
    state["current_state_snapshot"] = {"snapshot_id": "old"}
    original = deepcopy(state)
    summary = agent_state_summary(state)
    assert summary["current_project"]["active_model"]["version"] == "new"
    assert state == original


def test_confirmed_session_not_yet_applied_is_distinct_from_unconfirmed_draft():
    state = facts()
    config = deepcopy(state["confirmed_config"])
    config["run"]["stage_task_limit"] = 5
    card = current_project(
        state,
        {
            "status": "confirmed",
            "config": config,
            "confirmed_snapshot": {"config_version": "next", "config": config},
        },
    )
    assert card["draft"]["status"] == "confirmed_not_applied"
    assert card["configuration"]["effective_settings"]["run"]["stage_task_limit"] == 10000
    assert "尚未应用" in project_context_line(card)


def test_changed_candidate_file_changes_activation_approval_identity():
    from phase_agent.runtime.review_requests import additional_reviews

    state = facts()
    state["candidate_models"]["newer"].update(
        old_model_version="new",
        model={"version": "newer", "model_path": "/candidate.model"},
        agent_review={"choice": "activate"},
    )
    state["remote_finetune_jobs"] = {
        "j": {
            "status": "results_received",
            "training_handoff": {
                "stage": "awaiting_activation_approval",
                "candidate_model_version": "newer",
                "direction_proposal": {"direction": "activate", "reason": "用户选择"},
            },
        }
    }
    before = additional_reviews(state)[0]["proposal_hash"]
    state["candidate_models"]["newer"]["model"]["model_path"] = "/other.model"
    assert additional_reviews(state)[0]["proposal_hash"] != before


def test_editable_file_diff_is_labeled_unapplied_without_changing_effective_settings(tmp_path):
    from phase_agent.configuration.session.create_editable_config_json import FORMAT_ID

    state = facts()
    session = {
        "status": "confirmed",
        "config": state["confirmed_config"],
        "draft_revision": 16,
        "confirmed_snapshot": {"config_version": "config-v16", "config": state["confirmed_config"]},
    }
    editable = deepcopy(state["confirmed_config"])
    editable["run"]["stage_task_limit"] = 50
    write_json(
        tmp_path / "parameters/search_config.project.json",
        {"_format": FORMAT_ID, "config": editable},
    )
    handler = SimpleNamespace(
        state_path=tmp_path / "workflow_state/state.json",
        workflow_kwargs={"config_session": session},
    )
    card = handler_project_context(handler, state)
    assert card["draft"]["status"] == "file_changes_not_effective"
    assert "run.stage_task_limit" in card["editable_file"]["changed_paths"]
    assert card["configuration"]["effective_settings"]["run"]["stage_task_limit"] == 10000


def test_generated_H_and_reference_cache_are_not_misrepresented_as_user_edits():
    from phase_agent.analysis.state.project_authority import configuration_changes

    before = {
        "system": {
            "H_generation": {"enabled": True, "size_max": 12},
            "boundary": {"H": {"O3": [1]}},
            "phase_reference_directory": "/reference",
            "phase_references": {"O3": "loaded"},
        }
    }
    after = deepcopy(before)
    after["system"]["boundary"]["H"] = {}
    after["system"]["phase_references"] = {}
    assert not configuration_changes(before, after)
    after["system"]["H_generation"]["size_max"] = 16
    assert configuration_changes(before, after) == ["system.H_generation.size_max"]


def test_configuration_model_receives_current_run_and_unconfirmed_draft_separately(tmp_path):
    from phase_agent.runtime.configuration_chat import ConfigurationChatHandler

    state = facts(tmp_path)
    path = tmp_path / "workflow_state/state.json"
    write_json(path, state)
    seen = []

    def model(payload):
        seen.append(payload)
        return {"reply": "仅修改草稿", "patch": {}, "questions": [], "write_requested": False}

    owner = SimpleNamespace(
        state_path=path,
        agent_client=model,
        workflow_kwargs={"config_session": {"status": "draft", "draft_revision": 22, "config": {}}},
    )
    ConfigurationChatHandler._request_configuration_result(
        owner, {"mode": "configuration_dialogue", "draft_config": {}, "editable_field_catalog": {}}
    )
    assert seen[0]["current_project"]["active_model"]["version"] == "new"
    assert seen[0]["current_project"]["configuration"]["version"] == "config-v16"
    assert seen[0]["current_project"]["draft"]["revision"] == 22


def test_low_level_workflow_also_rejects_approval_after_model_file_change(tmp_path):
    executed = []
    handler = make_handler(
        tmp_path,
        lambda p: {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "检查"},
        {"check_convergence": lambda **kw: executed.append(kw)},
    )
    initial = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]["config"][
        "calculation"
    ]["mlip_version"]
    write_json(
        handler.state_path,
        {
            "active_model_version": initial,
            "active_model": {"version": initial, "model_path": "/original.model"},
        },
    )
    handler([{"role": "user", "content": "提出方案"}], conversation_id="t")
    state = read_json(handler.state_path, {})
    invocation = next(iter(state["pending_execution_policies"]))
    state["active_model"] = {
        "version": state.get("active_model_version"),
        "model_path": "/changed.model",
    }
    write_json(handler.state_path, state)
    result = handler._run(invocation, {"decision": "approve", "comment": "approve"}, "approve")
    assert result["status"] == "proposal_context_changed"
    assert not executed


def test_same_version_different_snapshot_does_not_enter_scientific_workflow(tmp_path):
    calls = []
    handler = make_handler(tmp_path, lambda payload: calls.append(payload))
    snapshot = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
    state = {
        "confirmed_config": deepcopy(snapshot["config"]),
        "confirmed_config_version": snapshot["config_version"],
    }
    state["confirmed_config"]["run"]["stage_task_limit"] = 17
    write_json(handler.state_path, state)
    result = handler._run("mismatch", None, "继续")
    assert result["status"] == "configuration_version_mismatch"
    assert not calls


def test_status_read_does_not_silently_rebind_stale_saved_plan(tmp_path):
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry

    handler = make_handler(
        tmp_path,
        lambda payload: {
            "tool": "check_convergence",
            "parameters": {},
            "budget": 0,
            "reason": "检查",
        },
    )
    handler([{"role": "user", "content": "提出方案"}], conversation_id="t")
    state = read_json(handler.state_path, {})
    pending = deepcopy(state["pending_execution_policies"])
    state["active_model"] = {"version": "changed", "model_path": "/other.model"}
    invocation = next(iter(pending))
    result = run_tool_step(
        state,
        handler.workflow_kwargs["config_session"],
        registry=create_tool_registry(),
        invocation_id=invocation,
        execution_mode="interactive",
    )
    assert result["status"] == "proposal_context_changed"
    assert result["state"]["pending_execution_policies"] == pending


def test_control_approval_identifies_runtime_project_instead_of_old_config_directory(tmp_path):
    from phase_agent.runtime.local_agent_control import LocalAgentControl

    state = facts(tmp_path / "old-project")
    state["pending_execution_policies"] = {
        "p": {"agent_proposal": {"recommended_action": "check_convergence"}}
    }
    path = tmp_path / "workflow_state/state.json"
    write_json(path, state)
    control = LocalAgentControl(SimpleNamespace(state_path=path))
    row = control.pending()["pending"][0]
    assert row["current_project"]["project_path"] == str(tmp_path)
    assert row["current_project"]["project"] == tmp_path.name


def test_scientific_model_uses_actual_project_even_with_old_workspace_in_snapshot(tmp_path):
    captured = []

    def model(payload):
        captured.append(payload)
        if len(captured) == 1:
            return {"kind": "advance", "answer": "分析当前项目"}
        return {"tool": "check_convergence", "parameters": {}, "budget": 0, "reason": "检查"}

    handler = make_handler(tmp_path, model)
    snapshot = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
    snapshot["config"].setdefault("storage", {})["workspace_root"] = str(tmp_path / "old-project")
    handler([{"role": "user", "content": "已经回收，分析下一步"}], conversation_id="t")
    assert len(captured) == 2
    assert captured[1]["state"]["current_project"]["project_path"] == str(tmp_path)
    assert captured[1]["conversation_facts"]["current_project"]["project"] == tmp_path.name

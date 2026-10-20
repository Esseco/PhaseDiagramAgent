from copy import deepcopy
from unittest.mock import Mock

import pytest

from tests.test_relax_mc_batch import fixture
from phase_agent.tools.relax_mc_batch.plan import attach_preview, relax_tasks


def prepared(root):
    manager, state, config, context, action = fixture(root, count=2, single_x=True)
    preview = attach_preview(action, state, context, config)
    old = relax_tasks(preview["parameters"]["relax_mc_preview"], config)[0]
    state["tasks"] = [{**old, "status": "cancelled"}]
    state["active_model"] = deepcopy(config["mlip"])
    state["branch_candidates"] = list(manager.data["branches"].values())
    action["target_ids"] = [old["branch_id"]]
    return manager, state, config, context, action


def test_fact_projection_matches_actual_diagnosis_including_cancellation(tmp_path):
    from phase_agent.tools.relax_mc_batch.plan import registration_facts, registration_conflicts

    manager, state, config, context, action = prepared(tmp_path)
    before = deepcopy(state)
    facts = registration_facts(state, manager, config)
    conflicts = registration_conflicts(action, facts)
    assert conflicts[0]["task_ids"] == [state["tasks"][0]["task_id"]]
    assert conflicts[0]["status_counts"] == {"cancelled": 1}
    with pytest.raises(ValueError, match="already have Relax tasks"):
        attach_preview(action, state, context, config)
    assert state == before
    changed = deepcopy(config)
    changed["mlip"]["relax_parameters"]["fmax"] = 0.001
    assert not registration_conflicts(action, registration_facts(state, manager, changed))
    assert not registration_conflicts(
        action,
        registration_facts(
            {
                **state,
                "tasks": [
                    {
                        **state["tasks"][0],
                        "model_version": "other",
                        "task_key": state["tasks"][0]["task_key"].replace("m1", "other"),
                    }
                ],
            },
            manager,
            config,
        ),
    )


@pytest.mark.parametrize("new_route", ["other_branch", "check_convergence"])
def test_invalid_plan_reaches_model_once_and_replacement_still_requires_approval(
    tmp_path, new_route
):
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry

    manager, state, config, context, action = prepared(tmp_path)
    session = confirm_config_snapshot(create_config_draft(config), user_confirmed=True)
    state["confirmed_config_version"] = session["confirmed_snapshot"]["config_version"]
    handler = Mock(side_effect=AssertionError("unapproved tools must not execute"))
    registry = create_tool_registry(
        {"prepare_local_batch_files": handler, "check_convergence": handler}
    )
    other = next(b for b in manager.data["branches"] if b not in action["target_ids"])
    calls = []

    def agent(payload):
        calls.append(payload)
        if len(calls) == 2 and new_route == "check_convergence":
            return {
                "tool": "check_convergence",
                "target_ids": [],
                "parameters": {},
                "budget": 0,
                "reason": "核对当前证据并比较下一步收益。",
                "_llm_usage": {"calls": 1, "input_tokens": 10, "output_tokens": 5, "cost": 0.2},
            }
        return {
            **action,
            "target_ids": action["target_ids"] if len(calls) == 1 else [other],
            "_llm_usage": {"calls": 1, "input_tokens": 10, "output_tokens": 5, "cost": 0.2},
        }

    agent._validates_original_facts = True
    original_tasks = deepcopy(state["tasks"])
    result = run_tool_step(
        state,
        session,
        registry=registry,
        agent_client=agent,
        context={**context, "user_message": "继续", "unified_dialogue": True},
        execution_mode="interactive",
        invocation_id="continue-after-cancel",
    )
    assert result["status"] == "awaiting_approval", result.get("reason")
    assert len(calls) == 2
    assert calls[1]["decision_context"]["proposal_validation_feedback"]["conflicts"][0][
        "task_ids"
    ] == [original_tasks[0]["task_id"]]
    assert result["agent_proposal"]["raw_action"]["target_ids"] == (
        [other] if new_route == "other_branch" else []
    )
    assert result["state"]["tasks"] == original_tasks
    assert state["tasks"] == original_tasks
    assert result["state"]["budget_usage"]["llm"]["calls"] == 2
    assert result["state"]["budget_usage"]["llm"]["cost"] == pytest.approx(0.4)
    handler.assert_not_called()


def test_repeated_conflict_is_bounded_and_never_inherits_approval(tmp_path):
    from phase_agent.tools.relax_mc_batch.plan import registration_facts
    from phase_agent.tools.workflows.relax_proposal_feedback import propose_with_relax_feedback

    manager, state, config, context, action = prepared(tmp_path)
    proposal = Mock(return_value={**action, "decision_source": "llm_agent"})
    result, _, error = propose_with_relax_feedback(
        {
            "decision_context": {
                "relax_mc_entry_conditions": registration_facts(state, manager, config)
            }
        },
        agent_client=lambda _: None,
        allowed_tools=["prepare_local_batch_files"],
        config=config,
        propose=proposal,
    )
    assert proposal.call_count == 2
    assert error and "修订" in error
    assert not (result.get("parameters") or {}).get("relax_mc_preview")


def test_feasible_action_uses_no_additional_proposal(tmp_path):
    from phase_agent.tools.relax_mc_batch.plan import registration_facts
    from phase_agent.tools.workflows.relax_proposal_feedback import propose_with_relax_feedback

    manager, state, config, context, action = prepared(tmp_path)
    other = next(b for b in manager.data["branches"] if b not in action["target_ids"])
    proposal = Mock(return_value={**action, "target_ids": [other]})
    result, usage, error = propose_with_relax_feedback(
        {
            "decision_context": {
                "relax_mc_entry_conditions": registration_facts(state, manager, config)
            }
        },
        agent_client=lambda _: None,
        allowed_tools=["prepare_local_batch_files"],
        config=config,
        propose=proposal,
    )
    assert proposal.call_count == 1 and usage is None and error is None
    assert result["target_ids"] == [other]


def test_summary_transport_keeps_conflict_feedback_and_mc_evidence(tmp_path):
    from phase_agent.decisions.agent.summary_request import summary_payload, expand_topics
    from phase_agent.tools.relax_mc_batch.plan import registration_facts

    manager, state, config, context, action = prepared(tmp_path)
    facts = registration_facts(state, manager, config)
    feedback = {"code": "relax_inputs_already_registered", "conflicts": facts["blocked_branches"]}
    payload, sources = summary_payload(
        {
            "state": {},
            "decision_context": {
                "relax_mc_entry_conditions": facts,
                "proposal_validation_feedback": feedback,
            },
        }
    )
    assert payload["decision_context"]["proposal_validation_feedback"] == feedback
    assert sources["relax_mc_entry_conditions"] == facts
    expand_topics(payload, sources, ["mc"])
    assert payload["decision_context"]["relax_mc_entry_conditions"] == facts


def test_semantic_evidence_uses_same_registration_facts_as_scientific_entry(tmp_path):
    from phase_agent.graphs.dialogue.evidence import read_topics
    from phase_agent.tools.relax_mc_batch.plan import registration_facts

    manager, state, config, context, action = prepared(tmp_path)
    state["confirmed_config"] = config
    before = deepcopy(state)
    evidence = read_topics(state, ["mc"], manager=manager)
    assert evidence["relax_mc_entry_conditions"] == registration_facts(state, manager, config)
    assert state == before


@pytest.mark.parametrize("message", ["继续", "接着往下做，先把下一步方案给我看"])
def test_real_chat_entry_revises_conflict_and_waits_for_approval(tmp_path, monkeypatch, message):
    import json
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.runtime import studio_runtime
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot

    manager, state, config, context, action = prepared(tmp_path / "inputs")
    path = tmp_path / "state.json"
    session = confirm_config_snapshot(create_config_draft(config), user_confirmed=True)
    state["confirmed_config_version"] = session["confirmed_snapshot"]["config_version"]
    state["confirmed_config"] = session["confirmed_snapshot"]["config"]
    path.write_text(json.dumps(state), encoding="utf-8")
    other = next(b for b in manager.data["branches"] if b not in action["target_ids"])
    calls = []
    proposals = []

    def model(payload):
        calls.append(payload)
        if payload.get("semantic_dialogue"):
            return {"kind": "advance"}
        context = payload.get("decision_context") or {}
        if not context.get("relax_mc_entry_conditions"):
            return {
                "kind": "inspect_evidence",
                "topics": [
                    "configuration",
                    "mc",
                    "tasks",
                    "model",
                    "budget",
                    "convergence",
                    "history",
                ],
            }
        proposals.append(payload)
        return {
            **action,
            "target_ids": [other]
            if context.get("proposal_validation_feedback")
            else action["target_ids"],
        }

    forbidden = Mock(side_effect=AssertionError("unapproved task must not execute"))
    handler = RunWorkflowChatHandler(
        {
            "state_path": str(path),
            "config_session": session,
            "manager": manager,
            "phase_references": context["phase_references"],
            "agent_client": model,
            "run_config": {
                **config,
                "state_path": str(path),
                "ledger_path": str(tmp_path / "ledger.json"),
                "phase_diagram_directory": str(tmp_path / "analysis"),
            },
            "handlers": {"prepare_local_batch_files": forbidden},
        }
    )
    monkeypatch.setattr(studio_runtime, "_handler", handler)
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PHASE_AGENT_RUNTIME_CONFIG", str(runtime))
    text = studio_runtime.send_project_message(message, "semantic-continuation")
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after.get("pending_execution_policies"), text
    assert len(proposals) == 2, [
        (p.get("mode"), list((p.get("decision_context") or {}))) for p in calls
    ]
    saved = next(iter(after["pending_execution_policies"].values()))["agent_proposal"]["raw_action"]
    assert saved["target_ids"] == [other]
    assert after["tasks"] == state["tasks"]
    assert "同意" in text or "审批" in text
    assert "already have Relax tasks" not in text
    forbidden.assert_not_called()
    if message == "继续":
        from pathlib import Path
        from phase_agent.tools.relax_mc_batch.prepare import prepare_batch

        # Approve the exact saved replacement through the same chat entry.
        # Real local input packaging is allowed only in this private fixture.
        handler.workflow_kwargs["handlers"]["prepare_local_batch_files"] = prepare_batch
        before_calls = len(calls)
        approved_text = studio_runtime.send_project_message("同意", "semantic-continuation")
        approved = json.loads(path.read_text(encoding="utf-8"))
        assert len(calls) == before_calls, "approval must reuse the saved model plan"
        assert len(approved.get("relax_mc_batches") or {}) == 1, approved_text
        batch = next(iter(approved["relax_mc_batches"].values()))
        folder = Path(batch["upload_folder"])
        assert folder.name.endswith("_input") and folder.is_dir()
        assert {"submit_all.sh", "run_upload.py", "inputs.zip"} <= {
            p.name for p in folder.iterdir()
        }
        assert (
            Path(batch["expected_results_archive"]).parent.name
            == folder.name.removesuffix("_input") + "_output"
        )
        assert (
            next(t for t in approved["tasks"] if t["task_id"] == state["tasks"][0]["task_id"])
            == state["tasks"][0]
        )
        assert len(approved["tasks"]) == 2

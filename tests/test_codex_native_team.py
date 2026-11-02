from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

from phase_agent.runtime.local_agent_control import LocalAgentControl


def make_control(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    control = LocalAgentControl(SimpleNamespace(state_path=state, lock=RLock()))
    context = {
        "context_version": "v1",
        "context_digest": "d1",
        "config_version": "c1",
        "model_version": "m1",
        "project": {"project_path": str(tmp_path)},
        "state": {"current_project": {"project_path": str(tmp_path)}},
        "decision_context": {"evidence_catalog": {"references": {"e1": {}}}},
        "allowed_tools": ["pause_search"],
    }
    control.codex_context = lambda: deepcopy(context)
    return control, context


def team(control):
    from phase_agent.agents.strategic.codex_team import CodexTeam

    return CodexTeam(control)


def advice(view, role="exploration"):
    return {
        "role": role,
        "snapshot_id": view["snapshot"]["snapshot_id"],
        "assessment": "coverage needs review",
        "recommendation": "defer",
        "options": [
            {"tool": "pause_search", "reason": "review", "target_ids": [], "parameters": {}}
        ],
        "evidence_refs": ["e1"],
        "expected_outcome": "review first",
        "limitations": "no new results",
    }


def prepared(tmp_path):
    control, context = make_control(tmp_path)
    t = team(control)
    view = t.start(question="compare next steps", consult=["exploration"])
    t.advise(view["team_id"], 1, advice(view))
    return control, context, t, view


def test_team_requests_share_evidence_and_never_write_scientific_state(tmp_path):
    control, _ = make_control(tmp_path)
    before = control.state_path.read_bytes()
    view = team(control).start(question="compare routes", consult=["exploration", "reliability"])
    assert set(view["requests"]) == {"exploration", "reliability"}
    for role, request in view["requests"].items():
        assert request["agent_role"] == role
        assert request["allowed_tools"] == []
        assert request["evidence_snapshot"]["snapshot_id"] == view["snapshot"]["snapshot_id"]
    assert control.state_path.read_bytes() == before
    assert team(control).read()["team_id"] == view["team_id"]


def test_team_builds_context_once_per_locked_operation(tmp_path):
    from unittest.mock import Mock

    control, _ = make_control(tmp_path)
    control.codex_context = Mock(wraps=control.codex_context)
    t = team(control)
    view = t.start(question="review", consult=["exploration"])
    assert control.codex_context.call_count == 1
    t.advise(view["team_id"], 1, advice(view))
    assert control.codex_context.call_count == 2
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    assert control.codex_context.call_count == 3
    t.feedback(view["team_id"], 1, "accept", "accept")
    assert control.codex_context.call_count == 4
    t.read()
    assert control.codex_context.call_count == 5


def test_team_requires_returned_training_handoff_before_consultation(tmp_path):
    import json

    control, _ = make_control(tmp_path)
    returned = tmp_path / "training" / "results"
    returned.mkdir(parents=True)
    (returned / "training.finished.json").write_text("{}", encoding="utf-8")
    state = {
        "remote_finetune_jobs": {
            "J": {"directory": str(returned.parent), "status": "inputs_prepared"}
        }
    }
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="training_recovery_required.*phase_recover"):
        team(control).start(question="review", consult=[])
    assert not (tmp_path / "codex_team.json").exists()
    job = state["remote_finetune_jobs"]["J"]
    job.update(
        returned_results={"fingerprint": "F"}, training_handoff={"training_fingerprint": "old"}
    )
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="training_recovery_required"):
        team(control).start(question="review", consult=[])
    job.update(
        status="results_received",
        returned_results={"fingerprint": "F"},
        training_handoff={"training_fingerprint": "F", "stage": "awaiting_activation_approval"},
    )
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    assert team(control).start(question="review", consult=[])["status"] == "consulting"


def test_activation_reason_comes_only_from_real_accept_feedback(tmp_path):
    from phase_agent.agents.simulation.agent import action_hash, validate_plan

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["update_mlip"]
    t = team(control)
    view = t.start(question="review candidate", consult=[])
    action = {
        "operation": "activate_mlip_candidate",
        "tool": "update_mlip",
        "parameters": {
            "action": "ACTIVATE_CANDIDATE_MODEL",
            "candidate_model_version": "candidate",
        },
    }
    plan = {
        "action_hash": action_hash(action),
        "rationale": "local switch only",
        "steps": [
            {"kind": "preflight", "description": "require explicit approval"},
            {"kind": "invoke", "tool": "update_mlip", "description": "switch approved candidate"},
            {"kind": "verify", "description": "verify identity and no new computation"},
        ],
        "checks": ["no automatic approval"],
        "failure_policy": "return_to_strategic",
    }
    t.draft(view["team_id"], 1, action, rationale="review CV candidate", simulation_plan=plan)
    before = t._action(t._load())
    assert "user_approval_reason" not in before["parameters"]
    accepted = t.feedback(
        view["team_id"], 1, "accept", "同意切换，采用新模型继续探索并保留独立测试限制"
    )
    prepared_action = t._action(t._load())
    assert (
        prepared_action["parameters"]["user_approval_reason"] == accepted["events"][-1]["feedback"]
    )
    assert t._load()["draft"]["action"]["parameters"] == action["parameters"]
    validate_plan(prepared_action["_simulation_plan"], prepared_action["_simulation_intent"])
    assert accepted["execution_approved"] is False
    assert control.state_path.read_text(encoding="utf-8") == "{}"
    revised = t.feedback(view["team_id"], 1, "revise", "重新核对候选")
    t.draft(
        view["team_id"],
        revised["revision"],
        action,
        rationale="revised review",
        simulation_plan=plan,
    )
    assert "user_approval_reason" not in t._action(t._load())["parameters"]


def test_draft_waits_for_human_and_revision_reconsults(tmp_path):
    control, _, t, view = prepared(tmp_path)
    tid = view["team_id"]
    draft = t.draft(tid, 1, {"tool": "pause_search"}, rationale="review uncertainty")
    assert draft["status"] == "awaiting_human_input"
    revised = team(control).feedback(tid, 1, "revise", "先比较补DFT，不新增branch")
    assert revised["revision"] == 2
    assert revised["advice"] == []
    assert revised["draft"] is None
    assert "先比较补DFT" in revised["requests"]["exploration"]["instruction"]
    assert revised["snapshot"]["snapshot_id"] != view["snapshot"]["snapshot_id"]
    with pytest.raises(ValueError, match="revision"):
        t.advise(tid, 1, advice(view))


def test_accept_only_unlocks_frozen_proposal_not_execution(tmp_path):
    control, _, t, view = prepared(tmp_path)
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="need review")
    accepted = t.feedback(view["team_id"], 1, "accept", "同意这个方向")
    assert accepted["status"] == "ready_to_propose"
    assert accepted["execution_approved"] is False
    assert control.state_path.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("kind", ["context_digest", "model_version", "config_version", "project"])
def test_changed_facts_cannot_reuse_expert_opinion(tmp_path, kind):
    control, context, t, view = prepared(tmp_path)
    context[kind] = {"project_path": "another"} if kind == "project" else "changed"
    with pytest.raises(ValueError, match="stale"):
        t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")


def test_bad_role_reference_tool_and_duplicate_are_rejected(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    view = t.start(question="review", consult=["exploration"])
    value = advice(view)
    value["evidence_refs"] = ["invented"]
    with pytest.raises(ValueError, match="unavailable"):
        t.advise(view["team_id"], 1, value)
    value = advice(view)
    value["options"][0]["tool"] = "update_mlip"
    with pytest.raises(ValueError, match="scope"):
        t.advise(view["team_id"], 1, value)
    with pytest.raises(ValueError, match="requested"):
        t.advise(view["team_id"], 1, advice(view, "reliability"))
    t.advise(view["team_id"], 1, advice(view))
    assert t.advise(view["team_id"], 1, advice(view))["advice"] == [advice(view)]
    changed = advice(view)
    changed["assessment"] = "different"
    with pytest.raises(ValueError, match="already"):
        t.advise(view["team_id"], 1, changed)


def test_missing_expert_and_unknown_memory_cannot_form_draft(tmp_path):
    control, _ = make_control(tmp_path)
    t = team(control)
    view = t.start(question="review", consult=["exploration"])
    with pytest.raises(ValueError, match="missing"):
        t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    t.advise(view["team_id"], 1, advice(view))
    with pytest.raises(ValueError, match="memory"):
        t.draft(
            view["team_id"], 1, {"tool": "pause_search"}, rationale="review", memory_ids=["fake"]
        )


def test_paused_team_blocks_proposal_but_can_restart(tmp_path):
    control, _, t, view = prepared(tmp_path)
    t.feedback(view["team_id"], 1, "pause", "暂缓")
    assert t.read()["status"] == "paused"
    with pytest.raises(ValueError, match="review"):
        control.propose_codex({"tool": "pause_search"}, context_version="v1", context_digest="d1")
    new = t.start(question="重新讨论", consult=[])
    assert new["team_id"] != view["team_id"]


def test_unreviewed_or_changed_proposal_cannot_bypass_team(tmp_path):
    control, _, t, view = prepared(tmp_path)
    with pytest.raises(ValueError, match="review"):
        control.propose_codex({"tool": "pause_search"}, context_version="v1", context_digest="d1")
    t.draft(view["team_id"], 1, {"tool": "pause_search", "reason": "review"}, rationale="review")
    t.feedback(view["team_id"], 1, "accept", "接受")
    with pytest.raises(ValueError, match="frozen"):
        control.propose_codex(
            {"tool": "pause_search", "reason": "different"},
            context_version="v1",
            context_digest="d1",
        )


def test_team_id_cannot_escape_project_directory(tmp_path):
    control, _ = make_control(tmp_path)
    with pytest.raises(ValueError, match="team_id"):
        team(control).read("../outside")


def test_team_mcp_strict_contracts(tmp_path):
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    control, _ = make_control(tmp_path)
    server = CodexMcpServer(SimpleNamespace(control=control))

    def call(name, args):
        return server.handle(
            {"id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
        )

    result = call(
        "phase_team_start", {"question": "review", "consult": ["exploration", "exploration"]}
    )
    assert "error" in result
    assert "error" in call("phase_memory_review", {"proposal_id": "m1", "approved": "yes"})
    assert "error" in call("phase_team_start", {"question": "review", "extra": "no"})


def test_successful_submission_is_frozen_and_never_replayed(tmp_path):
    control, _, t, view = prepared(tmp_path)
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    t.feedback(view["team_id"], 1, "accept", "接受方向")
    calls = []

    def submit(action, **kwargs):
        calls.append(deepcopy(action))
        return {"reply": {"status": "awaiting_approval", "record_id": "p1", "submitted": False}}

    control._propose_codex = submit
    control.pending = lambda: {"count": 1, "pending": [{"plan_id": "p1"}]}
    result = team(control).propose(view["team_id"], 1)
    assert result["pending"]["count"] == 1
    assert len(calls) == 1
    assert calls[0]["_decision_team"]["human_review"][-1]["kind"] == "human_accept"
    with pytest.raises(ValueError, match="review"):
        team(control).propose(view["team_id"], 1)
    assert len(calls) == 1


def test_uncertain_submission_does_not_retry_or_start_another_discussion(tmp_path):
    control, _, t, view = prepared(tmp_path)
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    t.feedback(view["team_id"], 1, "accept", "接受方向")

    def fail(*args, **kwargs):
        kwargs["_on_dispatch"]()
        raise OSError("unknown response after dispatch")

    control._propose_codex = fail
    with pytest.raises(OSError):
        t.propose(view["team_id"], 1)
    assert team(control).read()["status"] == "submission_uncertain"
    with pytest.raises(ValueError, match="review"):
        team(control).propose(view["team_id"], 1)
    with pytest.raises(ValueError, match="active"):
        team(control).start(question="another", consult=[])


def test_actual_memory_review_stales_existing_team_and_keeps_user_guidance_local(tmp_path):
    from phase_agent.runtime.codex.context import build_codex_context
    from phase_agent.runtime.codex.team import call_team_tool
    from phase_agent.tools.step_runner.file_protocol import read_json, write_json

    control, _ = make_control(tmp_path)
    control.chat_handler.workflow_kwargs = {
        "config_session": {"config": {"agent": {"allowed_tools": ["pause_search"]}}}
    }
    control.codex_context = lambda: build_codex_context(control.chat_handler)
    t = team(control)
    view = t.start(question="review memory", consult=["exploration"])
    queued = call_team_tool(
        control,
        "phase_memory_propose",
        {
            "record": {
                "scope": "user",
                "category": "search_rules",
                "statement": "优先小批次验证",
                "maturity": "heuristic",
                "evidence_refs": [],
            }
        },
    )
    assert queued["status"] == "pending_review"
    current = control.codex_context()
    assert current["decision_context"]["memory_context"]["approved_knowledge"] == []
    call_team_tool(
        control, "phase_memory_review", {"proposal_id": queued["proposal_id"], "approved": True}
    )
    assert t.read()["stale"] is True
    revised = t.feedback(view["team_id"], 1, "revise", "本轮先检查DFT")
    memory = revised["snapshot"]["facts"]["decision_context"]["memory_context"]
    assert memory["approved_knowledge"][0]["statement"] == "优先小批次验证"
    assert (
        "knowledge:" + memory["approved_knowledge"][0]["knowledge_id"]
        in revised["snapshot"]["available_evidence_refs"]
    )
    state = read_json(control.state_path)
    assert all(
        "本轮先检查DFT" not in row["statement"] for row in state["decision_memory"]["records"]
    )
    state["decision_memory"]["version"] += 1
    write_json(control.state_path, state)
    with pytest.raises(ValueError, match="stale"):
        t.advise(view["team_id"], 2, advice(revised))


def test_reviewed_memory_is_filtered_by_system_model_status_and_approval():
    from phase_agent.persistence.memory.decision_context import decision_memory_context

    base = {
        "knowledge_id": "good",
        "scope": "project",
        "system_id": "s1",
        "category": "search_rules",
        "statement": "review near hull",
        "status": "active",
        "maturity": "heuristic",
        "applicability": {},
        "approved_by_user": True,
    }
    variants = [
        base,
        {**base, "knowledge_id": "wrong-system", "system_id": "s2"},
        {**base, "knowledge_id": "pending", "approved_by_user": False},
        {**base, "knowledge_id": "superseded", "status": "superseded"},
        {**base, "knowledge_id": "wrong-model", "model_dependent": True, "model_version": "m0"},
    ]
    state = {
        "system_id": "s1",
        "active_model_version": "m1",
        "decision_memory": {"version": 3, "records": variants},
    }
    result = decision_memory_context(state)
    assert result["version"] == 3
    assert [row["knowledge_id"] for row in result["approved_knowledge"]] == ["good"]


def test_preflight_failure_is_revisable_and_never_marked_uncertain(tmp_path):
    control, context, t, view = prepared(tmp_path)
    context["required_next_tool"] = "generate_branches"
    # The frozen context in this unit fixture retains the same facts/digest;
    # the normal preflight still independently rejects the wrong stage tool.
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    t.feedback(view["team_id"], 1, "accept", "接受方向")
    with pytest.raises(ValueError, match="codex_stage_requires"):
        t.propose(view["team_id"], 1)
    assert t.read()["status"] == "ready_to_propose"
    assert t.feedback(view["team_id"], 1, "revise", "重新比较当前阶段")["revision"] == 2
    assert control.state_path.read_text(encoding="utf-8") == "{}"


def test_existing_pending_plan_is_not_relabelled_as_a_new_team_decision(tmp_path):
    from phase_agent.tools.step_runner.file_protocol import write_json

    control, _ = make_control(tmp_path)
    write_json(control.state_path, {"pending_execution_policies": {"p1": {"record_id": "r1"}}})
    with pytest.raises(ValueError, match="existing_pending_plan"):
        team(control).start(question="new proposal", consult=[])


def test_uncertain_submission_with_exact_durable_record_can_resume_without_replay(tmp_path):
    from phase_agent.tools.step_runner.file_protocol import write_json

    control, _, t, view = prepared(tmp_path)
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    t.feedback(view["team_id"], 1, "accept", "接受方向")

    def fail(action, **kwargs):
        kwargs["_on_dispatch"]()
        write_json(
            control.state_path,
            {
                "action_records": [
                    {
                        "record_id": "r1",
                        "status": "completed",
                        "agent_proposal": {"raw_action": deepcopy(action)},
                    }
                ]
            },
        )
        raise OSError("lost response")

    control._propose_codex = fail
    with pytest.raises(OSError):
        t.propose(view["team_id"], 1)
    assert t.read()["recorded_submission"]["record_id"] == "r1"
    new = t.start(question="next decision", consult=[])
    assert new["team_id"] != view["team_id"]
    assert t.read(view["team_id"])["status"] == "submitted"
    with pytest.raises(ValueError, match="archived"):
        t.propose(view["team_id"], 1)


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "rejected", "submitted": False, "validation": {"valid": False}},
        {"status": "not_configured", "submitted": False, "steps_executed": 0},
    ],
)
def test_proven_graph_rejection_can_be_revised_without_replay(tmp_path, reply):
    control, _, t, view = prepared(tmp_path)
    t.draft(view["team_id"], 1, {"tool": "pause_search"}, rationale="review")
    t.feedback(view["team_id"], 1, "accept", "接受方向")

    def reject(action, **kwargs):
        kwargs["_on_dispatch"]()
        return {"reply": reply}

    control._propose_codex = reject
    assert t.propose(view["team_id"], 1)["reply"] == reply
    assert t.read()["status"] == "ready_to_propose"
    assert t.feedback(view["team_id"], 1, "revise", "修正方案")["revision"] == 2


def test_concurrent_clients_cannot_create_two_active_discussions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    left, _ = make_control(tmp_path)
    right, _ = make_control(tmp_path)

    def start(control):
        try:
            return team(control).start(question="review", consult=[])["team_id"]
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, [left, right]))
    assert sum(value.startswith("team-") for value in results) == 1
    assert sum("codex_team_active" in value for value in results) == 1


@pytest.mark.parametrize(
    "invalid", [None, "reason", "version", "model", "job", "scope", "validation"]
)
def test_canonical_activation_receipt_settles_training_handoff(tmp_path, invalid):
    import json
    from phase_agent.agents.simulation.agent import action_hash

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["update_mlip", "pause_search"]
    t = team(control)
    view = t.start(question="review candidate", consult=[])
    action = {
        "operation": "activate_mlip_candidate",
        "tool": "update_mlip",
        "parameters": {"action": "ACTIVATE_CANDIDATE_MODEL", "candidate_model_version": "m2"},
    }
    if invalid == "scope":
        action["parameters"]["unreviewed_operation"] = True
    plan = {
        "action_hash": action_hash(action),
        "rationale": "switch only",
        "steps": [
            {"kind": "preflight", "description": "verify approval"},
            {"kind": "invoke", "tool": "update_mlip", "description": "switch"},
            {"kind": "verify", "description": "verify activation receipt"},
        ],
        "checks": ["no compute"],
        "failure_policy": "return_to_strategic",
    }
    t.draft(view["team_id"], 1, action, rationale="review", simulation_plan=plan)
    t.feedback(view["team_id"], 1, "accept", "approve candidate direction")
    session = t._load()
    session["status"] = "submission_uncertain"
    t._save(session)
    model = {"version": "m2", "sha256": "a" * 64}
    state = {
        "active_model_version": "m2",
        "active_model": model,
        "candidate_models": {
            "m2": {
                "status": "activated",
                "model": model,
                "old_model_version": "m1",
                "dft_data_version": "D",
                "validation_data_version": "V",
            }
        },
        "model_activation_history": [
            {
                "new_model_version": "m2",
                "old_model_version": "m1",
                "approval_reason": "explicit final approval",
                "dft_data_version": "D",
                "validation_data_version": "V",
            }
        ],
        "remote_finetune_jobs": {
            "J": {"activated": True, "training_handoff": {"candidate_model_version": "m2"}}
        },
    }
    if invalid == "reason":
        state["model_activation_history"][0]["approval_reason"] = ""
    if invalid == "version":
        state["model_activation_history"][0]["new_model_version"] = "m3"
    if invalid == "model":
        state["active_model"] = {"version": "m2", "sha256": "b" * 64}
    if invalid == "job":
        state["remote_finetune_jobs"]["J"]["activated"] = False
    if invalid == "validation":
        state["model_activation_history"][0]["validation_data_version"] = "other"
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    if invalid is not None:
        assert t.read()["recorded_submission"] is None
        with pytest.raises(ValueError, match="codex_team_active"):
            t.start(question="next decision", consult=[])
    else:
        recorded = t.read()["recorded_submission"]
        assert recorded["status"] == "activated"
        assert recorded["record_id"] == "review:model_activation:J"
        assert t.start(question="next decision", consult=[])["team_id"] != view["team_id"]
        assert t.read(view["team_id"])["status"] == "submitted"

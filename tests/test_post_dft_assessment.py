"""Synthetic round evidence and formal training dispatch; no production I/O."""
from copy import deepcopy

from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment
from phase_agent.tools.workflows.create_workflow_handlers import create_workflow_handlers
from phase_agent.runtime.chat_state_presentation import brief_chat_state
from tests.test_dft_comparison_csv import add_result


def test_closed_round_uses_original_model_metrics():
    state = {}
    add_result(state, energy_error=1., force_error=.2)
    assessment = post_dft_assessment(state, {"mlip": {"version": "m1"}})
    assert assessment["status"] == "evaluated"
    assert assessment["metrics"]["energy_per_atom"]["rmse"] == .5
    assert abs(assessment["metrics"]["forces"]["mae"] - .2) < 1e-10
    assert "DFT 本轮误差已评估" in brief_chat_state(state)
    assert post_dft_assessment(state, {"mlip": {"version": "m2"}}) is None


def test_partial_round_requires_wait_or_exact_waiver():
    state = {}
    add_result(state)
    pending = {**state["tasks"][0], "task_id": "missing", "status": "pending"}
    state["tasks"].append(pending)
    assert post_dft_assessment(state) is None
    pending["recovery_wait_waived"] = True
    assessment = post_dft_assessment(state)
    assert assessment["waived_task_ids"] == ["missing"]
    assert pending["status"] == "pending"
    state["post_dft_decided_rounds"] = [assessment["scope_key"]]
    assert post_dft_assessment(state) is None


def test_missing_comparison_not_zero_and_no_training():
    state = {}
    add_result(state, evaluator=False)
    assessment = post_dft_assessment(state)
    assert assessment["status"] == "evaluation_incomplete"
    assert assessment["metrics"]["energy_per_atom"]["mae"] is None
    called = []
    result = create_workflow_handlers()["update_mlip"](
        action={"parameters": {}}, context={"event_state": state,
        "effective_config": {"mlip_finetune": {"enabled": True}},
        "model_update_handler": lambda **kw: called.append(kw)})
    assert result["status"] == "not_configured"
    assert not called


def test_training_callback_connected_but_disabled_config_blocks():
    state = {}
    add_result(state)
    called = []
    def callback(**kw):
        called.append(kw)
        return {"status": "awaiting_activation_approval", "state": deepcopy(kw["state"])}
    handler = create_workflow_handlers()["update_mlip"]
    context = {"event_state": state, "effective_config": {}, "model_update_handler": callback}
    assert handler(action={}, context=context)["status"] == "not_configured"
    context["effective_config"] = {"mlip_finetune": {"enabled": True, "training": {"minimum_new_dft_records": 1}}}
    assert handler(action={}, context=context)["status"] == "awaiting_activation_approval"
    assert called[0]["trigger"]["action"] == "RETRAIN_MLIP"


def test_retry_missing_comparison_reuses_labels_and_cached_predictions():
    from phase_agent.tools.workflows.refresh_dft_comparisons import refresh_dft_comparisons
    state = {}
    add_result(state, evaluator=False)
    before = deepcopy(state.get("new_dft_records"))
    calls = []
    def evaluator(**kw):
        calls.append(kw)
        return {"model_version": "m1", "energy": -7., "energy_unit": "eV",
                "forces": [[.2, -.1, .4]] * 2, "forces_unit": "eV/angstrom"}
    updated = refresh_dft_comparisons(state, assessment=post_dft_assessment(state), evaluator=evaluator)
    assert post_dft_assessment(updated)["status"] == "evaluated"
    assert len(updated["dft_dataset_records"]) == 1
    assert updated["new_dft_records"] == before
    refresh_dft_comparisons(updated, assessment=post_dft_assessment(updated), evaluator=evaluator)
    assert len(calls) == 1


def test_closed_dft_routes_to_llm_not_old_mc_dft(monkeypatch):
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
    from phase_agent.configuration.session.create_config_draft import create_config_draft
    from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
    config = default_layered_search_config()
    session = confirm_config_snapshot(create_config_draft(config), user_confirmed=True)
    state = {}
    add_result(state)
    state["tasks"].append({"task_id": "mc", "stage": "deep_search", "model_version": "m1",
                           "segment_index": 1, "status": "completed"})
    state["mc_second_round_allocations"] = [{"model_version": "m1", "task_ids": ["mc"]}]
    state["pending_execution_policies"] = {"old-branch": {"agent_proposal": {
        "raw_action": {"tool": "generate_branches", "_post_dft_review_version": 1,
                       "_post_dft_scope_key": "different-round"}}}}
    seen = []
    def proposal(snapshot, **kw):
        seen.append((snapshot, kw))
        from tests.test_post_dft_review import review
        assessment_review = review()
        assessment_review["choice"] = "convergence"
        return {"tool": "check_convergence", "task_key": "review-round", "target_ids": [],
                "parameters": {}, "budget": 0., "reason": "review evidence", "decision_source": "llm",
                "post_dft_review": assessment_review}
    monkeypatch.setattr("phase_agent.tools.workflows.run_tool_step.propose_agent_tool_action", proposal)
    result = run_tool_step(state, session, registry=create_tool_registry(create_workflow_handlers()),
        agent_client=lambda **kw: None, execution_mode="interactive",
        invocation_id="old-branch", human_feedback={"decision": "approve"},
        context={"effective_config": {"mlip": {"version": "m1"}}, "user_message": "继续"})
    assert result["status"] == "awaiting_approval"
    assert seen[0][0]["decision_context"]["post_dft_assessment"]["status"] == "evaluated"
    assert "generate_branches" in seen[0][1]["allowed_tools"]
    assert "update_mlip" in seen[0][1]["allowed_tools"]
    assert "select_dft_candidates" in seen[0][1]["allowed_tools"]
    assert result["agent_proposal"]["raw_action"]["_post_dft_scope_key"] == post_dft_assessment(state)["scope_key"]


def test_reply_reports_metrics_and_directory_before_next_proposal():
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
    state = {}
    add_result(state)
    assessment = post_dft_assessment(state)
    state["dft_result_exports"] = {"round": {"round_scope": assessment["scope"], "directory": "outputs/m1/Search-group-0001/DFT-round-0001"}}
    reply = format_workflow_reply({"status": "completed", "state": state}, "state.json")
    assert "回收 1/1" in reply and "合格配对 1" in reply
    assert "MAE/RMSE" in reply and "DFT-round-0001" not in reply
    full_reply = format_workflow_reply({"status": "completed", "state": state}, "state.json", verbose=True)
    assert "DFT-round-0001" in full_reply
    assert reply.index("MAE/RMSE") < reply.index("已完成")
    assert "当前微调未启用" not in reply

from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal
from phase_agent.tools.local.training_followup_plan import followup_errors


def test_specific_plan_is_reused_without_second_agent_decision():
    plan = {"tool": "adjust_strategy", "parameters": {"scope": "force_interface"},
            "target_ids": [], "budget": 0.0, "reason": "核查预测接口", "expected_purpose": "核对力尺度"}
    direction = {"direction": "other", "plan": plan, "training_fingerprint": "same"}
    state = {"active_model_version": "base", "remote_finetune_jobs": {"j": {
        "status": "results_received", "original_model_version": "base",
        "training_handoff": {"direction_status": "approved", "direction_proposal": direction}}}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not ask Agent to choose direction again")
    result = select_initial_proposal(current=state, mode="autonomous",
        config={"agent": {"allowed_tools": ["adjust_strategy", "update_mlip"]}},
        context={}, decision_state={}, registry={"adjust_strategy": {"handler": lambda: None}},
        agent_client=forbidden, invocation_id="specific-plan", propose=forbidden, revise=forbidden,
        prepare_debug=forbidden, is_verified_second=lambda *args: False,
        mc_block=forbidden, model_failed=lambda action: False,
        record_id_factory=lambda *args: "specific-plan")
    action = result["proposal"]["raw_action"]
    assert action["parameters"] == plan["parameters"]
    assert action["decision_source"] == "saved_agent_direction_plan"
    assert action["_approved_direction_hash"]


def test_plan_outside_direction_is_rejected():
    assert followup_errors("other", {"tool": "update_mlip", "reason": "train", "expected_purpose": "train"})
    assert followup_errors("supplement_dft", None)

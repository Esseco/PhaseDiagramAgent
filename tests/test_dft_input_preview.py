from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs
from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
from phase_agent.runtime.agent_api import format_workflow_reply


def test_empty_plan_cannot_be_approved():
    preview, error = preview_dft_inputs({"parameters": {"decisions": []}}, {}, {})
    assert preview is None and error


def test_concrete_plan_has_nonzero_cost():
    config = {"qbc": default_dft_decision_config(), "dft": {"selection": {"max_relax_fraction": 1}}}
    state = {"confirmed_config_version": "v", "qbc_candidates": [{"candidate_id": "s", "atom_count": 48}]}
    action = {"parameters": {"decisions": [{"candidate_id": "s", "action": "DFT_RELAX", "reason": "凸包基态"}]}}
    preview, error = preview_dft_inputs(action, state, config)
    assert error is None and preview["task_count"] == 1 and preview["relative_cost"] > 0


def test_chat_names_input_generation():
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": {
        "recommended_action": "select_dft_candidates", "estimated_cost": {"estimated_total_cost": 10},
        "action_parameters": {"dft_input_preview": {"task_count": 3, "reasons": ["凸包基态"],
            "selected_structures": [{"candidate_id": "s", "phase": "O3", "x_Na_per_O2": 0.5, "ehull": 0.01}]}}}}, "state.json")
    assert "生成 3 个 DFT 结构优化输入" in text and "QBC" in text


def test_rejected_event_reason_reaches_chat():
    text = format_workflow_reply({"status": "rejected", "events": [
        {"status": "rejected", "reason": "当前配置只允许最多 10% 的 DFT 候选做结构优化"}]}, "state.json")
    assert "10%" in text and "执行动作被拒绝" not in text


def test_mixed_plan_preserves_single_point_first_policy():
    config = {"qbc": default_dft_decision_config(), "dft": {"selection": {"max_relax_fraction": 0.1}}}
    state = {"confirmed_config_version": "v", "qbc_candidates": [
        {"candidate_id": f"s{i}", "atom_count": 48} for i in range(10)]}
    decisions = [{"candidate_id": f"s{i}", "action": "DFT_RELAX" if i == 0 else "DFT_SINGLE_POINT",
                  "reason": "凸包校验"} for i in range(10)]
    preview, error = preview_dft_inputs({"parameters": {"decisions": decisions}}, state, config)
    assert error is None
    assert preview["single_point_count"] == 9 and preview["relax_count"] == 1
    assert len(preview["workload"]) == 2

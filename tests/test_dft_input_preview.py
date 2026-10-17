from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs
from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.runtime.agent_api import format_workflow_reply
import pytest


def candidate(candidate_id):
    return {
        "candidate_id": candidate_id,
        "atom_count": 48,
        "phase": "O3",
        "x_Na_per_O2": 0.5,
        "predicted_Ehull": 0.01,
        "geometry_optimization_evidence": {
            "source_task_id": "relax-" + candidate_id,
            "residual_force_ev_per_angstrom": 0.12,
            "force_tolerance_ev_per_angstrom": 0.05,
        },
    }


def test_empty_plan_cannot_be_approved():
    preview, error = preview_dft_inputs({"parameters": {"decisions": []}}, {}, {})
    assert preview is None and error


def test_concrete_plan_has_nonzero_cost():
    config = {"qbc": default_dft_decision_config(), "dft": {"selection": {"max_relax_fraction": 1}}}
    state = {"confirmed_config_version": "v", "qbc_candidates": [candidate("s")]}
    action = {
        "parameters": {
            "decisions": [{"candidate_id": "s", "action": "DFT_RELAX", "reason": "凸包基态"}]
        }
    }
    preview, error = preview_dft_inputs(action, state, config)
    assert error is None and preview["task_count"] == 1 and preview["relative_cost"] > 0


def test_chat_names_input_generation(tmp_path):
    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "confirmed_config_version": "v",
        "qbc_candidates": [candidate(f"s{i}") for i in range(3)],
    }
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": f"s{i}", "action": "DFT_SINGLE_POINT", "reason": "凸包校验"}
                for i in range(3)
            ]
        },
    }
    preview, error = preview_dft_inputs(action, state, {"qbc": default_dft_decision_config()})
    assert error is None
    action["parameters"]["dft_input_preview"] = preview
    result = {
        "status": "awaiting_approval",
        "state": state,
        "agent_proposal": {
            "raw_action": action,
            "recommended_action": "select_dft_candidates",
            "action_parameters": action["parameters"],
        },
    }
    text = format_workflow_reply(result, "state.json")
    assert "选中 3 个结构" in text and "DFT单点 3 个" in text
    assert "提交超算属于后续操作" in text
    from phase_agent.tools.workflows.dft_selection_artifacts import proposal_artifacts
    from phase_agent.tools.workflows.branch_review_view import raw_record
    from pathlib import Path
    import json

    artifacts = proposal_artifacts(result["agent_proposal"], state)
    detail = raw_record(json.loads(Path(artifacts["json"]).read_text(encoding="utf-8")))
    assert "DFT 选点" in text
    assert detail["summary"]["selected_count"] == 3
    assert {row["candidate_id"] for row in detail["decisions"]} == {"s0", "s1", "s2"}
    assert detail["input_preview"] == preview


def test_rejected_event_reason_reaches_chat():
    text = format_workflow_reply(
        {
            "status": "rejected",
            "events": [
                {"status": "rejected", "reason": "当前配置只允许最多 10% 的 DFT 候选做结构优化"}
            ],
        },
        "state.json",
    )
    assert "10%" in text and "执行动作被拒绝" not in text


def test_mixed_plan_preserves_single_point_first_policy():
    config = {
        "qbc": default_dft_decision_config(),
        "dft": {"selection": {"max_relax_fraction": 0.1}},
    }
    state = {
        "confirmed_config_version": "v",
        "qbc_candidates": [candidate(f"s{i}") for i in range(10)],
    }
    decisions = [
        {
            "candidate_id": f"s{i}",
            "action": "DFT_RELAX" if i == 0 else "DFT_SINGLE_POINT",
            "reason": "凸包校验",
        }
        for i in range(10)
    ]
    preview, error = preview_dft_inputs({"parameters": {"decisions": decisions}}, state, config)
    assert error is None
    assert preview["single_point_count"] == 9 and preview["relax_count"] == 1
    assert len(preview["workload"]) == 2


@pytest.mark.parametrize("missing", ["predicted_Ehull", "geometry_optimization_evidence"])
def test_relax_without_phase_or_geometry_evidence_is_rejected(missing):
    row = candidate("s")
    row.pop(missing)
    state = {"confirmed_config_version": "v", "qbc_candidates": [row]}
    action = {
        "parameters": {
            "decisions": [{"candidate_id": "s", "action": "DFT_RELAX", "reason": "凸包基态"}]
        }
    }
    preview, error = preview_dft_inputs(action, state, {"qbc": default_dft_decision_config()})
    assert preview is None and "dft_relax_requires_hull_and_geometry_evidence" in error

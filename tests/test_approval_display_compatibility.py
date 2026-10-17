from copy import deepcopy

from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.runtime.review_presentation import review_card
import pytest


def test_preview_only_saved_dft_proposal_reports_incomplete_plan_without_zero_or_approval():
    proposal = {
        "recommended_action": "select_dft_candidates",
        "action_parameters": {
            "dft_input_preview": {
                "task_count": 3,
                "selected_structures": [{"candidate_id": "s", "phase": "O3", "ehull": 0.01}],
            }
        },
    }
    saved = deepcopy(proposal)
    text = format_workflow_reply(
        {"status": "awaiting_approval", "agent_proposal": proposal}, "state.json"
    )
    assert "3" in text and "不可批准" in text and "缺少" in text
    assert "0 个目标" not in text and "回复‘同意’" not in text
    assert proposal == saved


def test_refresh_display_does_not_authorize_undetermined_supplement():
    proposal = {
        "raw_action": {
            "tool": "prepare_local_batch_files",
            "parameters": {
                "refresh_preview": {
                    "candidates": [{"structure_id": "s"}],
                    "maximum_supplemental_cost": 2,
                }
            },
        }
    }
    card = review_card("p", {"agent_proposal": proposal})
    assert "另行审批" in card["approval_boundary"]
    assert "首批及" not in card["approval_boundary"]


@pytest.mark.parametrize("storage", ["raw", "presentation", "both"])
def test_saved_executable_dft_formats_keep_same_scope_and_remain_immutable(storage):
    params = {
        "decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT", "reason": "check"}],
        "dft_input_preview": {"task_count": 1},
    }
    proposal = {"recommended_action": "select_dft_candidates"}
    if storage in {"raw", "both"}:
        proposal["raw_action"] = {"tool": "select_dft_candidates", "parameters": params}
    if storage in {"presentation", "both"}:
        proposal["action_parameters"] = params
    before = deepcopy(proposal)
    text = format_workflow_reply(
        {"status": "awaiting_approval", "agent_proposal": proposal}, "state.json"
    )
    assert "选中 1 个结构" in text and "DFT单点 1 个" in text
    assert "0 个目标" not in text
    assert proposal == before

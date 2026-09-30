from execution_layer.workflows.preview_dft_inputs import stale_dft_preview
from run.open_webui_api import format_workflow_reply


def test_old_preview_requires_refresh():
    proposal = {"recommended_action": "select_dft_candidates", "action_parameters": {
        "dft_input_preview": {"task_count": 15, "candidate_ids": ["s"]}}}
    assert stale_dft_preview(proposal)
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": proposal}, "state.json")
    assert "旧 DFT" in text and "继续" in text and "回复“同意”" not in text


def test_current_preview_is_not_stale():
    proposal = {"recommended_action": "select_dft_candidates", "action_parameters": {
        "dft_input_preview": {"schema_version": 2, "selected_structures": [{"candidate_id": "s"}]}}}
    assert not stale_dft_preview(proposal)

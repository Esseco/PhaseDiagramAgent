from phase_agent.runtime.evidence_reference_presentation import evidence_reference_lines
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def test_approval_reply_exposes_unknown_evidence_without_claiming_execution():
    result = {"status": "awaiting_approval", "agent_proposal": {
        "recommended_action": "pause_search", "raw_action": {
            "evidence_reference_check": {"status": "unknown_references", "missing": ["phase_diagram:mlip:old"]}}}}
    reply = format_workflow_reply(result, "state.json")
    assert "phase_diagram:mlip:old" in reply
    assert "需核对" in reply


def test_missing_and_matched_refs_are_not_scientific_validation():
    assert "未提供" in evidence_reference_lines({"evidence_reference_check": {"status": "no_references"}})[0]
    assert "不代表科学校验通过" in evidence_reference_lines({"evidence_refs": ["a"],
        "evidence_reference_check": {"status": "references_found"}})[0]

from phase_agent.tools.state.execution_receipts import begin_execution, recovery_report
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def test_audited_no_effect_requires_matching_identity_and_evidence(tmp_path):
    path = tmp_path / "state.json"
    identity = {"invocation_id": "interrupted", "config_version": "v1",
                "action_hash": "sha256:" + "a" * 64, "tool": "generate_branches"}
    assert begin_execution(path, identity)["allowed"]
    record = {"resolution": "verified_no_effect", "identity": identity,
              "evidence": {"tasks": 0, "structures": 0}, "reviewed_at": "reviewed"}
    state = {"execution_reconciliations": {"interrupted": record}}
    assert recovery_report(path, state)["status"] == "clear"
    assert not begin_execution(path, identity)["allowed"]
    record["identity"] = {**identity, "config_version": "other"}
    assert recovery_report(path, state)["unsettled"]


def test_unsettled_reply_explains_blocker():
    text = format_workflow_reply({"status": "execution_reconciliation_required"}, "state.json")
    assert "中断记录" in text and "建议：" in text
    assert "回复‘继续’" not in text

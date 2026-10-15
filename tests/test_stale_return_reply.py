from copy import deepcopy
from phase_agent.analysis.state.batch_return_facts import batch_return_facts
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.decisions.agent.summary_request import summary_payload


def facts():
    return {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "b": {
                "bundle_id": "b",
                "model_version": "m1",
                "status": "completed",
                "import_checksum": "verified",
                "plan_checksum": "p",
                "result_summary": {
                    "status": "completed",
                    "model_version": "m1",
                    "task_count": 2,
                    "plan_checksum": "p",
                },
            }
        },
        "tasks": [
            {"model_version": "m1", "stage": "relax_and_feature", "status": "completed"},
            {"model_version": "m1", "stage": "deep_search", "status": "completed"},
        ],
        "phase_diagrams": {
            "mlip": {"model_version": "m1", "status": "completed", "entries": [{"energy": 1}]}
        },
    }


def test_verified_results_override_stale_transfer_failure_without_state_change():
    s = facts()
    before = deepcopy(s)
    text = format_turn_reply("结果包仍未登记；请在接收端修复传输清单大小限制，重新回传。", s)
    assert "已核验并入账，共 2 项" in text and "MLIP 相图已有 1 条" in text
    assert "请在接收端" not in text and "无需为该旧错误重新上传" in text
    assert s == before


def test_mixed_incomplete_failed_or_new_model_returns_do_not_erase_current_problem():
    for field, value in [
        ("status", "prepared"),
        ("return_error", "size error"),
        ("import_checksum", None),
        ("plan_checksum", "mismatch"),
    ]:
        s = facts()
        s["relax_mc_batches"]["b"][field] = value
        assert not batch_return_facts(s)["all_registered_returns_verified"]
    s = facts()
    s["active_model_version"] = "m2"
    assert not batch_return_facts(s)["all_registered_returns_verified"]
    s = facts()
    s["relax_mc_batches"]["new"] = {"model_version": "m1", "status": "prepared"}
    assert not batch_return_facts(s)["all_registered_returns_verified"]


def test_legitimate_history_explanation_and_unrelated_question_are_not_replaced():
    for text in [
        "当时出现 Transfer manifest exceeds size limit，现已修复。",
        "解释一下 QBC 的意义。",
    ]:
        reply = format_turn_reply(text, facts())
        assert text in reply


def test_verified_facts_are_visible_in_initial_summary_and_preserve_receipt():
    s = facts()
    request, sources = summary_payload(
        {
            "mode": "autonomous_search",
            "state": s,
            "decision_context": {},
            "allowed_tools": ["check_convergence"],
        }
    )
    assert request["state"]["batch_return_facts"]["all_registered_returns_verified"]
    assert request["state"]["batch_return_facts"]["reported_returned_tasks"] == 2
    assert sources["relax_mc_batches"] == s["relax_mc_batches"]


def test_receipt_summary_does_not_treat_zero_new_collection_as_no_existing_results():
    s = facts()
    s["result_collection"] = {"recovered_count": 0, "failure_groups": []}
    assert batch_return_facts(s)["reported_returned_tasks"] == 2


def test_existing_approval_is_preserved_and_still_shown():
    s = facts()
    s["pending_execution_policies"] = {
        "p": {
            "agent_proposal": {
                "recommended_action": "check_convergence",
                "reason": "检查已有数据",
                "expected_purpose": "评估下一步",
                "raw_action": {"tool": "check_convergence", "task_key": "check"},
            }
        }
    }
    before = deepcopy(s)
    reply = format_turn_reply("结果包仍未登记", s)
    assert "已核验并入账" in reply and "同意" in reply
    assert s == before

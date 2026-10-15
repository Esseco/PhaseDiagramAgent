from copy import deepcopy
import json
from unittest.mock import patch
import pytest
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.decisions.agent.summary_request import summary_payload
from phase_agent.analysis.state.task_failure_facts import task_failure_facts
from phase_agent.graphs.dialogue.evidence import conversation_facts


def pending():
    return {
        "active_model_version": "m1",
        "pending_execution_policies": {
            "p": {
                "agent_proposal": {
                    "recommended_action": "check_convergence",
                    "expected_purpose": "检查已有结果",
                    "reason": "缺少收敛证据",
                    "raw_action": {"tool": "check_convergence", "task_key": "check"},
                }
            }
        },
    }


def test_status_query_with_pending_plan_has_one_approval_prompt_and_one_read():
    from phase_agent.runtime.plan_queries import pending_plan_reply

    state = pending()
    before = deepcopy(state)
    text = "当前：操作方案待确认。\n待审批方案 1 个：检查收敛。\n下一步：查看该方案后回复“同意”“拒绝”或提出修改；本次查询不会批准。"
    with patch(
        "phase_agent.runtime.plan_queries.pending_plan_reply", wraps=pending_plan_reply
    ) as read:
        reply = format_turn_reply(text, state)
    assert read.call_count == 1 and reply.count("下一步：") == 1
    assert "完成本次动作后停止" in reply and state == before


def test_blocked_plan_is_rendered_once_without_approval_advice():
    state = pending()
    state["pending_execution_policies"]["p"]["project_identity"] = {"changed": True}
    reply = format_turn_reply("本轮未执行：缺少必要结果", state)
    assert reply.count("当前模型或生效设置与方案依据不同") == 1
    assert reply.count("下一步：") == 1 and "回复‘同意’" not in reply


@pytest.mark.parametrize(
    "batch",
    [
        {"model_version": "old", "status": "failed", "return_error": "old manifest error"},
        {"model_version": "m1", "status": "cancelled", "return_error": "cancelled manifest error"},
    ],
)
def test_inactive_error_does_not_label_current_model_as_blocked(batch):
    state = {"active_model_version": "m1", "relax_mc_batches": {"old": batch}}
    assert "回传核对受阻" not in format_turn_reply("当前：等待下一步评估。", state)


def test_current_real_error_is_visible_even_without_pending_task_rows():
    state = {
        "active_model_version": "m1",
        "relax_mc_batches": {
            "current": {
                "model_version": "m1",
                "status": "prepared",
                "return_error": "current failure",
            }
        },
    }
    assert "current failure" in format_turn_reply("当前：等待下一步评估。", state)


def test_projected_receipt_survives_summary_and_model_change_invalidates_it():
    projection = {
        "model_version": "m1",
        "all_registered_returns_verified": True,
        "reported_returned_tasks": 1053,
    }
    p = {
        "mode": "autonomous_search",
        "state": {"active_model_version": "m1", "batch_return_facts": projection},
        "decision_context": {"relax_mc_batches": []},
    }
    short, _ = summary_payload(p)
    assert short["state"]["batch_return_facts"] == projection
    p["state"]["active_model_version"] = "m2"
    short, _ = summary_payload(p)
    assert not short["state"]["batch_return_facts"]["all_registered_returns_verified"]


def test_live_batch_and_model_override_stale_context_and_invalidate_cache_identity():
    from tests.test_stale_return_reply import facts

    s = facts()
    p = {
        "mode": "autonomous_search",
        "state": s,
        "decision_context": {"active_model_version": "old", "relax_mc_batches": []},
    }
    first, _ = summary_payload(p)
    assert first["state"]["batch_return_facts"]["all_registered_returns_verified"]
    p["state"]["relax_mc_batches"]["b"]["return_error"] = "new checksum failure"
    second, _ = summary_payload(p)
    assert not second["state"]["batch_return_facts"]["all_registered_returns_verified"]
    assert first["evidence_revision"] != second["evidence_revision"]


def attempts(status="completed"):
    failed = {
        "task_id": "old",
        "model_version": "m1",
        "stage": "relax_and_feature",
        "structure_id": "s",
        "status": "failed",
        "retry_task_id": "retry",
        "error": "old missing model",
    }
    retry = {**failed, "task_id": "retry", "status": status}
    retry.pop("retry_task_id")
    retry.pop("error")
    return {"active_model_version": "m1", "tasks": [failed, retry]}


@pytest.mark.parametrize("status", ["completed", "pending", "running", "prepared"])
def test_successful_or_registered_retry_is_not_advertised_for_another_retry(status):
    s = attempts(status)
    before = deepcopy(s)
    f = task_failure_facts(s)
    assert not f["actionable_failures"] and s == before
    view = conversation_facts(s, {})
    assert view["recent_failures"] == [] and view["task_counts"]["failed"] == 1
    assert view["task_count_scope"] == "all_recorded_attempts_including_history"


def test_failed_retry_only_advertises_latest_same_scope_attempt():
    s = attempts("failed")
    assert [r["task_id"] for r in task_failure_facts(s)["actionable_failures"]] == ["retry"]
    s["tasks"][1]["structure_id"] = "other"
    assert "old" in [r["task_id"] for r in task_failure_facts(s)["actionable_failures"]]
    s["tasks"][1]["structure_id"] = "s"
    s["tasks"][1]["retry_task_id"] = "old"
    assert len(task_failure_facts(s)["actionable_failures"]) == 2


@pytest.mark.parametrize("content", ["", '{"tool":'])
def test_scientific_length_retry_preserves_reasoning(content):
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client

    bodies = []

    def provider(request, **kwargs):
        bodies.append(json.loads(request.data))
        first = len(bodies) == 1

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "length" if first else "stop",
                                "message": {
                                    "content": content if first else '{"tool":"check_convergence"}',
                                    "reasoning_content": "reasoning" if first else "",
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 8192 if first else 100},
                    }
                ).encode()

        return Response()

    client = create_deepseek_client(api_key="fake", thinking="enabled")
    with patch(
        "phase_agent.decisions.agent.create_deepseek_client.urllib.request.urlopen",
        side_effect=provider,
    ):
        result = client(
            {"mode": "autonomous_search", "output_contracts": {"scientific_decision": {}}}
        )
    assert len(bodies) == 2 and all(b["thinking"] == {"type": "enabled"} for b in bodies)
    assert (
        bodies[1]["max_tokens"] > bodies[0]["max_tokens"]
        and result["_llm_usage"]["reasoning_enabled"]
    )

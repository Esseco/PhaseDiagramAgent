"""Conversation facts and version-bound approval; no intent classifier."""

from pathlib import Path
from copy import deepcopy
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.policy.file_approval import proposal_hash


def _binding_path(handler):
    import hashlib

    conversation = getattr(handler, "conversation_id", None)
    if not conversation:
        return None
    name = hashlib.sha256(str(conversation).encode()).hexdigest()
    return Path(handler.state_path).parent / "presented_plans" / (name + ".json")


def _save_binding(handler, binding):
    from phase_agent.tools.step_runner.file_protocol import write_json

    handler.presented_plan_binding = binding
    path = _binding_path(handler)
    if path is not None:
        write_json(path, {"binding": binding})


def _load_binding(handler):
    from phase_agent.tools.step_runner.file_protocol import read_json

    path = _binding_path(handler)
    if path is not None:
        binding = (read_json(path, {}) or {}).get("binding")
        if binding and binding.get("state_version"):
            from phase_agent.tools.step_runner.file_protocol import content_id

            facts = read_json(handler.state_path, {}) or {}
            summary = build_status_summary(
                facts, config_version=facts.get("confirmed_config_version")
            )
            legacy = content_id(
                {key: value for key, value in summary.items() if key != "summary_id"}, "summary"
            )
            previous_summary = {
                key: value
                for key, value in summary.items()
                if key not in {"summary_id", "project_identity"}
            }
            previous_full = content_id(previous_summary, "summary")
            budget = previous_summary["budget_usage"]
            budget.pop("llm", None)
            budget.setdefault("total_relative_cost", 0.0)
            budget.setdefault("stages", {})
            previous_scientific = content_id(previous_summary, "summary")
            if (
                binding["state_version"] in {legacy, previous_full, previous_scientific}
                and binding["state_version"] != summary["summary_id"]
            ):
                binding = {**binding, "state_version": summary["summary_id"]}
                _save_binding(handler, binding)
        return binding
    return getattr(handler, "presented_plan_binding", None)


def _review_rows(handler, state):
    from phase_agent.runtime.review_requests import pending_reviews

    return pending_reviews(state)


def bind_presented_proposal(handler, state):
    rows = _review_rows(handler, state)
    _save_binding(handler, None)
    if len(rows) == 1:
        _save_binding(
            handler, {key: rows[0][key] for key in ("plan_id", "state_version", "proposal_hash")}
        )


def review_presented_proposal(handler, state, message, *, requested_review=None):
    from phase_agent.runtime.plan_queries import pending_plan_reply
    from phase_agent.runtime.chat_approval_rules import is_sensitive_proposal
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    binding = _load_binding(handler) or {}
    if requested_review is not None and binding.get("kind") == "mc_regeneration":
        if requested_review.get("plan_id") != binding.get("plan_id") or requested_review.get(
            "decision"
        ) not in {"approve", "reject"}:
            return "所指 MC 方案与刚展示的方案不一致；未执行。"
        if _review_rows(handler, state):
            return "同时存在其他待审批方案，请指定审批对象；未执行。"
        from phase_agent.graphs.dialogue.maintenance.mc import handle

        return (
            handle(
                handler,
                "同意" if requested_review["decision"] == "approve" else "拒绝",
                state,
                [],
                handler.conversation_id,
            )
            or "当前 MC 方案已不再待审批；未执行。"
        )
    rows = _review_rows(handler, state)
    if len(rows) != 1:
        return "当前没有唯一待审批方案，请在审批页选择对象；未执行任务。"
    row = rows[0]
    plan_id = row["plan_id"]
    if requested_review is not None:
        if requested_review.get("plan_id") != plan_id or requested_review.get("decision") not in {
            "approve",
            "reject",
        }:
            return "所指方案与当前待审批方案不一致；未执行，请核对具体方案。"
        decision = requested_review["decision"]
    else:
        decision = "approve" if message.strip().lower() in {"approve", "同意"} else "reject"
    binding = _load_binding(handler)
    if binding is None or binding.get("plan_id") != plan_id:
        bind_presented_proposal(handler, state)
        if decision == "approve":
            return pending_plan_reply(state) + "\n请核对以上方案后再批准或拒绝。"
        binding = handler.presented_plan_binding
    sensitive = (row.get("review_card") or {}).get("sensitive") or is_sensitive_proposal(
        row.get("agent_proposal") or {}
    )
    if decision == "approve" and sensitive:
        if row.get("review_kind") == "model_activation":
            # The current conversation has already displayed this exact switch.
            # Reuse its approval owner with the original version/hash guards.
            decision = "confirm_sensitive"
        else:
            import os

            return (
                "请在本机审批页核对并批准具体影响："
                + f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
            )
    try:
        outcome = handler.review_pending(
            plan_id,
            decision,
            expected_state_version=binding["state_version"],
            expected_proposal_hash=binding["proposal_hash"],
            comment=message,
        )
    except ValueError:
        _save_binding(handler, None)
        return "方案或项目状态已变化，请重新查看方案后审核；未执行任务。"
    _save_binding(handler, None)
    result = outcome.get("result") or {}
    if result.get("status") == "awaiting_approval":
        from phase_agent.tools.step_runner.file_protocol import read_json

        bind_presented_proposal(handler, read_json(handler.state_path, {}) or {})
    if row.get("review_kind") != "scientific_action":
        return result.get("reason") or outcome.get("reason") or str(outcome.get("status"))
    return format_workflow_reply(result, handler.state_path)


def fresh_turn_context(handler, state, messages, conversation_id):
    from phase_agent.runtime.conversation_context import conversation_context
    from phase_agent.persistence.memory.conversation import recent_turns

    from phase_agent.runtime.project_context import handler_project_context

    context = conversation_context(state, messages)
    context["current_project"] = handler_project_context(handler, state)
    from phase_agent.text_safety import clean

    context["project_settings"] = clean(
        context["current_project"]["configuration"]["effective_settings"]
    )
    context["recent_turns"] = recent_turns(handler.state_path, conversation_id)
    from phase_agent.graphs.dialogue.evidence import progress_signature

    context["facts_signature"] = progress_signature(state)
    previous_signature = (
        context["recent_turns"][-1].get("facts_signature") if context["recent_turns"] else None
    )
    context["facts_changed_since_previous_reply"] = (
        context["facts_signature"] != previous_signature if previous_signature else None
    )
    context["previous_reply"] = (
        context["recent_turns"][-1]["assistant"]
        if context["recent_turns"]
        else next(
            (
                str(row.get("content", ""))[:1600]
                for row in reversed(list(messages))
                if isinstance(row, dict) and row.get("role") == "assistant"
            ),
            "",
        )
    )
    context["pending_plan_facts"] = [
        {
            "plan_id": str(plan_id)[:180],
            "tool": (row.get("agent_proposal") or {}).get("recommended_action"),
            "status": row.get("status"),
            "task_status": (state.get("effective_decisions") or {})
            .get(((row.get("agent_proposal") or {}).get("raw_action") or {}).get("task_key"), {})
            .get("status"),
        }
        for plan_id, row in list((state.get("pending_execution_policies") or {}).items())[:4]
    ]
    context["presented_plan_binding"] = deepcopy(_load_binding(handler))
    context["authority"] = "当前项目记录优先；对话记忆仅辅助理解，不是批准或完成证据。"
    return context

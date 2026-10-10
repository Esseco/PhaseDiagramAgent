"""Conversation facts and version-bound approval; no intent classifier."""

from pathlib import Path
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
        return (read_json(path, {}) or {}).get("binding")
    return getattr(handler, "presented_plan_binding", None)


def bind_presented_proposal(handler, state):
    pending = state.get("pending_execution_policies") or {}
    _save_binding(handler, None)
    if len(pending) == 1:
        plan_id, record = next(iter(pending.items()))
        binding = {
            "plan_id": plan_id,
            "state_version": build_status_summary(
                state, config_version=state.get("confirmed_config_version")
            )["summary_id"],
            "proposal_hash": proposal_hash(record.get("agent_proposal") or {}),
        }
        _save_binding(handler, binding)


def review_presented_proposal(handler, state, message):
    from phase_agent.runtime.plan_queries import pending_plan_reply
    from phase_agent.runtime.chat_approval_rules import is_sensitive_proposal
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    pending = state.get("pending_execution_policies") or {}
    if len(pending) != 1:
        return "当前没有唯一待审批方案，请查看项目状态；未执行任务。"
    plan_id, record = next(iter(pending.items()))
    decision = "approve" if message.strip().lower() in {"approve", "同意"} else "reject"
    if decision == "approve" and is_sensitive_proposal(record.get("agent_proposal") or {}):
        import os

        return (
            "请在本机审批页核对并批准具体影响："
            + f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
        )
    binding = _load_binding(handler)
    if binding is None or binding.get("plan_id") != plan_id:
        bind_presented_proposal(handler, state)
        if decision == "approve":
            return pending_plan_reply(state) + "\n请核对以上方案后再批准或拒绝。"
        binding = handler.presented_plan_binding
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
    return format_workflow_reply(result, handler.state_path)


def fresh_turn_context(handler, state, messages, conversation_id):
    from phase_agent.runtime.conversation_context import conversation_context
    from phase_agent.graphs.dialogue.memory import recent_turns

    context = conversation_context(state, messages)
    config = (
        state.get("confirmed_config")
        or (handler.workflow_kwargs.get("config_session") or {})
        .get("confirmed_snapshot", {})
        .get("config")
        or {}
    )
    context["project_settings"] = {
        key: config.get(key)
        for key in ("system", "python_environments", "run", "calculation", "budgets", "convergence")
    }
    from phase_agent.runtime.turn_process import clean

    context["project_settings"] = clean(context["project_settings"])
    context["recent_turns"] = recent_turns(handler.state_path, conversation_id)
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
    context["authority"] = "当前项目记录优先；对话记忆仅辅助理解，不是批准或完成证据。"
    return context

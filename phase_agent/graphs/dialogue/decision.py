"""One semantic decision; read-only outcomes never enter scientific lifecycle."""

from copy import deepcopy
import hashlib
import json
from phase_agent.decisions.agent.dialogue_contract import is_dialogue, dialogue_result
from phase_agent.decisions.agent.propose_tool_action import propose_agent_tool_action
from phase_agent.analysis.state.build_state_snapshot import build_state_snapshot
from phase_agent.runtime.turn_process import timed_call, process_event


def _facts_signature(snapshot, manager, config):
    # Compare the same bounded scientific view on both sides of recovery.
    # Snapshot indexes and conversational memory do not change scientific evidence.
    facts = {
        key: snapshot.get(key)
        for key in ("current_convex_hull", "search_coverage", "uncertainty", "available_branches")
    }
    model = snapshot.get("mlip_status") or {}
    facts["model"] = {
        **model,
        "active_version": model.get("active_version")
        or (config.get("calculation") or {}).get("mlip_version"),
    }
    memory = snapshot.get("short_term_memory") or {}
    facts["tasks"] = memory.get("task_status") or {}
    facts["failed_tasks"] = memory.get("failed_tasks") or []
    facts["screening"] = memory.get("pending_branch_screening")
    facts["post_dft"] = (snapshot.get("decision_context") or {}).get("post_dft_assessment")
    facts["branches"] = (getattr(manager, "data", {}) or {}).get("branches") or {}
    facts["configuration"] = config
    return hashlib.sha256(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()


def decide_from_saved_facts(handler, facts, message, run_scientific, *, defer_scientific=None):
    client = handler.workflow_kwargs.get("agent_client")
    if not callable(client) or not handler.workflow_kwargs.get("config_session"):
        return run_scientific()
    session = handler.workflow_kwargs.get("config_session") or {}
    config = (
        facts.get("confirmed_config")
        or (session.get("confirmed_snapshot") or {}).get("config")
        or session.get("config")
        or {}
    )
    manager = handler.workflow_kwargs.get("manager")
    summary = build_state_snapshot(
        facts,
        snapshot_index=int(facts.get("state_snapshot_index", 0)),
        config_version=facts.get("confirmed_config_version"),
    )
    summary["decision_context"] = {
        **(summary.get("decision_context") or {}),
        "unified_dialogue": True,
        "confirmed_configuration": deepcopy(config),
        "saved_facts_notice": "这些是本轮读取的持久事实；外部任务刚产生但尚未回收的结果未知。科学动作进入原流程回收、复核和审批。",
    }
    allowed = [
        name
        for name in (config.get("agent") or {}).get("allowed_tools") or []
        if name != "run_calculation_stage"
    ]

    def request(payload):
        return timed_call(
            "dialogue_model",
            client,
            {
                **payload,
                "user_instruction": message,
                "conversation_facts": getattr(handler, "turn_context", {}),
                "interaction_policy": {
                    "reply_style": "Default to concise Chinese: answer the user's question first in 1-3 short sentences; include only a necessary next step. For progress updates use: result (only if actually available), current stage, recommended next step and approval requirement. Omit unavailable results; do not claim proposed work is completed. For an ordinary question answer directly without this template. Use familiar task names and explain necessary scientific terms briefly. Distinguish planned candidates, selected branches, generated structures and completed calculations. Show only this action's scope and budget by default; leave hypothetical future costs to detailed views. Expand when asked. Do not repeat status, boilerplate advice or raw validation traces in ordinary answers.",
                    "semantics": "Resolve short follow-ups using previous_reply, recent_turns and current project facts. A question about a failure asks for an explanation, not a retry of the failed action. Generic continuation is not approval. If no stored proposal exists, an affirmative reply to your question about preparing the next proposal means prepare and display that proposal, never execute it. After useful generation results, proactively propose the Relax input/task preparation action for approval rather than ask permission to propose it. When the user reports returned results or completed recovery, enter the scientific workflow to recover and analyze latest files, build the MLIP hull from valid Relax evidence and propose the next feasible action for approval; do not merely tell the user to ask again. An unverified completion report is not proof of task completion. Explicit configuration changes go to configure; science remains a proposal requiring approval. Ask only for genuinely missing/ambiguous information. When the user asks to continue and prerequisites are satisfied, propose the next action directly; do not ask permission merely to prepare a proposal. Execution still requires approval of the concrete stored proposal. Describe past corrected failures as historical, not as unresolved current parameter defects.",
                    "evidence": "previous_reply is historical conversation, not authoritative execution evidence. Explain the known cause and actual effect; distinguish an occupied task key from a submitted/completed task. If evidence is unavailable, say so rather than inventing a cause or re-running work.",
                },
            },
            category="model",
            mode=payload.get("mode"),
        )

    action = propose_agent_tool_action(
        summary, agent_client=request, allowed_tools=allowed, config=config
    )
    facts = _record_usage(handler, facts, action.get("_llm_usage"))
    action.pop("_llm_usage", None)
    if str(action.get("fallback_reason") or "").startswith("llm_failed:"):
        return {
            "status": "not_configured",
            "state": facts,
            "reason": "模型未返回有效对话或方案，本轮未执行。原因：" + action["fallback_reason"],
            "submitted": False,
            "steps_executed": 0,
        }
    if is_dialogue(action):
        return dialogue_result(action, facts)
    if len(facts.get("pending_execution_policies") or {}) > 1:
        return {
            "status": "answered",
            "answer": "有多个待审批方案，请先指定要处理的方案。未执行任务。",
            "state": facts,
            "submitted": False,
            "steps_executed": 0,
        }
    # An action is still only a proposal. The original workflow rechecks its
    # prerequisites and creates a bound approval; it never treats this as approval.
    signature = _facts_signature(summary, manager, config)
    used = False

    def reuse_or_refresh(payload):
        nonlocal used
        observed = payload.get("state") or {}
        changed = _facts_signature(observed, manager, config) != signature
        if not used and not changed:
            used = True
            process_event(
                "复用本轮模型建议", {"重新请求模型": False, "说明": "仍须原流程校验及人工审批"}
            )
            reuse_or_refresh.last_call_reused = True
            return deepcopy(action)
        process_event("重新分析最新事实", {"原因": "回收事实变化或原流程要求修正"})
        reuse_or_refresh.last_call_reused = False
        response = client(payload)
        used = True
        return response

    reuse_or_refresh.decision_backend = handler.decision_backend
    if defer_scientific is not None:
        return defer_scientific(reuse_or_refresh)
    handler.workflow_kwargs["agent_client"] = reuse_or_refresh
    try:
        return run_scientific()
    finally:
        handler.workflow_kwargs["agent_client"] = client


def _record_usage(handler, facts, usage):
    if not usage:
        return facts
    from phase_agent.tools.budget.record_budget_usage import record_budget_usage
    from phase_agent.tools.step_runner.file_protocol import write_json

    updated = record_budget_usage(
        facts, {"llm_usage": usage, "iteration": facts.get("iteration", 0)}
    )
    write_json(handler.state_path, updated)
    return updated

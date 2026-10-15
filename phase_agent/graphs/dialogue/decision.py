"""Bounded semantic conversation; read-only answers never enter scientific execution."""

from copy import deepcopy
import hashlib
import json
from phase_agent.decisions.agent.dialogue_contract import is_dialogue, dialogue_result
from phase_agent.analysis.state.build_state_snapshot import build_state_snapshot
from phase_agent.runtime.turn_process import timed_call, process_event


def _facts_signature(snapshot, manager, config):
    # Compare the same bounded scientific view on both sides of recovery.
    # Snapshot indexes and conversational memory do not change scientific evidence.
    facts = {
        key: snapshot.get(key)
        for key in (
            "current_convex_hull",
            "search_coverage",
            "uncertainty",
            "available_branches",
        )
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


def decide_from_saved_facts(
    handler, facts, message, run_scientific, *, defer_scientific=None, evidence=None
):
    client = handler.workflow_kwargs.get("agent_client")
    if not callable(client) or not handler.workflow_kwargs.get("config_session"):
        return run_scientific()
    from phase_agent.runtime.project_context import current_config_session, handler_project_context
    from phase_agent.analysis.state.project_authority import confirmed_configuration, AUTHORITY_RULE

    session = current_config_session(handler)
    config = confirmed_configuration(facts, session)
    manager = handler.workflow_kwargs.get("manager")
    from phase_agent.graphs.dialogue.evidence import conversation_facts

    summary = conversation_facts(facts, handler_project_context(handler, facts))
    from phase_agent.decisions.agent.summary_request import project_summary
    summary["current_project"] = project_summary(summary["current_project"])
    refresh = summary.get("refresh_result_facts") or {}
    summary["refresh_result_facts"] = {k: deepcopy(v) for k,v in refresh.items() if not isinstance(v, (dict,list))}
    summary["refresh_result_facts"]["details_available"] = bool(refresh)
    summary["decision_context"] = {
        **(evidence or {}),
        "unified_dialogue": True,
        "conversation_controls": True,
        "current_settings_authority": AUTHORITY_RULE,
        "evidence_reader": {
            "max_reads": 4,
            "topics": [
                "mc",
                "dft",
                "tasks",
                "convergence",
                "budget",
                "history",
                "model",
                "configuration",
                "plans",
            ],
            "instruction": "按需用inspect_evidence读取具体证据，不写数据或执行。已读内容保留在本轮上下文；不要重复读取同一主题。",
        },
        "saved_facts_notice": "本轮已核对可发现的回传文件并更新结果事实。回传通知应进入原科学流程完成分析和下一步方案；缺失结果具体说明，不反复要求用户先同意读取或分析。",
    }
    allowed = [
        name
        for name in (config.get("agent") or {}).get("allowed_tools") or []
        if name != "run_calculation_stage"
    ]

    def request(payload):
        turn = getattr(handler, "turn_context", {})
        conversation = {
            key: deepcopy(turn[key])
            for key in (
                "recent_turns",
                "pending_plan_facts",
                "presented_plan_binding",
                "authority",
                "facts_changed_since_previous_reply",
            )
            if key in turn
        }
        settings = (summary["current_project"].get("configuration") or {}).get(
            "effective_settings"
        ) or {}
        conversation["project_settings"] = {
            "calculation": deepcopy(settings.get("calculation") or {})
        }
        return timed_call(
            "dialogue_model",
            client,
            {
                **payload,
                "user_instruction": message,
                "conversation_facts": conversation,
                "semantic_dialogue": True,
                "scientific_evidence_loaded": bool(evidence),
                "interaction_policy": {
                    "reply_style": "Default to concise Chinese: answer the user's question first in 1-3 short sentences; include only a necessary next step. For progress updates use: result (only if actually available), current stage, recommended next step and approval requirement. Omit unavailable results; do not claim proposed work is completed. For an ordinary question answer directly without this template. Use familiar task names and explain necessary scientific terms briefly. Distinguish planned candidates, selected branches, generated structures and completed calculations. Show only this action's scope and budget by default; leave hypothetical future costs to detailed views. Expand when asked. Do not repeat status, boilerplate advice or raw validation traces in ordinary answers.",
                    "semantics": "Resolve short follow-ups using previous_reply, recent_turns and current project facts. A question about a failure asks for an explanation, not a retry of the failed action. Generic continuation is not approval. If no stored proposal exists, an affirmative reply to your question about preparing the next proposal means prepare and display that proposal, never execute it. After useful generation results, proactively propose the Relax input/task preparation action for approval rather than ask permission to propose it. When the user reports returned results or completed recovery, enter the scientific workflow to recover and analyze latest files, build the MLIP hull from valid Relax evidence and propose the next feasible action for approval; do not merely tell the user to ask again. An unverified completion report is not proof of task completion. Explicit configuration changes go to configure; science remains a proposal requiring approval. Ask only for genuinely missing/ambiguous information. When the user asks to continue and prerequisites are satisfied, propose the next action directly; do not ask permission merely to prepare a proposal. Execution still requires approval of the concrete stored proposal. Describe past corrected failures as historical, not as unresolved current parameter defects.",
                    "evidence": "previous_reply is historical conversation, not authoritative execution evidence. Explain the known cause and actual effect; distinguish an occupied task key from a submitted/completed task. If evidence is unavailable, say so rather than inventing a cause or re-running work.",
                    "continuity": "A return/recovery notification or request to advance uses kind=advance when scientific observation is needed; it prepares one concrete next approval in this same turn. Do not ask permission to read, analyze or propose. When the user explicitly approves/rejects the previously displayed stored plan, use kind=review with its actual plan_id and decision; never turn approval into a fresh proposal. A question about the plan remains answer. Only a concrete missing input or consequential unresolved choice warrants clarification. Scientific tools are proposals, never permission to execute. The previous reply and presented_plan_binding identify the displayed plan; project facts remain authoritative.",
                },
            },
            category="model",
            mode=payload.get("mode"),
        )

    from phase_agent.decisions.agent.decision_backend import request_decision_action
    from phase_agent.decisions.agent.dialogue_contract import dialogue_output_contract

    payload = {
        "mode": "autonomous_search",
        "state": {k: v for k, v in summary.items() if k != "decision_context"},
        "decision_context": summary["decision_context"],
        "unified_dialogue": True,
        "allowed_tools": allowed,
        "output_contracts": {"dialogue_outcome": dialogue_output_contract(allow_controls=True)},
        "instruction": "Understand the current conversation. Return answer, inspect_evidence, inspect_configuration, configure, advance or review. Answer the question first; read only needed evidence for detailed questions. Advance is for a requested scientific next step, never an ordinary question. Use recent_turns to resolve follow-ups. A known blocker requires cause, concrete remedy and what happens after repair; do not repeat approval/continue questions that cannot fix it. Reading and planning need no separate permission. Historical tool outcomes remain compatible but are rechecked by the original scientific workflow. Conversation memory confers no authority.",
    }
    try:
        action = request_decision_action(request, payload)
    except Exception as error:
        facts = _record_usage(handler, facts, getattr(error, "llm_usage", None))
        return {
            "status": "not_configured",
            "state": facts,
            "reason": "未取得有效答复，本轮未执行任务：" + str(error),
            "submitted": False,
            "steps_executed": 0,
        }
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
    if action.get("kind") == "advance":
        return (defer_scientific or run_scientific)()
    if action.get("kind") == "review":
        return {"status": "proposal_review_requested", "review": deepcopy(action)}
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
    if action.get("tool") not in {"check_convergence", "pause_search"} and not evidence:
        return (defer_scientific or run_scientific)()
    if evidence and action.get("tool") not in {"check_convergence", "pause_search"}:
        from phase_agent.decisions.agent.summary_request import required_topics

        missing = required_topics(action) - set(evidence.get("_loaded_topics") or [])
        if missing:
            # Reading one topic is not proof that every action prerequisite was read.
            return dialogue_result({"kind": "inspect_evidence", "topics": sorted(missing)}, facts)
    scientific_summary = build_state_snapshot(
        facts,
        snapshot_index=int(facts.get("state_snapshot_index", 0)),
        config_version=facts.get("confirmed_config_version"),
    )
    signature = _facts_signature(scientific_summary, manager, config)
    used = False

    def reuse_or_refresh(payload):
        nonlocal used
        observed = payload.get("state") or {}
        changed = _facts_signature(observed, manager, config) != signature
        if not used and not changed:
            used = True
            process_event(
                "复用本轮模型建议",
                {"重新请求模型": False, "说明": "仍须原流程校验及人工审批"},
            )
            reuse_or_refresh.last_call_reused = True
            return deepcopy(action)
        process_event("重新分析最新事实", {"原因": "回收事实变化或原流程要求修正"})
        reuse_or_refresh.last_call_reused = False
        from phase_agent.decisions.agent.summary_request import request_summary_first
        response = request_summary_first(client, payload)
        used = True
        return response

    reuse_or_refresh.decision_backend = handler.decision_backend
    reuse_or_refresh._validates_original_facts = True
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

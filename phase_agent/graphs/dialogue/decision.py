"""Bounded semantic conversation; read-only answers never enter scientific execution."""

from copy import deepcopy
import hashlib
import json
from phase_agent.decisions.agent.dialogue_contract import is_dialogue, dialogue_result
from phase_agent.analysis.state.build_state_snapshot import build_state_snapshot
from phase_agent.runtime.turn_process import timed_call, process_event


def recovery_notice(error):
    detail = str(error)[:500] or "未知错误"
    return "结果回收失败：" + detail + "。本轮仅可答疑或查看已保存参数；未推进科研任务。"


def _recovery_blocked_result(facts, error, detail=None):
    answer = recovery_notice(error)
    if detail:
        answer += " " + str(detail)[:300]
    return {
        "status": "answered",
        "answer": answer,
        "state": facts,
        "submitted": False,
        "steps_executed": 0,
    }


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
    handler,
    facts,
    message,
    run_scientific,
    *,
    defer_scientific=None,
    evidence=None,
    recovery_error=None,
):
    client = handler.workflow_kwargs.get("agent_client")
    recovery_limited = bool(recovery_error)
    if recovery_limited and (
        not callable(client) or not handler.workflow_kwargs.get("config_session")
    ):
        return _recovery_blocked_result(facts, recovery_error)
    if not callable(client) or not handler.workflow_kwargs.get("config_session"):
        return run_scientific()
    from phase_agent.runtime.project_context import current_config_session, handler_project_context
    from phase_agent.analysis.state.project_authority import confirmed_configuration, AUTHORITY_RULE

    session = current_config_session(handler)
    config = confirmed_configuration(facts, session)
    manager = handler.workflow_kwargs.get("manager")
    from phase_agent.graphs.dialogue.evidence import conversation_facts

    project_card = handler_project_context(handler, facts)
    summary = conversation_facts(facts, project_card)
    from phase_agent.decisions.agent.summary_request import project_summary

    summary["current_project"] = project_summary(summary["current_project"])
    refresh = summary.get("refresh_result_facts") or {}
    summary["refresh_result_facts"] = {
        k: deepcopy(v) for k, v in refresh.items() if not isinstance(v, (dict, list))
    }
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
        "saved_facts_notice": (
            recovery_notice(recovery_error)
            if recovery_limited
            else "本轮已核对可发现的回传文件并更新结果事实。回传通知应进入原科学流程完成分析和下一步方案；缺失结果具体说明，不反复要求用户先同意读取或分析。"
        ),
    }
    if facts.get("_recovery_just_approved") or facts.get("post_recovery_next_step_pending"):
        summary["decision_context"]["post_recovery_choice"] = (
            "结果回收刚获得明确批准并已完成。现在必须基于回收证据判断下一步优先是 "
            "select_dft_candidates（DFT）还是 generate_branches（生成新 branch），"
            "只提出其中一项具体待审批方案；本轮不得执行、提交或把建议当作批准。"
        )
    allowed = (
        []
        if recovery_limited
        else [
            name
            for name in (config.get("agent") or {}).get("allowed_tools") or []
            if name != "run_calculation_stage"
        ]
    )
    post_recovery_only = bool(
        facts.get("post_recovery_next_step_pending")
        or facts.get("_recovery_just_approved")
    )
    if post_recovery_only and not recovery_limited:
        # This is a phase transition boundary, not a wording preference.
        # Relax/MC input preparation is a later action and must not be
        # available until the user approves a DFT/branch choice first.
        allowed = [
            name for name in allowed if name in {"select_dft_candidates", "generate_branches"}
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
        # Send the most recent answer once, including transport history after a restart.
        conversation["previous_reply"] = turn.get("previous_reply", "")
        if conversation.get("recent_turns"):
            conversation["recent_turns"][-1].pop("assistant", None)
        settings = (project_card.get("configuration") or {}).get("effective_settings") or {}
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
                    "semantics": "Resolve short follow-ups using previous_reply, recent_turns and current project facts. A question about a failure asks for an explanation, not a retry of the failed action. Generic continuation is not approval. If a verified partial_recovery fact is present, ‘继续’ means proceed with the successful subset while explicitly marking incomplete coverage; ‘补全’ means propose restart_failed_task only for the listed retryable_task_ids. Never rerun or rebuild the full batch, and never count a failed task as completed. If no stored proposal exists, an affirmative reply to your question about preparing the next proposal means prepare and display that proposal, never execute it. After useful generation results, proactively propose the Relax input/task preparation action for approval rather than ask permission to propose it. When the user reports returned results or completed recovery, enter the scientific workflow to recover and analyze latest files, build the MLIP hull from valid Relax evidence and propose the next feasible action for approval; do not merely tell the user to ask again. An unverified completion report is not proof of task completion. Explicit configuration changes go to configure; science remains a proposal requiring approval. Ask only for genuinely missing/ambiguous information. When the user asks to continue and prerequisites are satisfied, propose the next action directly; do not ask permission merely to prepare a proposal. Execution still requires approval of the concrete stored proposal. Describe past corrected failures as historical, not as unresolved current parameter defects.",
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
        "output_contracts": {
            "dialogue_outcome": dialogue_output_contract(allow_controls=not recovery_limited)
        },
        "instruction": (
            "Recovery failed. Return only kind=answer or kind=inspect_configuration. "
            "Answer from the supplied saved facts or show current settings. Do not "
            "advance, review, configure, cancel waits, inspect evidence, draw diagrams, "
            "change policy, call tools or prepare science. Make the recovery error visible."
            if recovery_limited
            else "Understand the current conversation. Return answer, inspect_evidence, inspect_configuration, configure, advance, review or cancel_waits, or phase_diagram (plot/locate/export_csv existing current diagram files), phase_quality_policy (audited H/D cutoff changes only; manual fixed values take precedence; no deletion or science in this turn), or phase_quality_review (explicit human retain/remove/restore of bound review IDs). Answer the question first; read only needed evidence for detailed questions. Advance is for a requested scientific next step, never an ordinary question. Use recent_turns to resolve follow-ups. A known blocker requires cause, concrete remedy and what happens after repair; do not repeat approval/continue questions that cannot fix it. Reading and planning need no separate permission. Historical tool outcomes remain compatible but are rechecked by the original scientific workflow. Conversation memory confers no authority."
            + (
                "刚完成结果回收：必须在 DFT 与 generate_branches 中择一提出单项方案，"
                "先展示方案并等待人工批准，禁止直接执行。"
                if facts.get("_recovery_just_approved")
                or facts.get("post_recovery_next_step_pending")
                else ""
            )
        ),
    }

    def retry_invalid_dialogue(error):
        """Give the model one bounded correction for an invalid dialogue envelope."""
        message = str(error)
        if not any(
            marker in message
            for marker in (
                "LangGraph proposal validation failed",
                "Invalid LLM action",
                "not allowed in this request",
            )
        ):
            return None
        feedback = {
            "code": "dialogue_proposal_validation_failed",
            "errors": list(getattr(error, "validation_errors", ()) or ())
            or [message[:1000]],
            "allowed_tools": list(allowed),
            "instruction": (
                "上一版输出不是合法对话结果。请严格返回当前对话协议允许的 kind；"
                "如果用户只是要求继续，请返回 kind=advance；不要返回工具动作，"
                "也不要把不在 allowed_tools 中的 tool 放进结果。"
            ),
        }
        corrected = deepcopy(payload)
        corrected_context = deepcopy(corrected.get("decision_context") or {})
        corrected_context["proposal_validation_feedback"] = feedback
        corrected["decision_context"] = corrected_context
        corrected["instruction"] = (
            corrected.get("instruction", "") + " " + feedback["instruction"]
        )
        return corrected

    try:
        action = request_decision_action(request, payload)
    except Exception as error:
        facts = _record_usage(handler, facts, getattr(error, "llm_usage", None))
        corrected_payload = retry_invalid_dialogue(error)
        if corrected_payload is not None:
            try:
                action = request_decision_action(request, corrected_payload)
            except Exception as retry_error:
                error = retry_error
            else:
                error = None
        if error is None:
            pass
        elif recovery_limited:
            return _recovery_blocked_result(facts, recovery_error, "模型答复无效：" + str(error))
        elif corrected_payload is not None:
            return {
                "status": "answered",
                "answer": (
                    "本轮模型没有返回符合当前项目允许范围的对话结果，未执行任何任务；"
                    "项目状态和任务记录已保留。请再次发送‘继续’重新生成方案。"
                ),
                "state": facts,
                "submitted": False,
                "steps_executed": 0,
            }
        else:
            return {
                "status": "not_configured",
                "state": facts,
                "reason": "未取得有效答复，本轮未执行任务：" + str(error),
                "submitted": False,
                "steps_executed": 0,
            }
    facts = _record_usage(handler, facts, action.get("_llm_usage"))
    action.pop("_llm_usage", None)
    if post_recovery_only and action.get("kind") == "advance":
        # The dialogue envelope is only a request to enter the scientific
        # planner; the planner still receives the hard DFT/branch allow-list.
        return (defer_scientific or run_scientific)()
    if post_recovery_only and action.get("tool") not in {
        "select_dft_candidates",
        "generate_branches",
    }:
        return {
            "status": "answered",
            "answer": (
                "回收已完成；下一步只能先提出 DFT 或生成 branch 方案。"
                "本轮未生成 Relax/MC 输入、未提交计算，请再次发送‘继续’重新提出单项方案。"
            ),
            "state": facts,
            "submitted": False,
            "steps_executed": 0,
        }
    if str(action.get("fallback_reason") or "").startswith("llm_failed:"):
        if recovery_limited:
            return _recovery_blocked_result(facts, recovery_error, action["fallback_reason"])
        return {
            "status": "not_configured",
            "state": facts,
            "reason": "模型未返回有效对话或方案，本轮未执行。原因：" + action["fallback_reason"],
            "submitted": False,
            "steps_executed": 0,
        }
    if recovery_limited and action.get("kind") not in {"answer", "inspect_configuration"}:
        return _recovery_blocked_result(facts, recovery_error)
    if action.get("kind") == "phase_diagram":
        from phase_agent.runtime.phase_diagram_chat import phase_diagram_reply

        before = deepcopy(facts.get("phase_diagrams") or {})
        answer = phase_diagram_reply(
            facts,
            operation=action["operation"],
            method=action.get("method", "mlip"),
            formats=action.get("formats", ["png"]),
            directory=handler.workflow_kwargs.get("phase_diagram_directory"),
        )
        if facts.get("phase_diagrams", {}) != before:
            from phase_agent.tools.step_runner.file_protocol import write_json

            write_json(handler.state_path, facts)
        return {
            "status": "answered",
            "answer": answer,
            "state": facts,
            "submitted": False,
            "steps_executed": 0,
        }
    if action.get("kind") == "phase_quality_policy":
        from pathlib import Path
        from phase_agent.analysis.phase.phase_quality_policy import commit_quality_policy

        try:
            changed = commit_quality_policy(
                handler.state_path,
                facts,
                manager,
                ledger_path=config.get("ledger_path")
                or handler.workflow_kwargs.get("ledger_path")
                or Path(handler.state_path).parent / "ledgers" / "phase_data.json",
                cutoffs=action["cutoffs"],
                source=action["source"],
                reason=action["reason"],
                expected_revision=(facts.get("phase_quality_policy") or {}).get("revision", 0),
                release_manual=action.get("release_manual", []),
            )
        except (OSError, ValueError, KeyError) as error:
            return {
                "status": "answered",
                "state": facts,
                "answer": "阈值未调整：" + str(error),
                "submitted": False,
                "steps_executed": 0,
            }
        policy = changed["policy"]
        return {
            "status": "answered",
            "state": changed["state"],
            "submitted": False,
            "steps_executed": 0,
            "answer": f"已记录相图核查阈值：H审查>{policy['hull_shift_mev_per_O2']:g}、D离群>{policy['outlier_gap_mev_per_O2']:g}、H直接移除>{policy['direct_removal_mev_per_O2']:g} meV/O₂。原因：{action['reason']}。本轮仅调整参数，未删除结构或提交计算。",
        }
    if action.get("kind") == "phase_quality_review":
        from pathlib import Path
        from phase_agent.analysis.phase.phase_quality_persistence import commit_human_quality_review

        try:
            reviewed = commit_human_quality_review(
                handler.state_path,
                facts,
                manager,
                review_ids=action["review_ids"],
                decision=action["decision"],
                user_message=message,
                ledger_path=config.get("ledger_path")
                or handler.workflow_kwargs.get("ledger_path")
                or Path(handler.state_path).parent / "ledgers" / "phase_data.json",
                directory=handler.workflow_kwargs.get("phase_diagram_directory"),
            )
            facts = reviewed["state"]
        except (OSError, ValueError, KeyError) as error:
            return {
                "status": "answered",
                "state": facts,
                "answer": "此次审核未应用：" + str(error),
                "submitted": False,
                "steps_executed": 0,
            }
        process_event(
            "相图人工核查已记录",
            {"decision": action["decision"], "review_ids": action["review_ids"]},
        )
        return (defer_scientific or run_scientific)()
    if action.get("kind") == "advance":
        return (defer_scientific or run_scientific)()
    if action.get("kind") == "cancel_waits":
        from phase_agent.tools.state.task_wait_cancellation import commit_task_wait_cancellation

        try:
            facts = commit_task_wait_cancellation(
                handler.state_path, facts, action["scope_id"], user_message=message
            )
        except ValueError as error:
            return {
                "status": "answered",
                "answer": "待处理MC的范围已变化，本次未取消；请按最新批次明确范围。原因："
                + str(error),
                "state": facts,
                "submitted": False,
                "steps_executed": 0,
            }
        process_event(
            "停止本地MC等待",
            {
                "scope_id": action["scope_id"],
                "remote_cancel_sent": False,
                "unknown_costs_settled": False,
            },
        )
        return (defer_scientific or run_scientific)()
    if action.get("kind") == "review":
        return {"status": "proposal_review_requested", "review": deepcopy(action)}
    if is_dialogue(action):
        result = dialogue_result(action, facts)
        if recovery_limited and result.get("answer"):
            result["answer"] = recovery_notice(recovery_error) + "\n" + result["answer"]
        return result
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

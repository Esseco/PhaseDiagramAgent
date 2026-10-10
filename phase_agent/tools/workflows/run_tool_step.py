"""Confirmed-config tool step with an explicit execution policy gate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from phase_agent.decisions.agent.propose_tool_action import (
    _apply_generation_defaults,
    propose_agent_tool_action,
)
from phase_agent.decisions.agent.revise_tool_proposal import revise_tool_proposal
from phase_agent.persistence.memory.decision_memory import (
    update_long_term_advice,
    update_long_term_memory,
)
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.policy.execution_policy import apply_execution_policy, build_agent_proposal
from phase_agent.tools.budget.reserve_action_budget import reserve_action_budget
from phase_agent.tools.budget.record_budget_usage import record_budget_usage
from phase_agent.tools.workflows.tool_outcomes import (
    _audit_record,
    _upsert_record,
    _store_completed,
    _policy_rejection,
    _apply_execution_result,
)
from phase_agent.tools.policy.validate_tool_action import validate_tool_action
from phase_agent.tools.state.state_manager import agent_state_summary, update_state_snapshot
from phase_agent.tools.workflows.compact_action_history import compact_action_history


def _model_failure_action(action):
    return action.get("decision_source") == "rule" and str(
        action.get("fallback_reason") or ""
    ).startswith("llm_failed:")


def run_tool_step(
    state: dict | None,
    session: dict,
    *,
    registry: dict,
    agent_client=None,
    context=None,
    execute=False,
    invocation_id=None,
    execution_mode: str | None = None,
    human_feedback: str | dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
    tool_graph=None,
) -> dict:
    """Run explicit proposal → approval → validation → tool → audit graph nodes."""
    from phase_agent.graphs.actions.graph import run_tool_action_graph

    stages = tool_step_stages(
        state,
        session,
        registry=registry,
        agent_client=agent_client,
        context=context,
        execute=execute,
        invocation_id=invocation_id,
        execution_mode=execution_mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
    )
    return run_tool_action_graph(tool_graph=tool_graph, registry=registry, **stages)


def tool_step_stages(
    state,
    session,
    *,
    registry,
    agent_client=None,
    context=None,
    execute=False,
    invocation_id=None,
    execution_mode=None,
    human_feedback=None,
    replay_record=None,
):
    """Expose existing callbacks without invoking them or changing their order."""
    return dict(
        initialize=lambda: _initialize_tool_step(
            state,
            session,
            registry=registry,
            agent_client=agent_client,
            context=context,
            execute=execute,
            invocation_id=invocation_id,
            execution_mode=execution_mode,
            human_feedback=human_feedback,
            replay_record=replay_record,
        ),
        refresh=_refresh_graph_gate,
        prepare=_prepare_tool_proposal,
        approve=_approval_graph_node,
        validate=_validation_graph_node,
        dispatch=_dispatch_graph_node,
        finish=_finish_graph_node,
    )


def _initialize_tool_step(
    state: dict | None,
    session: dict,
    *,
    registry: dict,
    agent_client=None,
    context=None,
    execute=False,
    invocation_id=None,
    execution_mode: str | None = None,
    human_feedback: str | dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
) -> dict:
    """Propose, gate, validate, and optionally execute one tool action.

    Interactive mode returns ``awaiting_approval`` on the first call. Call it
    again with the same ``invocation_id`` and ``human_feedback`` to continue.
    """
    current = deepcopy(
        state
        or {
            "status": "ready",
            "tasks": [],
            "decisions": [],
            "action_records": [],
            "effective_decisions": {},
            "invocations": {},
            "pending_execution_policies": {},
        }
    )
    current.setdefault("action_records", [])
    current.setdefault("pending_execution_policies", {})
    if invocation_id and invocation_id in current.get("invocations", {}):
        return {
            **deepcopy(current["invocations"][invocation_id]),
            "state": current,
            "idempotent_replay": True,
        }

    explicit_mode = execution_mode is not None
    mode = execution_mode or ("autonomous" if execute else "dry_run")
    config = (
        (session.get("confirmed_snapshot") or {}).get("config")
        if session.get("status") == "confirmed"
        else session.get("config", {})
    )
    if session.get("status") == "confirmed":
        snapshot = session["confirmed_snapshot"]
        snapshot_version = snapshot["config_version"]
        bound_version = current.get("confirmed_config_version")
        if bound_version is None or (
            bound_version != snapshot_version and _can_rebind_empty_run(current)
        ):
            # A config can be confirmed after an empty/debug run has already
            # created state.json.  Rebind only before any scientific work or
            # reservation exists; populated runs must never mix versions.
            current["confirmed_config_version"] = snapshot_version
            current["confirmed_config"] = deepcopy(snapshot["config"])
        elif bound_version != snapshot_version:
            return {
                "status": "configuration_version_mismatch",
                "execution_mode": mode,
                "reason": f"当前运行绑定 {bound_version}，新配置为 {snapshot_version}；已有任务或预算记录，不能直接混用。",
                "agent_proposal": None,
                "final_action": None,
                "action": None,
                "validation": None,
                "execution": None,
                "execution_result": None,
                "state": current,
                "idempotent_replay": False,
            }
    current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
    return {
        "_graph_prepared": True,
        "current": current,
        "mode": mode,
        "config": config,
        "explicit_mode": explicit_mode,
        "session": session,
        "registry": registry,
        "agent_client": agent_client,
        "context": context,
        "execute": execute,
        "invocation_id": invocation_id,
        "human_feedback": human_feedback,
        "replay_record": replay_record,
    }


def _refresh_graph_gate(frame):
    current = frame["current"]
    mode = frame["mode"]
    config = frame["config"]
    explicit_mode = frame["explicit_mode"]
    session = frame["session"]
    registry = frame["registry"]
    agent_client = frame["agent_client"]
    context = frame["context"]
    execute = frame["execute"]
    invocation_id = frame["invocation_id"]
    human_feedback = frame["human_feedback"]
    replay_record = frame["replay_record"]
    from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate

    refresh_result = run_model_refresh_gate(
        current,
        session,
        registry=registry,
        context=context or {},
        human_feedback=human_feedback,
        mode=mode,
        pending_key=invocation_id or "__single_interactive_action__",
    )
    if refresh_result is not None:
        return refresh_result
    return frame


def _prepare_tool_proposal(frame):
    if (frame.get("context") or {}).get("unified_dialogue"):
        from phase_agent.tools.workflows.react_proposal import prepare_react_proposal

        return prepare_react_proposal(frame)
    current = frame["current"]
    mode = frame["mode"]
    config = frame["config"]
    explicit_mode = frame["explicit_mode"]
    session = frame["session"]
    registry = frame["registry"]
    agent_client = frame["agent_client"]
    context = frame["context"]
    execute = frame["execute"]
    invocation_id = frame["invocation_id"]
    human_feedback = frame["human_feedback"]
    replay_record = frame["replay_record"]
    decision_state = agent_state_summary(current)
    decision_state["user_message"] = str((context or {}).get("user_message") or "")

    pending_key = invocation_id or "__single_interactive_action__"
    stored = current["pending_execution_policies"].get(pending_key)
    rejecting = isinstance(human_feedback, dict) and human_feedback.get("decision") == "reject"
    previous = ((stored or {}).get("agent_proposal") or {}).get("raw_action") or {}
    from phase_agent.tools.state.approved_direction import direction_action_errors

    if stored and not rejecting and direction_action_errors(previous, current):
        current.setdefault("superseded_direction_actions", []).append(
            {
                "record_id": pending_key,
                "record": deepcopy(stored),
                "reason": "Approved scientific direction changed or action was not bound.",
            }
        )
        current["pending_execution_policies"].pop(pending_key, None)
        stored = None
        human_feedback = None
    from phase_agent.tools.state.training_pending_validation import training_pending_validation

    if (
        stored
        and training_pending_validation(current)
        and (previous.get("tool") or previous.get("action_type")) == "update_mlip"
    ):
        current.setdefault("superseded_training_proposals", []).append(
            {
                "record_id": pending_key,
                "record": deepcopy(stored),
                "reason": "Training already recovered; duplicate proposal blocked.",
            }
        )
        current["pending_execution_policies"].pop(pending_key, None)
        stored = None
        human_feedback = None
    from phase_agent.analysis.state.round_budget_evidence import round_budget_evidence
    from phase_agent.decisions.agent.round_budget_review import round_budget_review_errors

    budget_context = {"round_budget_evidence": round_budget_evidence(current)}
    if (
        mode == "interactive"
        and stored
        and not rejecting
        and round_budget_review_errors(previous, budget_context)
    ):
        current["pending_execution_policies"].pop(pending_key, None)
        stored = None
        human_feedback = None
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment
    from phase_agent.decisions.agent.post_dft_review import valid_post_dft_review

    closed_dft = post_dft_assessment(current, (context or {}).get("effective_config") or config)
    if (
        mode == "interactive"
        and stored
        and not rejecting
        and closed_dft
        and previous.get("tool")
        in {
            "generate_branches",
            "update_mlip",
            "adjust_strategy",
            "select_dft_candidates",
            "check_convergence",
            "pause_search",
        }
        and (
            previous.get("_post_dft_review_version") != 5
            or not valid_post_dft_review(previous)
            or previous.get("_post_dft_scope_key") != closed_dft["scope_key"]
        )
    ):
        current["pending_execution_policies"].pop(pending_key, None)
        stored = None
        human_feedback = None  # Old approval does not approve a freshly assessed choice.
    if (
        mode == "interactive"
        and closed_dft
        and closed_dft["status"] != "evaluated"
        and not rejecting
    ):
        eligible_ids = {
            row.get("task_id")
            for row in current.get("dft_dataset_records") or []
            if row.get("checks_passed") is True
            and row.get("status") == "completed"
            and row.get("converged") is True
            and row.get("training_ready") is True
        }
        reasons = list(
            dict.fromkeys(
                str(row.get("reason"))
                for row in closed_dft["metrics"]["not_evaluated"]
                if row.get("reason") and row.get("task_id") in eligible_ids
            )
        )[:2]
        return {
            "status": "not_configured",
            "state": current,
            "reason": "DFT 回收已结束；本轮原模型误差评估未完成 "
            f"（已配对 {closed_dft['paired_structures']}/{closed_dft['eligible_structures']}）。"
            "请补齐超算同帧 MLIP 预测并回传 mlip_result.json 后继续；未重复派发 DFT、未训练。"
            + ("原因：" + "；".join(reasons) if reasons else ""),
        }
    if (
        mode == "interactive"
        and stored
        and not rejecting
        and closed_dft
        and not previous.get("_approved_direction_hash")
        and previous.get("tool")
        in {"select_dft_candidates", "allocate_mc_bohb", "prepare_local_batch_files"}
    ):
        current["pending_execution_policies"].pop(pending_key, None)
        stored = None
        human_feedback = None  # Never reuse approval of a superseded calculation plan.
    if mode == "interactive" and stored and _model_failure_action(previous) and not rejecting:
        if isinstance(human_feedback, dict) and human_feedback.get("decision") == "approve":
            return {
                "status": "rejected",
                "state": current,
                "reason": "该建议由模型通信失败产生，不是科学决策。请说“继续”重新获取方案；未执行任务。",
            }
        if str((context or {}).get("user_message") or "").strip().lower() in {
            "继续",
            "下一步",
            "continue",
            "next",
        }:
            current["pending_execution_policies"].pop(pending_key, None)
            stored = None
            human_feedback = None
    if mode == "interactive" and stored and not rejecting:
        from phase_agent.tools.workflows.preview_dft_inputs import stale_dft_preview

        if stale_dft_preview(stored.get("agent_proposal") or {}):
            decision = (human_feedback or {}).get("decision")
            if decision == "approve":
                return {
                    "status": "rejected",
                    "state": current,
                    "reason": "该 DFT 方案缺少新版候选清单与采点依据，请说“继续”刷新后再确认；未执行任务。",
                }
            message = str((context or {}).get("user_message") or "").strip().lower()
            if message in {"继续", "下一步", "continue", "next"}:
                current["pending_execution_policies"].pop(pending_key, None)
                stored = None  # Re-propose only; never carry an approval into a new plan.
                human_feedback = None
    if mode == "interactive" and stored and not rejecting and _is_pending_relax_preparation(stored):
        allowed = [
            name
            for name in (config.get("agent") or {}).get("allowed_tools") or []
            if callable((registry.get(name) or {}).get("handler"))
        ]
        from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action

        next_action = choose_debug_next_action(
            current,
            (context or {}).get("manager"),
            (context or {}).get("effective_config") or config,
            allowed_tools=allowed,
            user_message="继续",
        )
        has_mc_results = any(
            row.get("stage") == "deep_search" for row in current.get("tasks") or []
        )
        next_mode = ((next_action or {}).get("parameters") or {}).get("mode")
        is_mc_next_step = (next_action or {}).get("tool") == "allocate_mc_bohb" or (
            (next_action or {}).get("tool") == "prepare_local_batch_files"
            and next_mode == "mc_inputs"
        )
        if next_action and (is_mc_next_step or not has_mc_results):
            stored = deepcopy(stored)
            previous = (stored.get("agent_proposal") or {}).get("raw_action") or {}
            stored["revision"] = int(stored.get("revision", 0)) + 1
            stored.setdefault("feedback_history", []).append(
                {
                    "revision_status": (
                        "mc_round_transition_refreshed"
                        if has_mc_results
                        else "relax_results_recovered"
                    ),
                    "prior_action": previous.get("tool"),
                    "analysis": (
                        "已有 MC 结果与当前相图可用于下一段分配；以 MC 方案替换补充 Relax 方案。"
                        if has_mc_results
                        else "已回收的 Relax 结果覆盖该批结构；改为 MC 预算与输入文件准备。"
                    ),
                }
            )
            stored["agent_proposal"] = build_agent_proposal(
                next_action, decision_state, runtime_state=current
            )
            current["pending_execution_policies"][pending_key] = stored
            human_feedback = None  # Approval for the old Relax proposal cannot approve MC.
        elif has_mc_results:
            current["pending_execution_policies"].pop(pending_key, None)
            return _mc_continuation_block(
                current, mode, (context or {}).get("effective_config") or config
            )
    if mode == "interactive" and stored and not rejecting:
        previous = (stored.get("agent_proposal") or {}).get("raw_action") or {}
        stage = previous.get("stage") or (previous.get("parameters") or {}).get("stage")
        if previous.get("tool") == "run_calculation_stage" and stage == "deep_search":
            allowed = [
                name
                for name in (config.get("agent") or {}).get("allowed_tools") or []
                if callable((registry.get(name) or {}).get("handler"))
            ]
            corrected = _prepare_debug_relax_screen_action(
                previous, current, context, config, allowed, invocation_id=invocation_id
            )
            stored = deepcopy(stored)
            stored["revision"] = int(stored.get("revision", 0)) + 1
            stored.setdefault("feedback_history", []).append(
                {
                    "revision_status": "mc_batch_action_corrected",
                    "analysis": corrected["reason"],
                }
            )
            stored["agent_proposal"] = build_agent_proposal(
                corrected, decision_state, runtime_state=current
            )
            current["pending_execution_policies"][pending_key] = stored
            human_feedback = None  # The old single-task approval does not approve the new action.
    advice_changed = False
    if (
        mode == "interactive"
        and stored
        and isinstance(human_feedback, dict)
        and human_feedback.get("long_term_advice") is not None
    ):
        updated = update_long_term_advice(
            current,
            human_feedback["long_term_advice"],
            source=f"{pending_key}:revision-{stored.get('revision', 0)}",
        )
        advice_changed = updated.get("decision_memory") != current.get("decision_memory")
        current = updated
        if advice_changed:
            current = update_state_snapshot(
                current, config_version=current.get("confirmed_config_version")
            )
            decision_state = agent_state_summary(current)
    if (
        mode == "interactive"
        and stored
        and isinstance(human_feedback, dict)
        and human_feedback.get("long_term_memory") is not None
    ):
        updated = update_long_term_memory(
            current,
            human_feedback["long_term_memory"],
            source=f"{pending_key}:revision-{stored.get('revision', 0)}",
        )
        memory_changed = updated.get("decision_memory") != current.get("decision_memory")
        advice_changed = advice_changed or memory_changed
        current = updated
        if memory_changed:
            current = update_state_snapshot(
                current, config_version=current.get("confirmed_config_version")
            )
            decision_state = agent_state_summary(current)
    if mode == "interactive" and stored and not rejecting:
        prior_action = (stored.get("agent_proposal") or {}).get("raw_action") or {}
        has_mc_tasks = any(row.get("stage") == "deep_search" for row in current.get("tasks") or [])
        if (
            has_mc_tasks
            and prior_action.get("tool") == "allocate_mc_bohb"
            and not _is_verified_second_mc_action(
                prior_action, current, (context or {}).get("effective_config") or config
            )
        ):
            from phase_agent.decisions.agent.choose_debug_next_action import (
                choose_debug_next_action,
            )

            allowed = [
                name
                for name in (config.get("agent") or {}).get("allowed_tools") or []
                if callable((registry.get(name) or {}).get("handler"))
            ]
            replacement = choose_debug_next_action(
                current,
                (context or {}).get("manager"),
                (context or {}).get("effective_config") or config,
                allowed_tools=allowed,
                user_message="继续",
            )
            if replacement and replacement.get("parameters", {}).get("round_kind") == "second":
                stored = deepcopy(stored)
                stored["revision"] = int(stored.get("revision", 0)) + 1
                stored.setdefault("feedback_history", []).append(
                    {
                        "revision_status": "unverified_mc_allocation_replaced",
                        "analysis": "原 MC 建议缺少首轮来源与第二轮预算预览；已替换为重新校验的方案。",
                    }
                )
                stored["agent_proposal"] = build_agent_proposal(
                    replacement, decision_state, runtime_state=current
                )
                current["pending_execution_policies"][pending_key] = stored
                human_feedback = (
                    None  # Approval for an unverified MC proposal cannot approve its replacement.
                )
            else:
                current["pending_execution_policies"].pop(pending_key, None)
                return _mc_continuation_block(
                    current, mode, (context or {}).get("effective_config") or config
                )
    if mode == "interactive" and stored:
        proposal = deepcopy(stored["agent_proposal"])
        record_id = stored["record_id"]
    elif mode == "replay":
        replay_action = deepcopy((replay_record or {}).get("final_action") or {})
        proposal = build_agent_proposal(replay_action, decision_state, runtime_state=current)
        record_id = _record_id(current, invocation_id)
    else:
        from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

        selected = select_initial_proposal(
            current=current,
            mode=mode,
            config=config,
            context=context,
            decision_state=decision_state,
            registry=registry,
            agent_client=agent_client,
            invocation_id=invocation_id,
            propose=propose_agent_tool_action,
            revise=revise_tool_proposal,
            prepare_debug=_prepare_debug_relax_screen_action,
            is_verified_second=_is_verified_second_mc_action,
            mc_block=_mc_continuation_block,
            model_failed=_model_failure_action,
            record_id_factory=_record_id,
        )
        if not selected.get("_proposal_selected"):
            return selected
        current, decision_state = selected["current"], selected["decision_state"]
        proposal, record_id = selected["proposal"], selected["record_id"]

    if mode == "interactive" and proposal.get("recommended_action") == "select_dft_candidates":
        from phase_agent.tools.workflows.dft_template_review import gate_template

        proposal, template_wait = gate_template(
            proposal,
            current,
            stored,
            human_feedback,
            (context or {}).get("effective_config") or config,
            (context or {}).get("manager"),
            agent_client,
        )
        if template_wait:
            human_feedback = None
            if stored:
                stored = deepcopy(stored)
                stored["agent_proposal"] = deepcopy(proposal)
                current["pending_execution_policies"][pending_key] = stored
    opinion = _extract_opinion(human_feedback) if mode == "interactive" and stored else None
    if advice_changed:
        opinion = (
            str(human_feedback.get("comment") or "")
            + "\n请根据更新后的人工长期建议重新分析，生成供人工审批的新 proposal。"
        )
    if opinion is not None:
        from phase_agent.tools.workflows.tool_proposal_revision import revise_pending_proposal

        return revise_pending_proposal(
            current=current,
            mode=mode,
            config=config,
            context=context,
            proposal=proposal,
            stored=stored,
            opinion=opinion,
            decision_state=decision_state,
            record_id=record_id,
            pending_key=pending_key,
            invocation_id=invocation_id,
            agent_client=agent_client,
            registry=registry,
            revise=revise_tool_proposal,
            prepare_debug_relax=_prepare_debug_relax_screen_action,
            apply_generation_defaults=_apply_generation_defaults,
        )

    return {
        "_graph_prepared": True,
        "current": current,
        "mode": mode,
        "config": config,
        "explicit_mode": explicit_mode,
        "pending_key": pending_key,
        "stored": stored,
        "proposal": proposal,
        "record_id": record_id,
        "decision_state": decision_state,
        "human_feedback": human_feedback,
        "invocation_id": invocation_id,
        "context": context,
        "execute": execute,
        "session": session,
        "registry": registry,
        "agent_client": agent_client,
        "replay_record": replay_record,
    }


def _approval_graph_node(frame):
    current = frame["current"]
    proposal = frame["proposal"]
    mode = frame["mode"]
    human_feedback = frame["human_feedback"]
    replay_record = frame["replay_record"]
    stored = frame["stored"]
    record_id = frame["record_id"]
    pending_key = frame["pending_key"]
    invocation_id = frame["invocation_id"]
    policy = apply_execution_policy(
        proposal,
        execution_mode=mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
    )
    from phase_agent.tools.policy.native_approval import native_approval_gate

    recovery_block = native_approval_gate(frame, policy)
    if recovery_block is not None:
        return recovery_block
    if policy["status"] == "awaiting_approval":
        record = _audit_record(
            record_id,
            mode,
            proposal,
            None,
            None,
            None,
            "awaiting_approval",
            deepcopy((stored or {}).get("feedback_history") or []),
        )
        _upsert_record(current, record)
        current["pending_execution_policies"][pending_key] = {
            "record_id": record_id,
            "agent_proposal": deepcopy(proposal),
            "revision": int((stored or {}).get("revision", 0)),
            "feedback_history": deepcopy((stored or {}).get("feedback_history") or []),
        }
        response = {
            "status": "awaiting_approval",
            "execution_mode": mode,
            "agent_proposal": proposal,
            "human_feedback": None,
            "final_action": None,
            "action": proposal["raw_action"],
            "validation": None,
            "execution": None,
            "execution_result": None,
            "record_id": record_id,
            "revision": int((stored or {}).get("revision", 0)),
            "feedback_history": deepcopy((stored or {}).get("feedback_history") or []),
        }
        current = update_state_snapshot(
            current, config_version=current.get("confirmed_config_version")
        )
        return {**response, "state": current, "idempotent_replay": False}

    current["pending_execution_policies"].pop(pending_key, None)
    if policy["status"] == "rejected_by_user":
        response = _policy_rejection(record_id, mode, proposal, policy)
        _upsert_record(
            current,
            _audit_record(
                record_id,
                mode,
                proposal,
                policy["human_feedback"],
                None,
                None,
                response["status"],
                deepcopy((stored or {}).get("feedback_history") or []),
            ),
        )
        current.setdefault("decisions", []).append(deepcopy(response))
        _store_completed(current, invocation_id, response)
        current = update_state_snapshot(
            current, config_version=current.get("confirmed_config_version")
        )
        return {**response, "state": current, "idempotent_replay": False}

    return {**frame, "current": current, "policy": policy, "_graph_prepared": True}


def _validation_graph_node(frame):
    current = frame["current"]
    policy = frame["policy"]
    session = frame["session"]
    registry = frame["registry"]
    mode = frame["mode"]
    config = frame["config"]
    decision_state = frame["decision_state"]
    explicit_mode = frame["explicit_mode"]
    execute = frame["execute"]
    action = policy["final_action"]
    validation = validate_tool_action(action, current, session, registry)
    if (
        not validation["valid"]
        and mode != "interactive"
        and (config.get("agent") or {}).get("rule_fallback", True)
    ):
        allowed = list((config.get("agent") or {}).get("allowed_tools") or [])
        action = propose_agent_tool_action(decision_state, agent_client=None, allowed_tools=allowed)
        action["fallback_reason"] = ",".join(validation["errors"])
        validation = validate_tool_action(action, current, session, registry)

    from phase_agent.tools.state.approved_direction import direction_action_errors

    direction_errors = direction_action_errors(action, current)
    if direction_errors:
        validation["valid"] = False
        validation.setdefault("errors", []).extend(direction_errors)
    should_execute = policy["execute"] if explicit_mode else bool(execute)
    ready, status = False, "planned_only"
    if not validation["valid"]:
        status = "rejected"
    elif mode == "interactive" and action.get("tool") == "run_calculation_stage":
        validation["valid"] = False
        validation.setdefault("errors", []).append("debug_mode_requires_remote_preparation")
        status = "rejected"
    elif should_execute:
        ready = True
    return {
        **frame,
        "action": action,
        "validation": validation,
        "dispatch_ready": ready,
        "status": status,
        "execution": None,
    }


def _dispatch_graph_node(frame):
    current = frame["current"]
    action = frame["action"]
    registry = frame["registry"]
    validation = frame["validation"]
    context = frame["context"]
    config = frame["config"]
    mode = frame["mode"]
    policy = frame["policy"]
    record_id = frame["record_id"]
    formal = action.get("tool") not in {"check_convergence", "pause_search"}
    owns_action_reservation = (
        formal and registry.get(action.get("tool"), {}).get("reservation_mode") != "children"
    )
    if owns_action_reservation:
        current = reserve_action_budget(
            current, action, config_version=validation["config_version"]
        )
    execution = execute_tool_action(
        action,
        registry=registry,
        context={
            **(context or {}),
            "confirmed_config": config,
            "invocation_id": record_id,
            "config_version": validation["config_version"],
            "execution_mode": mode,
            "human_approved_mc_full_plan": (
                mode == "interactive"
                and (policy.get("human_feedback") or {}).get("decision") == "approve"
                and action.get("tool") == "allocate_mc_bohb"
                and bool((action.get("parameters") or {}).get("budget_preview"))
            ),
        },
    )
    if execution["status"] == "execution_reconciliation_required":
        return {
            **frame,
            "current": frame["current"],
            "status": execution["status"],
            "execution": execution,
        }
    current, payload_status = _apply_execution_result(
        current, action, execution, record_id=record_id, formal=owns_action_reservation
    )
    status = payload_status or execution["status"]

    return {**frame, "current": current, "status": status, "execution": execution}


def _finish_graph_node(frame):
    current = frame["current"]
    status = frame["status"]
    mode = frame["mode"]
    proposal = frame["proposal"]
    policy = frame["policy"]
    action = frame["action"]
    validation = frame["validation"]
    execution = frame["execution"]
    record_id = frame["record_id"]
    stored = frame["stored"]
    invocation_id = frame["invocation_id"]
    response = {
        "status": status,
        "execution_mode": mode,
        "agent_proposal": proposal,
        "human_feedback": deepcopy(policy["human_feedback"]),
        "final_action": deepcopy(action),
        "action": deepcopy(action),
        "validation": validation,
        "execution": execution,
        "execution_result": deepcopy(execution),
        "record_id": record_id,
    }
    _upsert_record(
        current,
        _audit_record(
            record_id,
            mode,
            proposal,
            policy["human_feedback"],
            action,
            execution,
            status,
            deepcopy((stored or {}).get("feedback_history") or []),
        ),
    )
    current.setdefault("decisions", []).append(
        {"config_version": validation.get("config_version"), **compact_action_history(response)}
    )
    _store_completed(current, invocation_id, response)
    current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
    return {**response, "state": current, "idempotent_replay": False}


def _record_id(state, invocation_id):
    return invocation_id or f"action-{len(state.get('action_records', [])) + 1:06d}"


def _is_pending_relax_preparation(stored):
    action = (stored.get("agent_proposal") or {}).get("raw_action") or {}
    return (
        action.get("tool") == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "relax_inputs"
    )


def _prepare_debug_relax_screen_action(
    action, state, context, config, allowed_tools, *, invocation_id=None
):
    """Route interactive calculation proposals to the corresponding safe workflow."""
    parameters = action.get("parameters") or {}
    if action.get("tool") == "prepare_local_batch_files" and parameters.get("rebuild_inputs"):
        return action
    is_relax_input = (
        action.get("tool") == "prepare_local_batch_files"
        and parameters.get("mode") == "relax_inputs"
    )
    if action.get("tool") != "run_calculation_stage" and not is_relax_input:
        return action
    stage = action.get("stage") or parameters.get("stage")
    if action.get("tool") == "run_calculation_stage" and stage == "deep_search":
        from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action

        preparation = choose_debug_next_action(
            state,
            (context or {}).get("manager"),
            (context or {}).get("effective_config") or config,
            allowed_tools=allowed_tools,
        )
        if preparation and preparation.get("tool") in {
            "allocate_mc_bohb",
            "prepare_local_batch_files",
        }:
            return preparation
        completed_mc = any(
            row.get("stage") == "deep_search" and row.get("status") == "completed"
            for row in state.get("tasks") or []
        )
        reason = (
            "本轮 MC 结果已回收，且当前策略未开放第二段 MC；先评估收敛与后续阶段。"
            if completed_mc
            and not (config.get("mc_policy") or {}).get("second_segment_enabled", True)
            else "当前没有可直接执行的单结构 MC 动作；先评估已完成结果，再确定下一批任务。"
        )
        return {
            "tool": "check_convergence",
            "target_ids": [],
            "parameters": {},
            "budget": 0.0,
            "reason": reason,
            "expected_purpose": "评估本轮已回收结果和下一阶段条件。",
            "decision_source": "debug_mc_batch_guard",
        }
    if not is_relax_input and stage not in {"relax_screen", "relax_and_feature"}:
        return action
    if "prepare_local_batch_files" not in allowed_tools:
        return action
    from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action

    preparation = choose_debug_next_action(
        state,
        (context or {}).get("manager"),
        (context or {}).get("effective_config") or config,
        allowed_tools=allowed_tools,
    )
    if preparation is None:
        return (
            action
            if is_relax_input
            else {
                **action,
                "tool": "prepare_local_batch_files",
                "parameters": {"mode": "relax_inputs", "selection_scope": "all_registered"},
                "reason": "先准备或复用所有已入库结构的 Relax 输入文件。",
                "decision_source": "debug_relax_preparation_guard",
            }
        )
    suffix = invocation_id or preparation.get("task_key") or "missing-invocation"
    preparation["task_key"] = f"prepare-relax-inputs:{suffix}"
    preparation["parameters"] = {"mode": "relax_inputs", "selection_scope": "all_registered"}
    return preparation


def _is_verified_second_mc_action(action, state, config):
    """Require a frozen second-round preview before MC can follow MC results."""
    from phase_agent.science.mc.second_round_state import (
        first_round_source,
        second_round_already_allocated,
    )

    params = action.get("parameters") or {}
    preview = params.get("budget_preview") or {}
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name")
    source = first_round_source(state, version)
    if (
        not (config.get("mc_policy") or {}).get("second_segment_enabled", True)
        or source is None
        or params.get("round_kind") != "second"
        or preview.get("round_kind") != "second"
        or params.get("first_round_source_checksum") != source["checksum"]
        or preview.get("first_round_source_checksum") != source["checksum"]
        or not preview.get("allocations")
        or second_round_already_allocated(state, version, source["checksum"])
    ):
        return False
    targets = set(action.get("target_ids") or [])
    allowed_targets = set(source["branch_ids"])
    allocations = preview["allocations"]
    try:
        segment_one_only = all(int(row.get("segment_index", -1)) == 1 for row in allocations)
    except (TypeError, ValueError):
        segment_one_only = False
    allocated_branches = {row.get("branch_id") for row in allocations}
    return bool(
        targets
        and targets.issubset(allowed_targets)
        and segment_one_only
        and allocated_branches.issubset(targets)
    )


def _mc_continuation_block(state, mode, config):
    """Stop unsafe LLM fallback and explain which prerequisite is missing."""
    if not (config.get("mc_policy") or {}).get("second_segment_enabled", True):
        status = "configuration_revision_required"
        reason = "首轮 MC 已完成，但当前已确认配置关闭了第二段 MC；请修订并确认 MC 策略后再生成第二轮方案。"
    else:
        status = "not_configured"
        reason = (
            "已有 MC 结果，但当前轮次或相图证据不足以生成经校验的下一段 MC 分配；"
            "系统不会回退到 Relax，也未生成新的计算任务。"
        )
    return {
        "status": status,
        "execution_mode": mode,
        "reason": reason,
        "agent_proposal": None,
        "final_action": None,
        "action": None,
        "validation": None,
        "execution": None,
        "execution_result": None,
        "record_id": None,
        "state": state,
        "idempotent_replay": False,
    }


def _can_rebind_empty_run(state):
    """Allow a newly confirmed config to replace only a scientifically empty run."""
    if state.get("tasks") or state.get("branch_candidates"):
        return False
    if any(
        item.get("status") in {"reserved", "submitted", "running", "completed"}
        for item in (state.get("budget_reservations") or {}).values()
    ):
        return False
    usage = state.get("budget_usage") or {}
    if float(usage.get("total_relative_cost") or 0.0) > 0:
        return False
    return not (usage.get("stages") or {})


def _extract_opinion(feedback):
    if isinstance(feedback, str):
        return (
            None if feedback.strip().lower() in {"approve", "同意", "reject"} else feedback.strip()
        )
    if not isinstance(feedback, dict):
        return None
    decision = feedback.get("decision")
    comment = str(feedback.get("comment") or "").strip()
    if comment.lower() in {"approve", "同意"}:
        return None
    if decision in {None, "comment", "revise"} and comment:
        return comment
    return None

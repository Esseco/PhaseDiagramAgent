"""Confirmed-config tool step with an explicit execution policy gate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from decision_layer.agent.propose_tool_action import (
    _apply_generation_defaults,
    propose_agent_tool_action,
)
from decision_layer.agent.revise_tool_proposal import revise_tool_proposal
from data_layer.memory.decision_memory import update_long_term_advice, update_long_term_memory
from execution_layer.dispatch.execute_tool_action import execute_tool_action
from execution_layer.policy.execution_policy import apply_execution_policy, build_agent_proposal
from execution_layer.budget.reserve_action_budget import reserve_action_budget
from execution_layer.budget.record_budget_usage import record_budget_usage
from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.policy.validate_tool_action import validate_tool_action
from execution_layer.state.state_manager import agent_state_summary, update_state_snapshot
from execution_layer.workflows.compact_action_history import compact_action_history


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
        return {**deepcopy(current["invocations"][invocation_id]), "state": current, "idempotent_replay": True}

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
            return {"status": "configuration_version_mismatch", "execution_mode": mode,
                    "reason": f"当前运行绑定 {bound_version}，新配置为 {snapshot_version}；已有任务或预算记录，不能直接混用。",
                    "agent_proposal": None, "final_action": None, "action": None,
                    "validation": None, "execution": None, "execution_result": None,
                    "state": current, "idempotent_replay": False}
    current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
    decision_state = agent_state_summary(current)
    decision_state["user_message"] = str((context or {}).get("user_message") or "")

    pending_key = invocation_id or "__single_interactive_action__"
    stored = current["pending_execution_policies"].get(pending_key)
    rejecting = isinstance(human_feedback, dict) and human_feedback.get("decision") == "reject"
    if mode == "interactive" and stored and not rejecting and _is_pending_relax_preparation(stored):
        allowed = [name for name in (config.get("agent") or {}).get("allowed_tools") or []
                   if callable((registry.get(name) or {}).get("handler"))]
        from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
        next_action = choose_debug_next_action(
            current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
            allowed_tools=allowed, user_message="继续",
        )
        if next_action and next_action.get("tool") == "allocate_mc_bohb":
            stored = deepcopy(stored)
            previous = (stored.get("agent_proposal") or {}).get("raw_action") or {}
            stored["revision"] = int(stored.get("revision", 0)) + 1
            stored.setdefault("feedback_history", []).append({
                "revision_status": "relax_results_recovered",
                "prior_action": previous.get("tool"),
                "analysis": "已回收的 Relax 结果覆盖该批结构；改为 MC 预算与输入文件准备。",
            })
            stored["agent_proposal"] = build_agent_proposal(next_action, decision_state)
            current["pending_execution_policies"][pending_key] = stored
            human_feedback = None  # Approval for the old Relax proposal cannot approve MC.
    advice_changed = False
    if mode == "interactive" and stored and isinstance(human_feedback, dict) and human_feedback.get("long_term_advice") is not None:
        updated = update_long_term_advice(current, human_feedback["long_term_advice"], source=f"{pending_key}:revision-{stored.get('revision', 0)}")
        advice_changed = updated.get("decision_memory") != current.get("decision_memory")
        current = updated
        if advice_changed:
            current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
            decision_state = agent_state_summary(current)
    if mode == "interactive" and stored and isinstance(human_feedback, dict) and human_feedback.get("long_term_memory") is not None:
        updated = update_long_term_memory(current, human_feedback["long_term_memory"],
                                          source=f"{pending_key}:revision-{stored.get('revision', 0)}")
        memory_changed = updated.get("decision_memory") != current.get("decision_memory")
        advice_changed = advice_changed or memory_changed
        current = updated
        if memory_changed:
            current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
            decision_state = agent_state_summary(current)
    if mode == "interactive" and stored:
        proposal = deepcopy(stored["agent_proposal"])
        record_id = stored["record_id"]
    elif mode == "replay":
        replay_action = deepcopy((replay_record or {}).get("final_action") or {})
        proposal = build_agent_proposal(replay_action, decision_state)
        record_id = _record_id(current, invocation_id)
    else:
        allowed = [name for name in (config.get("agent") or {}).get("allowed_tools") or []
                   if callable((registry.get(name) or {}).get("handler"))]
        from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
        mc_files_pending = any(row.get("stage") == "deep_search" and row.get("status") == "pending"
                               and not row.get("slurm_batch_id") for row in current.get("tasks") or [])
        safe_next = (choose_debug_next_action(
            current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
            allowed_tools=allowed, user_message=(context or {}).get("user_message"),
        ) if mode == "interactive" and (mc_files_pending or not callable(agent_client) or
             str((context or {}).get("user_message") or "").strip() in {"继续", "下一步", "然后呢"}) else None)
        action = safe_next or propose_agent_tool_action(
            decision_state, agent_client=agent_client, allowed_tools=allowed, config=config
        )
        if mode == "interactive" and action.get("decision_source") == "rule":
            recovery = choose_debug_next_action(
                current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
                allowed_tools=allowed, user_message=(context or {}).get("user_message"),
            )
            if recovery:
                recovery["_llm_usage"] = action.get("_llm_usage")
                recovery["fallback_reason"] = action.get("fallback_reason")
                action = recovery
        if mode == "interactive":
            original_usage = action.get("_llm_usage")
            original_evidence = action.get("evidence_refs")
            action = _prepare_debug_relax_screen_action(
                action, current, context, config, allowed, invocation_id=invocation_id)
            if original_usage:
                action["_llm_usage"] = original_usage
            if original_evidence:
                action["evidence_refs"] = original_evidence
        if action.get("_llm_usage"):
            current = record_budget_usage(
                current, {"llm_usage": action["_llm_usage"], "iteration": current.get("iteration", 0)}
            )
        mc_intent = current.get("mc_budget_intent") or {}
        if (action.get("tool") == "allocate_mc_bohb"
                and (action.get("parameters") or {}).get("mc_budget") == mc_intent.get("steps")):
            current.pop("mc_budget_intent", None)
        proposal = build_agent_proposal(action, decision_state)
        record_id = _record_id(current, invocation_id)

    opinion = _extract_opinion(human_feedback) if mode == "interactive" and stored else None
    if advice_changed:
        opinion = str(human_feedback.get("comment") or "") + "\n请根据更新后的人工长期建议重新分析，生成供人工审批的新 proposal。"
    if opinion is not None:
        from decision_layer.agent.resolve_mc_budget_feedback import (
            resolve_mc_budget_feedback, resolve_mc_full_plan_steps,
        )
        requested_mc_steps = (resolve_mc_budget_feedback(opinion, proposal.get("raw_action"))
                              or resolve_mc_full_plan_steps(opinion, proposal.get("raw_action")))
        maximum_mc_budget = (config.get("round_strategy") or {}).get("maximum_mc_budget")
        if (requested_mc_steps is not None and maximum_mc_budget is not None
                and requested_mc_steps > int(maximum_mc_budget)):
            current["pending_execution_policies"].pop(pending_key, None)
            current.setdefault("cancelled_proposals", []).append({
                "invocation_id": pending_key, "reason": "mc_budget_exceeds_confirmed_maximum",
                "requested_steps": requested_mc_steps,
                "config_version": current.get("confirmed_config_version")})
            current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
            return {"status": "configuration_revision_required", "execution_mode": mode,
                    "reason": (f"本轮 MC 目标 {requested_mc_steps} 步超过已确认配置的单轮上限 "
                               f"{maximum_mc_budget} 步；先修订 round_strategy.maximum_mc_budget "
                               "并确认新配置，不能批准旧建议。"),
                    "agent_proposal": None, "final_action": None, "action": None,
                    "validation": None, "execution": None, "execution_result": None,
                    "record_id": record_id, "state": current, "idempotent_replay": False}
        from decision_layer.agent.resolve_explicit_generation_request import _requested_branch_batch_size
        requested_batch = _requested_branch_batch_size(opinion)
        if requested_batch is not None and (proposal.get("raw_action") or {}).get("tool") == "generate_branches":
            run = config.get("run") or {}
            strategy = config.get("round_strategy") or {}
            quota_limit = max(int(run.get("total_quota", 0)),
                              int(strategy.get("generation_quota_total", 0)))
            if requested_batch > quota_limit:
                current["pending_execution_policies"].pop(pending_key, None)
                current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
                return {"status": "configuration_revision_required", "execution_mode": mode,
                        "reason": f"要求入选 {requested_batch} 个 branch，但当前已确认生成配额上限为 {quota_limit}；旧建议已取消。请先修订并确认配置。",
                        "agent_proposal": None, "final_action": None, "action": None,
                        "validation": None, "execution": None, "execution_result": None,
                        "record_id": record_id, "state": current, "idempotent_replay": False}
        allowed = [name for name in (config.get("agent") or {}).get("allowed_tools") or []
                   if callable((registry.get(name) or {}).get("handler"))]
        from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
        original_action = proposal.get("raw_action") or {}
        safe_next = (choose_debug_next_action(
            current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
            allowed_tools=allowed, user_message=opinion,
            target_branch_ids=original_action.get("target_ids"),
        ) if mode == "interactive" and original_action.get("tool") == "run_calculation_stage" else None)
        revision = ({"action": safe_next, "analysis": safe_next["reason"],
                     "revision_status": "debug_preparation_required"} if safe_next else
                    revise_tool_proposal(proposal, opinion, state=decision_state,
                                         allowed_tools=allowed, agent_client=agent_client,
                                         source_state=current, manager=(context or {}).get("manager"),
                                         config=(context or {}).get("effective_config") or config))
        if revision.get("revision_status") == "mc_budget_preview_unavailable":
            current["pending_execution_policies"].pop(pending_key, None)
            current.setdefault("cancelled_proposals", []).append({
                "invocation_id": pending_key, "reason": revision["revision_status"],
                "config_version": current.get("confirmed_config_version")})
            current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
            return {"status": "mc_budget_preview_unavailable", "execution_mode": mode,
                    "reason": revision["analysis"], "agent_proposal": None,
                    "final_action": None, "action": None, "validation": None,
                    "execution": None, "execution_result": None,
                    "record_id": record_id, "state": current, "idempotent_replay": False}
        if (revision.get("action") or {}).get("tool") == "generate_branches":
            revision["action"] = _apply_generation_defaults(revision["action"], decision_state, config)
        if mode == "interactive":
            candidate_action = revision.get("action") or {}
            revised_action = _prepare_debug_relax_screen_action(
                candidate_action, current, context, config, allowed, invocation_id=invocation_id)
            if revised_action.get("tool") != candidate_action.get("tool"):
                revision["action"] = revised_action
                revision["analysis"] = revised_action.get("reason", "先准备 Relax 输入文件。")
                revision["revision_status"] = "debug_preparation_required"
        revision_usage = revision.get("llm_usage") or (revision.get("action") or {}).get("_llm_usage")
        if revision_usage:
            current = record_budget_usage(current, {"llm_usage": revision_usage,
                                                   "iteration": current.get("iteration", 0)})
        proposal = build_agent_proposal(revision["action"], decision_state)
        history = deepcopy(stored.get("feedback_history") or [])
        history.append({"comment": opinion, "revision_status": revision["revision_status"], "analysis": revision["analysis"]})
        revision_number = int(stored.get("revision", 0)) + 1
        current["pending_execution_policies"][pending_key] = {"record_id": record_id, "agent_proposal": deepcopy(proposal), "revision": revision_number, "feedback_history": history}
        record = _audit_record(record_id, mode, proposal, {"decision": "comment", "comment": opinion}, None, None, "awaiting_approval")
        record["feedback_history"] = history
        _upsert_record(current, record)
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        return {"status": "awaiting_approval", "execution_mode": mode, "agent_proposal": proposal, "human_feedback": {"decision": "comment", "comment": opinion}, "feedback_history": history, "revision": revision_number, "final_action": None, "action": proposal["raw_action"], "validation": None, "execution": None, "execution_result": None, "record_id": record_id, "state": current, "idempotent_replay": False}

    policy = apply_execution_policy(
        proposal,
        execution_mode=mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
    )
    if policy["status"] == "awaiting_approval":
        record = _audit_record(record_id, mode, proposal, None, None, None, "awaiting_approval", deepcopy((stored or {}).get("feedback_history") or []))
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
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        return {**response, "state": current, "idempotent_replay": False}

    current["pending_execution_policies"].pop(pending_key, None)
    if policy["status"] == "rejected_by_user":
        response = _policy_rejection(record_id, mode, proposal, policy)
        _upsert_record(
            current,
            _audit_record(record_id, mode, proposal, policy["human_feedback"], None, None, response["status"], deepcopy((stored or {}).get("feedback_history") or [])),
        )
        current.setdefault("decisions", []).append(deepcopy(response))
        _store_completed(current, invocation_id, response)
        current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
        return {**response, "state": current, "idempotent_replay": False}

    action = policy["final_action"]
    validation = validate_tool_action(action, current, session, registry)
    if not validation["valid"] and mode != "interactive" and (config.get("agent") or {}).get("rule_fallback", True):
        allowed = list((config.get("agent") or {}).get("allowed_tools") or [])
        action = propose_agent_tool_action(decision_state, agent_client=None, allowed_tools=allowed)
        action["fallback_reason"] = ",".join(validation["errors"])
        validation = validate_tool_action(action, current, session, registry)

    should_execute = policy["execute"] if explicit_mode else bool(execute)
    if not validation["valid"]:
        status, execution = "rejected", None
    elif mode == "interactive" and action.get("tool") == "run_calculation_stage":
        validation["valid"] = False
        validation.setdefault("errors", []).append("debug_mode_requires_remote_preparation")
        status, execution = "rejected", None
    elif not should_execute:
        status, execution = "planned_only", None
    else:
        formal = action.get("tool") not in {"check_convergence", "pause_search"}
        owns_action_reservation = formal and registry.get(action.get("tool"), {}).get("reservation_mode") != "children"
        if owns_action_reservation:
            current = reserve_action_budget(current, action, config_version=validation["config_version"])
        execution = execute_tool_action(
            action,
            registry=registry,
            context={
                **(context or {}),
                "confirmed_config": config,
                "config_version": validation["config_version"],
                "execution_mode": mode,
                "human_approved_mc_full_plan": (
                    mode == "interactive"
                    and (policy.get("human_feedback") or {}).get("decision") == "approve"
                    and action.get("tool") == "allocate_mc_bohb"
                    and bool((action.get("parameters") or {}).get("budget_preview"))),
            },
        )
        current, payload_status = _apply_execution_result(
            current, action, execution, record_id=record_id, formal=owns_action_reservation
        )
        status = payload_status or execution["status"]

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
        _audit_record(record_id, mode, proposal, policy["human_feedback"], action, execution, status, deepcopy((stored or {}).get("feedback_history") or [])),
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
    action = ((stored.get("agent_proposal") or {}).get("raw_action") or {})
    return (action.get("tool") == "prepare_local_batch_files"
            and (action.get("parameters") or {}).get("mode") == "relax_inputs")


def _prepare_debug_relax_screen_action(action, state, context, config, allowed_tools, *, invocation_id=None):
    """Turn a proposed Relax calculation into portable input-file preparation."""
    parameters = action.get("parameters") or {}
    if action.get("tool") == "prepare_local_batch_files" and parameters.get("rebuild_inputs"):
        return action
    is_relax_input = (action.get("tool") == "prepare_local_batch_files"
                      and parameters.get("mode") == "relax_inputs")
    if action.get("tool") != "run_calculation_stage" and not is_relax_input:
        return action
    stage = action.get("stage") or parameters.get("stage")
    if not is_relax_input and stage not in {"relax_screen", "relax_and_feature"}:
        return action
    if "prepare_local_batch_files" not in allowed_tools:
        return action
    from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
    preparation = choose_debug_next_action(
        state, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
        allowed_tools=allowed_tools,
    )
    if preparation is None:
        return action if is_relax_input else {
            **action, "tool": "prepare_local_batch_files",
            "parameters": {"mode": "relax_inputs", "selection_scope": "all_registered"},
            "reason": "先准备或复用所有已入库结构的 Relax 输入文件。",
            "decision_source": "debug_relax_preparation_guard",
        }
    suffix = invocation_id or preparation.get("task_key") or "missing-invocation"
    preparation["task_key"] = f"prepare-relax-inputs:{suffix}"
    preparation["parameters"] = {"mode": "relax_inputs", "selection_scope": "all_registered"}
    return preparation


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
        return None if feedback.strip().lower() in {"approve", "同意", "reject"} else feedback.strip()
    if not isinstance(feedback, dict):
        return None
    decision = feedback.get("decision")
    comment = str(feedback.get("comment") or "").strip()
    if comment.lower() in {"approve", "同意"}:
        return None
    if decision in {None, "comment", "revise"} and comment:
        return comment
    return None


def _audit_record(record_id, mode, proposal, feedback, final_action, result, status, feedback_history=None):
    return {
        "record_id": record_id,
        "execution_mode": mode,
        "status": status,
        "agent_proposal": deepcopy(proposal),
        "human_feedback": deepcopy(feedback),
        "final_action": deepcopy(final_action),
        "execution_result": compact_action_history(result),
        "feedback_history": deepcopy(feedback_history or []),
    }


def _upsert_record(state, record):
    records = state.setdefault("action_records", [])
    for index, existing in enumerate(records):
        if existing.get("record_id") == record["record_id"]:
            records[index] = deepcopy(record)
            return
    records.append(deepcopy(record))


def _store_completed(state, invocation_id, response):
    if invocation_id:
        state.setdefault("invocations", {})[invocation_id] = compact_action_history(response)


def _policy_rejection(record_id, mode, proposal, policy):
    return {
        "status": "rejected_by_user",
        "execution_mode": mode,
        "agent_proposal": proposal,
        "human_feedback": deepcopy(policy["human_feedback"]),
        "final_action": None,
        "action": proposal["raw_action"],
        "validation": None,
        "execution": None,
        "execution_result": None,
        "record_id": record_id,
    }


def _apply_execution_result(current, action, execution, *, record_id, formal):
    if execution.get("status") != "completed" or not isinstance(execution.get("result"), dict):
        return current, None
    payload = deepcopy(execution["result"])
    if isinstance(payload.get("state"), dict):
        merged = deepcopy(current)
        merged.update(deepcopy(payload["state"]))
        # A handler receives the state from before this approval was consumed.
        # Its returned scientific state must not restore that pending approval.
        merged["pending_execution_policies"] = deepcopy(current.get("pending_execution_policies") or {})
        current = merged
    payload_status = payload.get("status")
    if not formal:
        return current, payload_status
    task_key = payload.get("task_key") or action.get("task_key")
    task_id = payload.get("task_id") or f"{record_id}:task"
    if payload_status in {"pending", "running", "completed", "failed", "timeout", "cancelled"}:
        task_result = {
            **payload,
            "task_id": task_id,
            "task_key": task_key,
            "status": payload_status,
        }
        reconciled = reconcile_task_results(current, [task_result])
        current = reconciled["state"]
    return current, payload_status

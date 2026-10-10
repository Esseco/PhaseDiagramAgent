"""Human feedback revision of a pending proposal; never executes tools."""

from copy import deepcopy
from phase_agent.tools.budget.record_budget_usage import record_budget_usage
from phase_agent.tools.policy.execution_policy import build_agent_proposal
from phase_agent.tools.state.state_manager import update_state_snapshot
from phase_agent.tools.workflows.tool_outcomes import _audit_record, _upsert_record


def revise_pending_proposal(
    *,
    current,
    mode,
    config,
    context,
    proposal,
    stored,
    opinion,
    decision_state,
    record_id,
    pending_key,
    invocation_id,
    agent_client,
    registry,
    revise,
    prepare_debug_relax,
    apply_generation_defaults,
):
    from phase_agent.decisions.agent.resolve_mc_budget_feedback import (
        resolve_mc_budget_feedback,
        resolve_mc_full_plan_steps,
    )

    requested_mc_steps = resolve_mc_budget_feedback(
        opinion, proposal.get("raw_action")
    ) or resolve_mc_full_plan_steps(opinion, proposal.get("raw_action"))
    maximum_mc_budget = (config.get("round_strategy") or {}).get("maximum_mc_budget")
    if (
        requested_mc_steps is not None
        and maximum_mc_budget is not None
        and requested_mc_steps > int(maximum_mc_budget)
    ):
        current["pending_execution_policies"].pop(pending_key, None)
        current.setdefault("cancelled_proposals", []).append(
            {
                "invocation_id": pending_key,
                "reason": "mc_budget_exceeds_confirmed_maximum",
                "requested_steps": requested_mc_steps,
                "config_version": current.get("confirmed_config_version"),
            }
        )
        current = update_state_snapshot(
            current, config_version=current.get("confirmed_config_version")
        )
        return {
            "status": "configuration_revision_required",
            "execution_mode": mode,
            "reason": (
                f"本轮 MC 目标 {requested_mc_steps} 步超过已确认配置的单轮上限 "
                f"{maximum_mc_budget} 步；先修订 round_strategy.maximum_mc_budget "
                "并确认新配置，不能批准旧建议。"
            ),
            "agent_proposal": None,
            "final_action": None,
            "action": None,
            "validation": None,
            "execution": None,
            "execution_result": None,
            "record_id": record_id,
            "state": current,
            "idempotent_replay": False,
        }
    from phase_agent.decisions.agent.resolve_explicit_generation_request import (
        _requested_branch_batch_size,
    )

    requested_batch = _requested_branch_batch_size(opinion)
    if (
        requested_batch is not None
        and (proposal.get("raw_action") or {}).get("tool") == "generate_branches"
    ):
        run = config.get("run") or {}
        strategy = config.get("round_strategy") or {}
        quota_limit = max(
            int(run.get("total_quota", 0)), int(strategy.get("generation_quota_total", 0))
        )
        if requested_batch > quota_limit:
            current["pending_execution_policies"].pop(pending_key, None)
            current = update_state_snapshot(
                current, config_version=current.get("confirmed_config_version")
            )
            return {
                "status": "configuration_revision_required",
                "execution_mode": mode,
                "reason": f"要求入选 {requested_batch} 个 branch，但当前已确认生成配额上限为 {quota_limit}；旧建议已取消。请先修订并确认配置。",
                "agent_proposal": None,
                "final_action": None,
                "action": None,
                "validation": None,
                "execution": None,
                "execution_result": None,
                "record_id": record_id,
                "state": current,
                "idempotent_replay": False,
            }
    allowed = [
        name
        for name in (config.get("agent") or {}).get("allowed_tools") or []
        if callable((registry.get(name) or {}).get("handler"))
    ]
    from phase_agent.decisions.agent.choose_debug_next_action import choose_debug_next_action

    original_action = proposal.get("raw_action") or {}
    safe_next = (
        choose_debug_next_action(
            current,
            (context or {}).get("manager"),
            (context or {}).get("effective_config") or config,
            allowed_tools=allowed,
            user_message=opinion,
            target_branch_ids=original_action.get("target_ids"),
        )
        if mode == "interactive" and original_action.get("tool") == "run_calculation_stage"
        else None
    )
    revision = (
        {
            "action": safe_next,
            "analysis": safe_next["reason"],
            "revision_status": "debug_preparation_required",
        }
        if safe_next
        else revise(
            proposal,
            opinion,
            state=decision_state,
            allowed_tools=allowed,
            agent_client=agent_client,
            source_state=current,
            manager=(context or {}).get("manager"),
            config=(context or {}).get("effective_config") or config,
        )
    )
    if revision.get("revision_status") == "mc_budget_preview_unavailable":
        current["pending_execution_policies"].pop(pending_key, None)
        current.setdefault("cancelled_proposals", []).append(
            {
                "invocation_id": pending_key,
                "reason": revision["revision_status"],
                "config_version": current.get("confirmed_config_version"),
            }
        )
        current = update_state_snapshot(
            current, config_version=current.get("confirmed_config_version")
        )
        return {
            "status": "mc_budget_preview_unavailable",
            "execution_mode": mode,
            "reason": revision["analysis"],
            "agent_proposal": None,
            "final_action": None,
            "action": None,
            "validation": None,
            "execution": None,
            "execution_result": None,
            "record_id": record_id,
            "state": current,
            "idempotent_replay": False,
        }
    if (revision.get("action") or {}).get("tool") == "generate_branches":
        revision["action"] = apply_generation_defaults(revision["action"], decision_state, config)
    if mode == "interactive":
        candidate_action = revision.get("action") or {}
        revised_action = prepare_debug_relax(
            candidate_action, current, context, config, allowed, invocation_id=invocation_id
        )
        if revised_action.get("tool") != candidate_action.get("tool"):
            revision["action"] = revised_action
            revision["analysis"] = revised_action.get("reason", "先准备 Relax 输入文件。")
            revision["revision_status"] = "debug_preparation_required"
    revision_usage = revision.get("llm_usage") or (revision.get("action") or {}).get("_llm_usage")
    if revision_usage:
        current = record_budget_usage(
            current, {"llm_usage": revision_usage, "iteration": current.get("iteration", 0)}
        )
    if revision.get("revision_status") == "revision_failed":
        return {
            "status": "rejected",
            "state": current,
            "reason": revision["analysis"] + "；原待确认方案保留，未执行任务。",
        }
    from phase_agent.tools.workflows.attach_dft_preview import attach_dft_preview

    revised_action, preview_error = attach_dft_preview(
        revision["action"], current, (context or {}).get("effective_config") or config
    )
    if preview_error:
        return {
            "status": "rejected",
            "state": current,
            "reason": preview_error + "；原方案未被替换，未执行任务。",
        }
    from phase_agent.analysis.cost.generation_preflight import attach_generation_preflight

    try:
        revised_action = attach_generation_preflight(
            revised_action, current, context, (context or {}).get("effective_config") or config
        )
    except ValueError as error:
        return {"status": "rejected", "state": current, "reason": str(error)}
    proposal = build_agent_proposal(revised_action, decision_state, runtime_state=current)
    history = deepcopy(stored.get("feedback_history") or [])
    history.append(
        {
            "comment": opinion,
            "revision_status": revision["revision_status"],
            "analysis": revision["analysis"],
        }
    )
    revision_number = int(stored.get("revision", 0)) + 1
    current["pending_execution_policies"][pending_key] = {
        "record_id": record_id,
        "agent_proposal": deepcopy(proposal),
        "revision": revision_number,
        "feedback_history": history,
    }
    record = _audit_record(
        record_id,
        mode,
        proposal,
        {"decision": "comment", "comment": opinion},
        None,
        None,
        "awaiting_approval",
    )
    record["feedback_history"] = history
    _upsert_record(current, record)
    current = update_state_snapshot(current, config_version=current.get("confirmed_config_version"))
    return {
        "status": "awaiting_approval",
        "execution_mode": mode,
        "agent_proposal": proposal,
        "human_feedback": {"decision": "comment", "comment": opinion},
        "feedback_history": history,
        "revision": revision_number,
        "final_action": None,
        "action": proposal["raw_action"],
        "validation": None,
        "execution": None,
        "execution_result": None,
        "record_id": record_id,
        "state": current,
        "idempotent_replay": False,
    }

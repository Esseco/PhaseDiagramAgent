"""Resolve MC scope before approval is consumed or execution receipts are created."""

from copy import deepcopy


def needs_choice(proposal):
    action = (proposal or {}).get("raw_action") or {}
    params = action.get("parameters") or {}
    return (
        action.get("tool") == "allocate_mc_bohb"
        and (params.get("budget_preview") or {}).get("compression_requires_user_choice")
        and params.get("compression_choice") != "reduce_branch_count"
    )


def prompt(proposal):
    preview = proposal["raw_action"]["parameters"]["budget_preview"]
    return (
        f"MC 范围需要选择：完整方案 {preview['full_plan_branch_count']} 个分支、{preview['full_plan_steps']} 步；"
        f"当前 {preview['target_steps']} 步可覆盖 {preview['selected_branch_count']} 个分支。\n"
        "回复“完整执行”或“按当前步数缩小范围”。选择后会展示更新方案，再回复“同意”执行；也可回复“拒绝”取消。"
    )


def preserve(frame, *, reason=None):
    state = frame["current"]
    key = frame["pending_key"]
    proposal = frame["proposal"]
    from phase_agent.analysis.state.project_authority import project_identity

    stored = deepcopy(frame.get("stored") or {})
    stored.update(
        record_id=frame["record_id"],
        agent_proposal=deepcopy(proposal),
        project_identity=project_identity(state),
    )
    state.setdefault("pending_execution_policies", {})[key] = stored
    return {
        "status": "awaiting_approval",
        "state": state,
        "agent_proposal": proposal,
        "mc_coverage_question": True,
        "reason": reason or prompt(proposal),
        "record_id": frame["record_id"],
        "steps_executed": 0,
        "submitted": False,
    }


def resolve(frame):
    state = frame["current"]
    context = frame.get("context") or {}
    key = frame["invocation_id"] or "__single_interactive_action__"
    stored = (state.get("pending_execution_policies") or {}).get(key)
    if stored is None:
        candidates = [
            (k, v)
            for k, v in (state.get("pending_execution_policies") or {}).items()
            if needs_choice(v.get("agent_proposal"))
        ]
        if len(candidates) == 1:
            key, stored = candidates[0]
    if not stored or not needs_choice(stored.get("agent_proposal")):
        return None
    feedback = frame.get("human_feedback") or {}
    message = str(
        context.get("user_message")
        or (feedback.get("comment") if isinstance(feedback, dict) else "")
        or ""
    ).strip()
    full = message in {"完整执行", "完整运行", "执行完整方案", "全部运行"}
    reduced = message in {"按当前步数缩小范围", "缩小范围", "缩小分支范围"}
    if not full and not reduced:
        return None
    proposal = deepcopy(stored["agent_proposal"])
    action = proposal["raw_action"]
    params = action["parameters"]
    preview = params["budget_preview"]
    if full:
        steps = preview["full_plan_steps"]
        maximum = (frame["config"].get("round_strategy") or {}).get("maximum_mc_budget")
        if maximum is not None and steps > maximum:
            return preserve(
                {
                    **frame,
                    "pending_key": key,
                    "proposal": proposal,
                    "stored": stored,
                    "record_id": stored["record_id"],
                },
                reason=f"完整方案需 {steps} 步，超过已确认单轮上限 {maximum} 步。请先修订并确认配置，或选择缩小范围；原方案仍保留。",
            )
        params["mc_budget"] = steps
    else:
        action["target_ids"] = list(dict.fromkeys(r["branch_id"] for r in preview["allocations"]))
        params["compression_choice"] = "reduce_branch_count"
    action["decision_source"] = "explicit_user_instruction"
    from phase_agent.tools.workflows.prepare_mc_preview import prepare_mc_preview
    from phase_agent.tools.policy.execution_policy import build_agent_proposal

    try:
        action = prepare_mc_preview(
            action, state, context, context.get("effective_config") or frame["config"]
        )
        proposal = build_agent_proposal(
            action, frame.get("decision_state") or state, runtime_state=state
        )
    except (ValueError, KeyError) as error:
        return preserve(
            {
                **frame,
                "pending_key": key,
                "proposal": stored["agent_proposal"],
                "stored": stored,
                "record_id": stored["record_id"],
            },
            reason="范围方案未通过重新核对，原方案保留：" + str(error),
        )
    stored = deepcopy(stored)
    stored["revision"] = int(stored.get("revision", 0)) + 1
    from phase_agent.tools.policy.file_approval import proposal_hash

    original = stored.setdefault("original_record_id", stored["record_id"])
    stored["record_id"] = original + ":mc-scope:" + proposal_hash(proposal)[:16]
    stored.setdefault("feedback_history", []).append(
        {"comment": message, "revision_status": "explicit_mc_scope_choice"}
    )
    return preserve(
        {
            **frame,
            "pending_key": key,
            "proposal": proposal,
            "stored": stored,
            "record_id": stored["record_id"],
        },
        reason="范围选择已记录，请查看更新方案后回复“同意”；本次未执行。",
    )

"""Scientific choice precedes human execution approval."""

from copy import deepcopy

REVIEW_POLICY = "trained_model_switch_v4"


def review_training_choice(
    state, waits, agent_client, *, state_path=None, user_message=None, revise_approved=False
):
    feedback = str(user_message or "").strip()
    if feedback.lower() in {"继续", "继续原运行", "同意", "拒绝", "continue", "approve", "reject"}:
        feedback = ""
    for handoff in waits:
        if (
            handoff.get("stage")
            not in {
                "awaiting_activation_approval",
                "awaiting_direction_approval",
                "execution_plan_ready",
                "execution_plan_approved",
            }
            or handoff.get("evidence_type") != "grouped_cross_validation_review"
        ):
            continue
        version = handoff["candidate_model_version"]
        candidate = state["candidate_models"][version]
        review = candidate.get("agent_review")
        if review and review.get("followup_result") and not feedback:
            handoff.update(
                stage="followup_completed", direction_status=review["followup_result"]["status"]
            )
            for job in state.get("remote_finetune_jobs", {}).values():
                if (job.get("training_handoff") or {}).get("candidate_model_version") == version:
                    job["training_handoff"] = deepcopy(handoff)
            continue
        revised = bool(
            review
            and (feedback and review.get("user_feedback") != feedback)
            and (revise_approved or handoff.get("direction_status") not in {"approved", "rejected"})
        )
        if revised:
            candidate.setdefault("superseded_agent_reviews", []).append(deepcopy(review))
            candidate.pop("agent_review", None)
            handoff.pop("direction_status", None)
            handoff.pop("direction_proposal", None)
            review = None
        if (
            review
            and review.get("choice") != "activate"
            and not isinstance(review.get("followup_action"), dict)
        ):
            review = None  # Older direction-only records need a concrete plan.
        if not review:
            review = {
                "choice": "activate",
                "review_policy": REVIEW_POLICY,
                "user_feedback": feedback,
                "reason": "训练成功回收，按已确认流程切换新模型，再刷新已有结构与凸包。",
                "alternatives": "DFT补采样与独立误差验收由后续搜索评估，不重复选择是否切换。",
                "followup_action": None,
            }
            candidate["agent_review"] = review
        proposal = {
            "direction": review["choice"],
            "review_policy": review.get("review_policy"),
            "user_feedback": review.get("user_feedback", ""),
            "candidate_model_version": version,
            "training_fingerprint": handoff.get("training_fingerprint"),
            "review_sha256": handoff.get("review_sha256"),
            "reason": review["reason"],
            "alternatives": deepcopy(review["alternatives"]),
            "plan": "switch_model"
            if review["choice"] == "activate"
            else deepcopy(review["followup_action"]),
        }
        handoff["direction_proposal"] = proposal
        handoff.setdefault("direction_status", "awaiting_approval")
        if state_path:
            from phase_agent.graphs.direction_review_graph import direction_checkpoint

            checkpoint = direction_checkpoint(state_path, proposal)
            handoff["direction_status"] = checkpoint.get("status", "awaiting_approval")
        if review["choice"] == "activate":
            handoff["stage"] = "awaiting_activation_approval"
            handoff["reason"] = (
                "Agent判断：采用新模型。"
                + review["reason"][:350]
                + "\n本次执行：切换模型。切换后展示已有结构与凸包刷新的具体范围、成本；该新增计算动作再审批。回复‘同意’执行，‘拒绝’取消。"
            )
        else:
            approved = handoff.get("direction_status") == "approved"
            rejected = handoff.get("direction_status") == "rejected"
            plan = review["followup_action"]
            plan_text = (
                "\n具体方案："
                + str(plan.get("expected_purpose") or plan.get("reason"))[:140]
                + f"；目标 {len(plan.get('target_ids') or [])} 个；相对成本 {plan.get('budget', 0)}。"
            )
            handoff.update(
                stage="direction_rejected"
                if rejected
                else "execution_plan_approved"
                if approved
                else "execution_plan_ready",
                reason="方向判断已完成。Agent建议："
                + review["reason"][:220]
                + plan_text
                + (
                    "；该动作已批准，复用原方案核对执行条件。"
                    if approved
                    else "；方向已拒绝，未执行。"
                    if rejected
                    else "；接下来核对并展示完整动作与成本，在执行方案处统一审批。"
                ),
            )
        for job in state.get("remote_finetune_jobs", {}).values():
            if (job.get("training_handoff") or {}).get("candidate_model_version") == version:
                job["training_handoff"] = deepcopy(handoff)
    return state, [row for row in waits if row.get("stage") != "followup_completed"]

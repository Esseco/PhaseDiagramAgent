"""Scientific choice precedes human execution approval."""

from copy import deepcopy
from phase_agent.tools.budget.record_budget_usage import record_budget_usage


def review_training_choice(state, waits, agent_client, *, state_path=None, user_message=None):
    feedback = str(user_message or "").strip()
    if feedback.lower() in {"继续", "继续原运行", "同意", "拒绝", "continue", "approve", "reject"}:
        feedback = ""
    for handoff in waits:
        if (
            handoff.get("stage")
            not in {"awaiting_activation_approval", "awaiting_direction_approval"}
            or handoff.get("evidence_type") != "grouped_cross_validation_review"
        ):
            continue
        version = handoff["candidate_model_version"]
        candidate = state["candidate_models"][version]
        review = candidate.get("agent_review")
        previous_review = deepcopy(review)
        revised = bool(
            feedback
            and review
            and review.get("user_feedback") != feedback
            and handoff.get("direction_status") not in {"approved", "rejected"}
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
            if not callable(agent_client):
                handoff.update(
                    stage="agent_review_required",
                    reason="微调结果已齐，Agent判断接口未配置；未激活。",
                )
                continue
            from phase_agent.analysis.state.build_decision_context import build_decision_context

            payload = {
                "mode": "training_model_review",
                "decision_kind": "model_update",
                "allowed_tools": ["adjust_strategy"],
                "state": {"active_model_version": state.get("active_model_version")},
                "decision_context": {
                    "training_already_completed": True,
                    "user_feedback": feedback,
                    "previous_review": previous_review,
                    "candidate_validation": candidate["validation"],
                    "search_evidence": build_decision_context(state),
                },
                "instruction": "由你作科学判断，比较换新模型、保留旧模型补DFT、核查接口或其他策略。禁止把选择交给用户。返回tool=adjust_strategy，parameters.choice=activate/supplement_dft/other，reason为简短科学理由，parameters.alternatives为其他路线暂不选的原因。K折不是最终模型独立测试；激活仅供下一轮刷新，之后判收敛和DFT缺口。不得执行或重训。",
            }
            if feedback:
                payload["user_instruction"] = feedback
                payload["instruction"] += (
                    " 当前用户反馈必须纳入重新判断；区分用户给定的解释与已核实证据。"
                    " 不得机械重复旧方向。说明此反馈如何改变判断，或具体说明仍有何证据冲突。"
                    " 用户纠正原因不等于批准任何动作。"
                )
            payload["output_contracts"] = {
                "training_model_review": {
                    "tool": "adjust_strategy",
                    "parameters": {
                        "choice": "activate | supplement_dft | other",
                        "alternatives": "简短说明其他路线为什么暂不选",
                    },
                    "reason": "简短科学理由",
                }
            }
            from phase_agent.decisions.agent.action_contracts import ActionEnvelope
            from phase_agent.decisions.agent.decision_contracts import PostDFTReview

            payload["output_contracts"]["followup_action"] = ActionEnvelope.model_json_schema()
            payload["output_contracts"]["post_dft_review"] = PostDFTReview.model_json_schema()
            require_round_review = bool(
                payload["decision_context"]["search_evidence"].get("post_dft_assessment")
            )
            payload["instruction"] += (
                " 严格按output_contracts返回完整JSON；choice和alternatives必须放在parameters中。"
                " 非activate方向必须同时给出parameters.followup_action完整动作，包含tool、parameters、target_ids、budget、reason、expected_purpose、evidence_refs。"
                " supplement_dft使用select_dft_candidates或必要配置修订adjust_strategy；other使用adjust_strategy。"
                " 候选必须来自提供的数据，无法形成DFT方案则明确提出配置或核查方案，不虚构任务。"
                " 若search_evidence.post_dft_assessment存在，followup_action必须带符合schema的post_dft_review；"
                " 说明微调已经完成，此处只做后续决策，不再次训练。方向和具体动作必须一致。"
            )
            for attempt in range(2):
                decision = agent_client(payload)
                if decision.get("_llm_usage"):
                    state = record_budget_usage(
                        state,
                        {
                            "llm_usage": decision["_llm_usage"],
                            "iteration": state.get("iteration", 0),
                        },
                    )
                    candidate = state["candidate_models"][version]
                parameters = decision.get("parameters") or {}
                choice = parameters.get("choice") or decision.get("choice")
                reason = str(decision.get("reason") or parameters.get("reason") or "").strip()
                alternatives = parameters.get("alternatives") or decision.get("alternatives")
                from phase_agent.tools.local.training_followup_plan import followup_errors

                followup = parameters.get("followup_action") or decision.get("followup_action")
                plan_errors = followup_errors(
                    choice, followup, require_round_review=require_round_review
                )
                if (
                    choice in {"activate", "supplement_dft", "other"}
                    and reason
                    and alternatives
                    and not plan_errors
                ):
                    break
                candidate["agent_review_diagnostic"] = {
                    "tool": decision.get("tool"),
                    "choice": choice,
                    "reason": reason,
                    "alternatives": deepcopy(alternatives),
                    "parameter_keys": list(parameters),
                    "validation_errors": deepcopy(plan_errors),
                    "followup_action": deepcopy(followup),
                }
                payload["validation_errors"] = [
                    "parameters.choice需为activate/supplement_dft/other；reason及parameters.alternatives不能为空",
                    *plan_errors,
                ]
                payload["invalid_action"] = deepcopy(decision)
                payload["repair_instruction"] = "保留你的科学判断，补齐上述格式缺项，返回完整JSON。"
            if (
                choice not in {"activate", "supplement_dft", "other"}
                or not reason
                or not alternatives
                or plan_errors
            ):
                handoff.update(
                    stage="agent_review_required",
                    reason="Agent未返回完整的模型选择和备选理由；未激活，请继续重试。",
                )
                continue
            review = {
                "choice": choice,
                "user_feedback": feedback,
                "reason": reason,
                "alternatives": deepcopy(alternatives),
                "followup_action": deepcopy(followup) if choice != "activate" else None,
            }
            candidate["agent_review"] = review
        proposal = {
            "direction": review["choice"],
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
        if state_path:
            from phase_agent.graphs.direction_review_graph import direction_checkpoint

            checkpoint = direction_checkpoint(state_path, proposal)
            handoff["direction_status"] = checkpoint.get("status", "awaiting_approval")
        if review["choice"] == "activate":
            handoff["reason"] = (
                "Agent判断：采用新模型。"
                + review["reason"][:350]
                + "\n执行方案：切换模型，随后准备已有结构与凸包刷新；刷新后判断收敛、补DFT或下一轮搜索。回复‘同意’执行，‘拒绝’取消。"
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
                else "validation_prerequisites_required"
                if approved
                else "awaiting_direction_approval",
                reason="方向判断已完成。Agent建议："
                + review["reason"][:220]
                + plan_text
                + (
                    "；方向已批准，准备具体执行方案，执行另行审批。"
                    if approved
                    else "；方向已拒绝，未执行。"
                    if rejected
                    else "\n回复‘同意’批准该方向及上述计划，随后核对执行条件；‘拒绝’取消。"
                ),
            )
        for job in state.get("remote_finetune_jobs", {}).values():
            if (job.get("training_handoff") or {}).get("candidate_model_version") == version:
                job["training_handoff"] = deepcopy(handoff)
    return state, waits

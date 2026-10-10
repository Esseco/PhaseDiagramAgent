"""Scientific choice precedes human execution approval."""

from copy import deepcopy

REVIEW_POLICY = "post_training_single_approval_v3"
from phase_agent.tools.budget.record_budget_usage import record_budget_usage


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
        previous_review = deepcopy(review)
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
            and (
                review.get("review_policy") != REVIEW_POLICY
                or (feedback and review.get("user_feedback") != feedback)
            )
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
                "instruction": "微调已完成。科学决策主要比较两条路线：1.activate：切换候选新模型，刷新已有结构与凸包，再进入下一轮搜索；2.supplement_dft：针对覆盖缺口或高误差区域补采样与DFT，回收新增数据后再微调。比较下一轮搜索的信息收益与新增DFT的边际收益、覆盖、误差分布和预算，推荐一条并解释另一条暂不选的原因。旧模型误差不能当作新模型误差；微调改善是用户工作假设，未测量的改善幅度不得虚构。若没有明确高收益补DFT目标，应说明切换并搜索是否更合理。不要把独立验证字段缺失自动当成这两条路线的阻塞，也不要仅凭预测力幅度小就诊断接口错误。K折不等于最终独立验证，但候选可按现有审批进入下一轮搜索，不代表已收敛。other仅限有直接证据的执行阻塞，须明确证据和解除条件。返回tool=adjust_strategy，parameters.choice=activate/supplement_dft/other，reason为简短收益比较，parameters.alternatives解释未选路线。此处只提出方案，不执行或重复本轮训练。",
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
                " 候选必须来自提供的数据，不能虚构候选；尚无可用目标时说明下一步怎样采样获得目标，不因无法列目标自动选择补验证配置。"
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
                "review_policy": REVIEW_POLICY,
                "user_feedback": feedback,
                "reason": reason,
                "alternatives": deepcopy(alternatives),
                "followup_action": deepcopy(followup) if choice != "activate" else None,
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

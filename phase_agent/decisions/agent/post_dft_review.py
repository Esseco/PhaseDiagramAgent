"""Require explicit LLM trade-offs, without imposing scientific error thresholds."""

REVIEW_TOOLS = {
    "generate_branches": "search",
    "update_mlip": "finetune",
    "adjust_strategy": "revise_strategy",
    "select_dft_candidates": "supplement_dft",
    "check_convergence": "convergence",
    "pause_search": "stop",
}


def normalize_review_choice(action):
    """Normalize equivalent wire labels, never infer a scientific recommendation."""
    from copy import deepcopy
    from phase_agent.tools.policy.training_input_action import is_training_input_action

    result = deepcopy(action)
    review = result.get("post_dft_review")
    if not isinstance(review, dict):
        return result
    old = review.get("choice")
    aliases = {**REVIEW_TOOLS, "fine_tune": "finetune", "fine-tune": "finetune"}
    choice = aliases.get(old, old) if isinstance(old, str) else old
    if (
        choice == "revise_strategy"
        and result.get("tool") == "update_mlip"
        and is_training_input_action(result)
        and review.get("finetune_recommendation") == "now"
        and review.get("stop_status") == "continue"
    ):
        choice = "finetune"
    if choice != old:
        review["choice"] = choice
        result["_review_choice_normalization"] = {"original": old, "normalized": choice}
    return result


def post_dft_review_errors(action, *, include_parameters=True):
    expected = REVIEW_TOOLS.get(action.get("tool") or action.get("action_type"))
    if expected is None:
        return []
    errors = []
    if expected == "supplement_dft" and include_parameters:
        from phase_agent.decisions.agent.dft_contracts import dft_contract_errors

        errors.extend(dft_contract_errors((action.get("parameters") or {}).get("decisions")))
    if expected == "search" and include_parameters:
        from phase_agent.decisions.agent.generation_plan import validate_generation_plan

        try:
            from phase_agent.tools.policy.region_sampling import (
                is_region,
                errors as region_errors,
            )

            if is_region(action):
                errors.extend(region_errors(action.get("parameters") or {}))
            else:
                validate_generation_plan(action.get("parameters") or {})
        except (ValueError, TypeError, AttributeError) as error:
            errors.append(f"generation_plan: {error}")
    review = action.get("post_dft_review")
    if not isinstance(review, dict):
        return errors + ["post_dft_review: 缺少分析对象"]
    from phase_agent.decisions.agent.decision_contracts import review_contract_errors

    errors.extend(review_contract_errors(review))
    if review.get("choice") != expected:
        errors.append(
            f"post_dft_review.choice: 当前动作需要 {expected}；收到 {review.get('choice')!r}。保持科学结论，返回与动作一致的choice；若结论矛盾则重新选择动作。"
        )
    recommendation = review.get("finetune_recommendation")
    stop = review.get("stop_status")
    if stop not in {"continue", "scientifically_converged", "budget_stop", "blocked"}:
        errors.append("stop_status: 需要continue/scientifically_converged/budget_stop/blocked")
    elif expected in {"search", "finetune", "supplement_dft"} and stop != "continue":
        errors.append("stop_status: 继续计算动作必须为continue")
    elif expected == "stop" and stop == "continue":
        errors.append("stop_status: 停止需说明科学收敛、预算停止或条件阻塞")
    if recommendation not in {"now", "defer", "insufficient_evidence"}:
        errors.append("finetune_recommendation: 需要 now/defer/insufficient_evidence")
    elif expected == "search" and recommendation != "defer":
        errors.append("finetune_recommendation: 搜索需明确defer；需微调或证据不足应修订策略")
    elif expected == "finetune" and recommendation != "now":
        errors.append("finetune_recommendation: 微调动作需要now")
    return errors


def valid_post_dft_review(action):
    return not post_dft_review_errors(action)


def request_review_repair(agent_client, payload, action):
    """Validate a saved response; errors are surfaced without another model call."""
    action = normalize_review_choice(action)
    errors = post_dft_review_errors(action)
    if not errors:
        return action
    from phase_agent.decisions.agent.proposal_validation import preserve_proposal_evidence

    error = ValueError("DFT 后方案无效，未重试：" + "; ".join(errors))
    preserve_proposal_evidence(error, action)
    raise error


def _unwrap_action_response(response):
    """Extract an action envelope without changing any model-selected fields."""
    if (
        isinstance(response, dict)
        and isinstance(response.get("action"), dict)
        and not (response.get("tool") or response.get("action_type"))
    ):
        action = dict(response["action"])
        action.update({key: value for key, value in response.items() if key.startswith("_")})
        return action
    return response


def _refresh_scientific_decision_brief(payload, fallback):
    """Prefer complete current-state evidence when a request carries it."""
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        return fallback
    if not state.get("confirmed_config_version") or not state.get("active_model_version"):
        return fallback

    from phase_agent.analysis.state.scientific_decision_brief import decision_brief

    refreshed = decision_brief(state)
    policy = refreshed.get("policy") if isinstance(refreshed, dict) else None
    if not (
        isinstance(refreshed, dict)
        and refreshed.get("config_version")
        and refreshed.get("model_version")
        and isinstance(policy, dict)
        and policy.get("schema_version") == 1
    ):
        return fallback
    return refreshed


def _repair_stale_scientific_decision(tracked_client, payload, action, brief, context):
    """Allow one model correction for a stale scientific version binding."""
    from copy import deepcopy

    from phase_agent.decisions.agent.proposal_validation import (
        preserve_proposal_evidence,
        proposal_schema_errors,
    )
    from phase_agent.decisions.agent.scientific_decisions import scientific_decision_errors

    errors = scientific_decision_errors(action, brief, context)
    stale_version_errors = {
        "scientific_decision_stale_config_version",
        "scientific_decision_stale_model_version",
    }
    stale_errors = [error for error in errors if error in stale_version_errors]
    if not stale_errors:
        return action

    original_tool = action.get("tool") or action.get("action_type")
    request = deepcopy(payload)
    request.update(
        mode="scientific_decision_repair",
        current_proposal=deepcopy(action),
        validation_errors=errors,
        scientific_decision_current_versions={
            "config_version": brief.get("config_version"),
            "model_version": brief.get("model_version"),
        },
        instruction=payload.get("instruction", "")
        + " Correct the stale scientific_decision version binding once. Use the exact current "
        "config_version and model_version supplied in scientific_decision_current_versions; "
        "re-evaluate the decision against the current brief instead of copying historical "
        "versions. Return one complete action, keep the selected tool unchanged, and do not "
        "execute or request approval.",
    )
    corrected = _unwrap_action_response(tracked_client(request))
    corrected_tool = (
        corrected.get("tool") or corrected.get("action_type")
        if isinstance(corrected, dict)
        else None
    )
    if corrected_tool != original_tool:
        raise ValueError("scientific_decision_repair_must_preserve_tool")

    schema_errors = proposal_schema_errors(corrected, context)
    if schema_errors:
        error = ValueError("Invalid scientific decision correction: " + "; ".join(schema_errors))
        preserve_proposal_evidence(error, corrected)
        raise error

    remaining = scientific_decision_errors(corrected, brief, context)
    if remaining:
        error = ValueError(
            "Invalid scientific decision after one correction: " + "; ".join(remaining)
        )
        preserve_proposal_evidence(error, corrected)
        raise error
    return corrected


def request_validated_action(agent_client, payload):
    context = payload.get("decision_context") or {}
    contracts = dict(payload.get("output_contracts") or {})
    from phase_agent.decisions.agent.action_contracts import ActionEnvelope

    contracts["action_common_fields"] = ActionEnvelope.model_json_schema()
    if payload.get("unified_dialogue"):
        from phase_agent.decisions.agent.dialogue_contract import (
            dialogue_output_contract,
        )

        contracts["dialogue_outcome"] = dialogue_output_contract(
            allow_controls=context.get("conversation_controls") is True
        )
        controls = (
            "回传/回收通知或要求推进需要先核对最新证据时，用kind=advance进入原科学流程，在本轮完成分析并提出下一步方案；不先询问是否读取或分析。明确批准/拒绝已展示方案用kind=review，填真实plan_id和decision=approve/reject，不重提新方案。"
            if context.get("conversation_controls")
            else "本轮已完成结果观察与核对；直接提出一项具体工具方案供审批，或说明真实等待条件。不要要求再发继续，不再使用advance/review。"
        )
        payload = {
            **payload,
            "instruction": "先理解用户目标。答疑用kind=answer；只读查看或返回设置用kind=inspect_configuration；修改持久设置用kind=configure。用户要求你选择或提出下一步科学方案时，返回工具契约供审批；提出方案本身不执行，也不需要用户先批准。用户要求等待审批不等于只要文字答疑。仅解释或讨论时才用answer，不要在answer正文中用自然语言代替可审核的动作对象。待审批方案不是执行授权，返回配置不等于修改或批准该方案。仅返回其中一种JSON结果。 "
            + controls
            + payload.get("instruction", ""),
        }
    if "allocate_mc_bohb" in payload.get("allowed_tools", []):
        from phase_agent.decisions.agent.mc_contracts import MCAllocationFields

        contracts["mc_allocation_common_fields"] = MCAllocationFields.model_json_schema()
    if "select_dft_candidates" in payload.get("allowed_tools", []):
        from phase_agent.decisions.agent.dft_contracts import DFTDecision

        contracts["dft_decision_item"] = DFTDecision.model_json_schema()
    if "generate_branches" in payload.get("allowed_tools", []):
        from phase_agent.decisions.agent.generation_contracts import (
            GenerationAllocation,
        )

        from phase_agent.tools.policy.region_sampling import VERSION, INSTRUCTION

        if (context.get("branch_count_policy") or {}).get("version") == VERSION:
            from phase_agent.decisions.agent.generation_contracts import GenerationRegion

            contracts["generation_regions"] = INSTRUCTION
            contracts["generation_region_item"] = GenerationRegion.model_json_schema()
        else:
            contracts["generation_plan_item"] = GenerationAllocation.model_json_schema()
        if context.get("branch_pool_review_required"):
            references = (context.get("evidence_catalog") or {}).get("references") or {}
            contracts["branch_review_evidence"] = {
                "references": {
                    key: {
                        name: row[name]
                        for name in ("kind", "source_ref", "method", "version", "model_version")
                        if name in row
                    }
                    for key, row in references.items()
                    if isinstance(row, dict)
                },
                "instruction": "For branch_pool_review, every group/default and generation region, use these exact keys in evidence_refs. JSON field paths and branch IDs are not reference keys. Read the corresponding facts before citing them; existence is not scientific adequacy.",
            }
    if contracts:
        payload = {**payload, "output_contracts": contracts}
    if context.get("post_dft_assessment"):
        from phase_agent.decisions.agent.decision_contracts import PostDFTReview

        payload = {
            **payload,
            "output_contracts": {
                **contracts,
                "post_dft_review": PostDFTReview.model_json_schema(),
            },
            "instruction": payload["instruction"] + " " + review_output_instruction(),
        }
    from phase_agent.analysis.state.training_context import training_review_active

    if context.get("post_refresh_search"):
        payload = {
            **payload,
            "instruction": payload.get("instruction", "")
            + " "
            + context["post_refresh_search"]["instruction"],
        }
    if training_review_active(context):
        payload = {
            **payload,
            "instruction": payload.get("instruction", "")
            + " 本轮训练已完成且结果回收，以上微调准备指引不适用。严禁update_mlip或重新训练；只选择allowed_tools内的动作。依据已记录的Agent模型审阅判断，若核查接口或修改验证条件，选择adjust_strategy并提出具体方案；若补DFT，选择select_dft_candidates。不得将旧模型误差当新模型误差，不得把缺少独立测试当自动禁用新模型的规则。",
        }
    from phase_agent.decisions.agent.round_budget_review import (
        round_budget_instruction,
        round_budget_review_errors,
    )

    if (context.get("round_budget_evidence") or {}).get("required"):
        payload = {
            **payload,
            "instruction": payload.get("instruction", "") + " " + round_budget_instruction(),
        }
    from phase_agent.decisions.agent.scientific_decisions import (
        attach_decision_contract,
    )

    payload = attach_decision_contract(payload)
    usages = []

    def tracked_client(request):
        try:
            result = agent_client(request)
        except Exception as error:
            if getattr(error, "llm_usage", None):
                usages.append(error.llm_usage)
            raise
        usage = result.get("_llm_usage") if isinstance(result, dict) else None
        if usage:
            usages.append(usage)
        return result

    def combined_usage():
        return {
            **usages[-1],
            **{
                key: sum(u.get(key) or 0 for u in usages)
                for key in ("calls", "input_tokens", "output_tokens")
            },
            "cost": sum(u["cost"] for u in usages)
            if all(u.get("cost") is not None for u in usages)
            else None,
        }

    try:
        action = tracked_client(payload)
        action = _unwrap_action_response(action)
        from phase_agent.decisions.agent.proposal_schema_repair import repair_proposal_schema

        action = repair_proposal_schema(tracked_client, payload, action)
        from phase_agent.decisions.agent.generation_plan import derive_generation_totals

        action = derive_generation_totals(action)
        from phase_agent.decisions.agent.branch_review_repair import repair_branch_review

        action = repair_branch_review(tracked_client, payload, action)

        from phase_agent.decisions.agent.dialogue_contract import (
            is_dialogue,
            dialogue_errors,
        )

        if is_dialogue(action):
            from phase_agent.decisions.agent.dialogue_contract import (
                normalize_dialogue,
                repair_read_only_dialogue,
            )

            action = normalize_dialogue(action)
            action = repair_read_only_dialogue(tracked_client, payload, action)
            errors = dialogue_errors(action, payload.get("unified_dialogue") is True)
            if errors:
                error = ValueError("; ".join(errors))
                if usages:
                    error.llm_usage = combined_usage()
                raise error
            if usages:
                action["_llm_usage"] = combined_usage()
            return action
        if context.get("post_dft_assessment") and isinstance(action, dict):
            try:
                action = request_review_repair(tracked_client, payload, action)
            except Exception as error:
                if usages:
                    error.llm_usage = combined_usage()
                raise
        budget_errors = (
            round_budget_review_errors(action, context) if isinstance(action, dict) else []
        )
        if budget_errors:
            raise ValueError("Invalid budget trade-off: " + "; ".join(budget_errors))
        from phase_agent.decisions.agent.scientific_decisions import (
            scientific_decision_errors,
        )

        brief = context.get("scientific_decision_brief") or {}
        refreshed_brief = _refresh_scientific_decision_brief(payload, brief)
        if refreshed_brief is not brief:
            context = {**context, "scientific_decision_brief": refreshed_brief}
            payload = {**payload, "decision_context": context}
            brief = refreshed_brief
        if (brief.get("policy") or {}).get("schema_version") == 1:
            errors = scientific_decision_errors(action, brief, context)
            if errors:
                repaired = _repair_stale_scientific_decision(
                    tracked_client, payload, action, brief, context
                )
                if repaired is not action:
                    action = derive_generation_totals(repaired)
                    from phase_agent.decisions.agent.proposal_validation import (
                        proposal_schema_errors,
                    )

                    schema_errors = proposal_schema_errors(action, context)
                    if schema_errors:
                        raise ValueError(
                            "Invalid scientific decision correction: " + "; ".join(schema_errors)
                        )
                    budget_errors = round_budget_review_errors(action, context)
                    if budget_errors:
                        raise ValueError(
                            "Invalid budget trade-off after scientific correction: "
                            + "; ".join(budget_errors)
                        )
                    if context.get("post_dft_assessment"):
                        review_errors = post_dft_review_errors(action)
                        if review_errors:
                            raise ValueError(
                                "Invalid post-DFT review after scientific correction: "
                                + "; ".join(review_errors)
                            )
                    errors = scientific_decision_errors(action, brief, context)
            if errors:
                raise ValueError("Invalid scientific decision: " + "; ".join(errors))
            if context.get("post_dft_assessment") and post_dft_review_errors(action):
                raise ValueError("Scientific decision repair must preserve post_dft_review")
            if round_budget_review_errors(action, context):
                raise ValueError("Scientific decision repair must preserve budget trade-off")
        action = derive_generation_totals(action)
        if usages and isinstance(action, dict):
            action["_llm_usage"] = combined_usage()
        return action

    except Exception as error:
        if usages:
            error.llm_usage = combined_usage()
        raise


def review_output_instruction():
    return (
        "When post_dft_assessment exists, compare FOUR alternatives before choosing a tool: "
        "For input-only update_mlip the review choice MUST be finetune (not revise_strategy or update_mlip); finetune_recommendation=now and stop_status=continue. "
        "finetune now, supplement DFT from the existing pool, generate new branches, and "
        "convergence/stopping. Include top-level post_dft_review: choice "
        "(generate_branches=search, update_mlip=finetune, adjust_strategy=revise_strategy, "
        "select_dft_candidates=supplement_dft, check_convergence=convergence, pause_search=stop), error_assessment, "
        "coverage_assessment, finetune_assessment, search_assessment, reference_assessment, "
        "round_findings, limitations, dft_assessment, convergence_assessment, "
        "stop_status (continue/scientifically_converged/budget_stop/blocked), "
        "finetune_recommendation (now, defer, insufficient_evidence). "
        "dft_assessment explains whether existing qualified data are representative enough to "
        "finetune or require more DFT; new branches may be preferred whenever their expected marginal value exceeds supplementing the existing pool. Coverage and error are evidence, not hard gates. "
        "convergence_assessment evaluates hull stability, phase/composition coverage, recent gains, "
        "model errors and budget; lack of new stable phases alone does not prove convergence. "
        "Distinguish scientific convergence from resource-limited stopping. "
        "Human experience treats excessive force error as an important reason to finetune; "
        "interface checking may accompany the recommendation, not automatically delay it. "
        "Decide scientific need for finetuning FIRST, independently of enablement. "
        "finetune_assessment must directly recommend finetuning or deferral with scientific "
        "reasons grounded in energy/force errors, representativeness and intended use; "
        "never substitute 'not enabled, needs assessment' for a recommendation. "
        "If finetuning is scientifically needed and data adequate, choose update_mlip with parameters.prepare_inputs_only=true and parameters.action=RETRAIN_MLIP, regardless of mlip_finetune.enabled. Recommend finetuning with scientific reasons, then request approval to generate HPC submission inputs (committee, K-fold evaluation and full-data main model), not to enable configuration. Approval only prepares files, never runs local training, submits jobs or activates models. Use adjust_strategy only for other genuine configuration revisions. "
        "If data inadequate, compare supplementing existing candidates with targeted new search and downstream DFT; select the higher expected value path. "
        "Search requires defer justified by scientific evidence; insufficient evidence requires "
        "a strategy revision rather than assuming search is preferable. "
        "round_findings summarizes stable-phase findings "
        "and search gains of this round; limitations describes missing returns, spin "
        "rejections and uncertainties. Explicitly acknowledge unavailable evidence "
        "instead of inventing gains or stable-phase changes. Each assessment "
        "is a concise nonempty string under 240 characters. Explain energy AND force "
        "errors, coverage/memory evidence, why choose or defer finetuning, and why "
        "choose or defer new search. Disabled finetuning is not scientific justification "
        "for search. reference_assessment summarizes relevant memory, configured criteria "
        "and useful recommendations or explicitly says evidence is missing. "
        "Do not invent thresholds or missing evidence."
    )

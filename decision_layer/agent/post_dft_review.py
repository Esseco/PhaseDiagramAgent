"""Require explicit LLM trade-offs, without imposing scientific error thresholds."""

REVIEW_TOOLS = {"generate_branches": "search", "update_mlip": "finetune",
                "adjust_strategy": "revise_strategy", "select_dft_candidates": "supplement_dft",
                "check_convergence": "convergence", "pause_search": "stop"}


def normalize_review_choice(action):
    """Normalize equivalent wire labels, never infer a scientific recommendation."""
    from copy import deepcopy
    from execution_layer.policy.training_input_action import is_training_input_action
    result = deepcopy(action)
    review = result.get("post_dft_review")
    if not isinstance(review, dict):
        return result
    old = review.get("choice")
    aliases = {**REVIEW_TOOLS, "fine_tune": "finetune", "fine-tune": "finetune"}
    choice = aliases.get(old, old) if isinstance(old, str) else old
    if (choice == "revise_strategy" and result.get("tool") == "update_mlip"
            and is_training_input_action(result)
            and review.get("finetune_recommendation") == "now"
            and review.get("stop_status") == "continue"):
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
        from decision_layer.agent.dft_contracts import dft_contract_errors
        errors.extend(dft_contract_errors((action.get("parameters") or {}).get("decisions")))
    if expected == "search" and include_parameters:
        from decision_layer.agent.generation_plan import validate_generation_plan
        try:
            validate_generation_plan(action.get("parameters") or {})
        except (ValueError, TypeError, AttributeError) as error:
            errors.append(f"generation_plan: {error}")
    review = action.get("post_dft_review")
    if not isinstance(review, dict):
        return errors + ["post_dft_review: 缺少分析对象"]
    from decision_layer.agent.decision_contracts import review_contract_errors
    errors.extend(review_contract_errors(review))
    if review.get("choice") != expected:
        errors.append(f"post_dft_review.choice: 当前动作需要 {expected}；收到 {review.get('choice')!r}。保持科学结论，返回与动作一致的choice；若结论矛盾则重新选择动作。")
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
    """One bounded model correction; never fill scientific conclusions in code."""
    action = normalize_review_choice(action)
    errors = post_dft_review_errors(action)
    if not errors:
        return action
    repaired = agent_client({**payload, "mode": "repair_post_dft_action",
        "invalid_action": action, "validation_errors": errors,
        "instruction": payload["instruction"] + " Correct the listed validation errors once. "
        "Return the complete action JSON, not a patch. Reconsider the scientific choice using "
        "the same evidence; do not invent data or bypass analysis by choosing pause_search."})
    if not isinstance(repaired, dict):
        raise ValueError("DFT 后方案修正失败：返回值不是对象；原问题：" + "; ".join(errors))
    repaired = normalize_review_choice(repaired)
    tool = repaired.get("tool") or repaired.get("action_type")
    remaining = post_dft_review_errors(repaired)
    if tool not in payload["allowed_tools"] or tool not in REVIEW_TOOLS:
        remaining.append("修正必须返回已注册的微调/搜索/修订策略动作")
    if remaining:
        error = ValueError("DFT 后方案修正失败：" + "; ".join(remaining))
        for candidate in (repaired, action):
            if not post_dft_review_errors(candidate, include_parameters=False):
                error.post_dft_review = candidate["post_dft_review"]
                break
        raise error
    return repaired


def request_validated_action(agent_client, payload):
    context = payload.get("decision_context") or {}
    contracts = dict(payload.get("output_contracts") or {})
    from decision_layer.agent.action_contracts import ActionEnvelope
    contracts["action_common_fields"] = ActionEnvelope.model_json_schema()
    if "allocate_mc_bohb" in payload.get("allowed_tools", []):
        from decision_layer.agent.mc_contracts import MCAllocationFields
        contracts["mc_allocation_common_fields"] = MCAllocationFields.model_json_schema()
    if "select_dft_candidates" in payload.get("allowed_tools", []):
        from decision_layer.agent.dft_contracts import DFTDecision
        contracts["dft_decision_item"] = DFTDecision.model_json_schema()
    if "generate_branches" in payload.get("allowed_tools", []):
        from decision_layer.agent.generation_contracts import GenerationAllocation
        contracts["generation_plan_item"] = GenerationAllocation.model_json_schema()
    if contracts:
        payload = {**payload, "output_contracts": contracts}
    if context.get("post_dft_assessment"):
        from decision_layer.agent.decision_contracts import PostDFTReview
        payload = {**payload, "output_contracts": {**contracts, "post_dft_review": PostDFTReview.model_json_schema()},
                   "instruction": payload["instruction"] + " " + review_output_instruction()}
    from decision_layer.agent.round_budget_review import round_budget_instruction, round_budget_review_errors
    if (context.get("round_budget_evidence") or {}).get("required"):
        payload = {**payload, "instruction": payload.get("instruction", "") + " " + round_budget_instruction()}
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
        return {**usages[-1], **{key: sum(u.get(key) or 0 for u in usages)
                for key in ("calls", "input_tokens", "output_tokens")},
                "cost": sum(u["cost"] for u in usages) if all(u.get("cost") is not None for u in usages) else None}
    action = tracked_client(payload)
    if isinstance(action, dict) and isinstance(action.get("action"), dict) and not (action.get("tool") or action.get("action_type")):
        envelope = action
        action = dict(envelope["action"])
        if envelope.get("_llm_usage"):
            action["_llm_usage"] = envelope["_llm_usage"]
    if context.get("post_dft_assessment") and isinstance(action, dict):
        try:
            action = request_review_repair(tracked_client, payload, action)
        except Exception as error:
            if usages:
                error.llm_usage = combined_usage()
            raise
    budget_errors = round_budget_review_errors(action, context) if isinstance(action, dict) else []
    if budget_errors:
        action = tracked_client({**payload, "mode": "round_budget_review_repair",
            "invalid_action": action, "validation_errors": budget_errors,
            "instruction": payload.get("instruction", "") + " Repair missing trade-off; return one complete action."})
        errors = round_budget_review_errors(action, context) if isinstance(action, dict) else ["action must be object"]
        if errors:
            raise ValueError("Invalid budget trade-off: " + "; ".join(errors))
        if context.get("post_dft_assessment") and post_dft_review_errors(action):
            raise ValueError("Budget repair must preserve valid post_dft_review")
    if usages and isinstance(action, dict):
        action["_llm_usage"] = combined_usage()
    return action


def review_output_instruction():
    return ("When post_dft_assessment exists, compare FOUR alternatives before choosing a tool: "
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
            "Do not invent thresholds or missing evidence.")

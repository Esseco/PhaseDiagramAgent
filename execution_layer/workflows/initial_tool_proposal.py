"""Initial scientific proposal selection and bounded preparation; no execution."""
from copy import deepcopy
from execution_layer.state.state_manager import agent_state_summary
from execution_layer.budget.record_budget_usage import record_budget_usage
from execution_layer.policy.execution_policy import build_agent_proposal
from decision_layer.agent.post_dft_review import valid_post_dft_review


def select_initial_proposal(*, current, mode, config, context, decision_state,
                            registry, agent_client, invocation_id, propose, revise,
                            prepare_debug, is_verified_second, mc_block,
                            model_failed, record_id_factory):
    allowed = [name for name in (config.get("agent") or {}).get("allowed_tools") or []
               if callable((registry.get(name) or {}).get("handler"))]
    from decision_layer.agent.choose_debug_next_action import choose_debug_next_action
    mc_files_pending = any(row.get("stage") == "deep_search" and row.get("status") == "pending"
                           and not row.get("slurm_batch_id") for row in current.get("tasks") or [])
    has_mc_tasks = any(row.get("stage") == "deep_search" for row in current.get("tasks") or [])
    continue_requested = str((context or {}).get("user_message") or "").strip().lower() in {
        "继续", "下一步", "然后呢", "continue", "next",
    }
    from scientific_layer.mc.second_round_state import second_round_completed
    effective = (context or {}).get("effective_config") or config
    model = effective.get("mlip") or {}
    version = model.get("version") or model.get("name") or current.get("active_model_version")
    from analysis_layer.state.post_dft_assessment import post_dft_assessment
    assessment = post_dft_assessment(current, effective)
    decision_state.setdefault("decision_context", {})["post_dft_assessment"] = assessment
    if assessment and (context or {}).get("manager") is not None:
        from analysis_layer.cost.generation_preflight import generation_framework_costs
        try:
            costs = generation_framework_costs(context["manager"], context.get("phase_references") or {}, effective, current)
            decision_state["decision_context"]["generation_framework_costs"] = [
                {key: row[key] for key in ("phase", "det_H", "atom_count_upper", "stage_costs", "mc_steps_upper")}
                for row in costs[:64]]
        except ValueError as error:
            decision_state["decision_context"]["generation_cost_warning"] = str(error)
    mc_ids = sorted(str(row.get("task_id")) for row in current.get("tasks") or []
                    if row.get("stage") == "deep_search" and row.get("model_version") == version)
    consumed_mc = (current.get("post_dft_decided_mc_tasks") or {}).get(version)
    post_mc = second_round_completed(current, version) and not assessment and mc_ids != consumed_mc
    if post_mc or assessment:
        from scientific_layer.qbc.post_mc_candidates import post_mc_candidates
        current["qbc_candidates"] = post_mc_candidates(current, version)
        if assessment:
            previous_dft = {t.get("structure_id") for t in current.get("tasks") or []
                if t.get("stage") in {"dft_single_point", "dft_relax"} and t.get("status") != "cancelled"}
            current["qbc_candidates"] = [r for r in current["qbc_candidates"]
                                          if r.get("candidate_id") not in previous_dft]
        decision_state["qbc_candidates"] = deepcopy(current["qbc_candidates"])
        decision_state.setdefault("decision_context", {})["qbc_candidates"] = deepcopy(current["qbc_candidates"])
        from decision_layer.qbc_selection.recommend_post_mc_dft import recommend_post_mc_dft
        recommendation = recommend_post_mc_dft(current["qbc_candidates"], effective, current)
        decision_state["decision_context"]["dft_sampling_recommendation"] = {
            "candidate_ids": [row["candidate_id"] for row in recommendation["selected_candidates"]],
            "summary": recommendation["summary"],
            "instruction": "此为可选参考而非最终决定；结合记忆与现状自行选择。兼顾near-hull、相、Na、已知QBC与抽查，缺失QBC不评分；偏离覆盖目标说明理由。"}
        from execution_layer.budget.estimate_stage_cost import estimate_dft_cost
        decision_state["decision_context"]["dft_candidate_costs"] = [{"candidate_id": row["candidate_id"],
            "single_point_cost": estimate_dft_cost("DFT_SINGLE_POINT", row, effective["qbc"]),
            "relax_cost": estimate_dft_cost("DFT_RELAX", row, effective["qbc"])} for row in current["qbc_candidates"]]
        decision_state["decision_context"]["dft_stage_limits"] = (effective.get("budgets") or {}).get("stage_limits")
        if not current["qbc_candidates"] and not assessment:
            return {"status": "not_configured", "state": current,
                    "reason": "第二轮 MC 已完成；DFT 候选缺少当前相图中唯一匹配的已识别结构与 Ehull/atom。未追加 MC 或回退 Relax。"}
    safe_next = (choose_debug_next_action(
        current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
        allowed_tools=allowed, user_message=(context or {}).get("user_message"),
    ) if mode == "interactive" and (mc_files_pending or not callable(agent_client) or
         continue_requested) else None)
    if assessment:
        safe_next = None
        allowed = [tool for tool in allowed if tool in {
            "generate_branches", "update_mlip", "select_dft_candidates", "check_convergence", "pause_search", "adjust_strategy"}]
    if post_mc:
        safe_next = None
        allowed = [tool for tool in allowed if tool not in {
            "allocate_mc_bohb", "generate_branches", "run_calculation_stage", "prepare_local_batch_files"}]
    if mode == "interactive" and has_mc_tasks and not post_mc and not assessment and (mc_files_pending or continue_requested) and not safe_next:
        return mc_block(current, mode,
            (context or {}).get("effective_config") or config)
    action = safe_next or propose(
        decision_state, agent_client=agent_client, allowed_tools=allowed, config=config
    )
    if assessment and not model_failed(action):
        if not valid_post_dft_review(action):
            return {"status": "not_configured", "state": current,
                    "reason": "DFT 后方案缺少误差、覆盖和微调/搜索取舍分析；未生成 branch 或训练。请继续重新评估。"}
        action["_post_dft_review_version"] = 5
        action["_post_dft_scope_key"] = assessment["scope_key"]
    if model_failed(action):
        if action.get("_llm_usage"):
            current = record_budget_usage(current, {"llm_usage": action["_llm_usage"],
                                                   "iteration": current.get("iteration", 0)})
        return {"status": "not_configured", "state": current,
                "post_dft_review": action.get("failed_post_dft_review"),
                "reason": "模型未能返回完整有效方案，未暂停搜索或执行任务。请继续重试。原因：" + str(action.get("fallback_reason"))}
    if mode == "interactive" and not post_mc and not assessment and action.get("decision_source") == "rule":
        recovery = choose_debug_next_action(
            current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
            allowed_tools=allowed, user_message=(context or {}).get("user_message"),
        )
        if recovery:
            recovery["_llm_usage"] = action.get("_llm_usage")
            recovery["fallback_reason"] = action.get("fallback_reason")
            action = recovery
    if (mode == "interactive" and has_mc_tasks
            and action.get("tool") == "allocate_mc_bohb"
            and not is_verified_second(action, current,
                (context or {}).get("effective_config") or config)):
        safe_next = choose_debug_next_action(
            current, (context or {}).get("manager"), (context or {}).get("effective_config") or config,
            allowed_tools=allowed, user_message="继续",
        )
        if safe_next and safe_next.get("parameters", {}).get("round_kind") == "second":
            action = safe_next
        else:
            return mc_block(current, mode,
                (context or {}).get("effective_config") or config)
    if mode == "interactive":
        original_usage = action.get("_llm_usage")
        original_evidence = action.get("evidence_refs")
        action = prepare_debug(
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
    if mode == "interactive" and action.get("tool") == "select_dft_candidates":
        from execution_layer.workflows.prepare_dft_proposal import prepare_dft_proposal
        action, current, error = prepare_dft_proposal(
            action, current, decision_state, (context or {}).get("effective_config") or config,
            agent_client=agent_client, revise=revise)
        if error:
            return {"status": "rejected", "state": current, "reason": error,
                    "post_dft_review": action.get("post_dft_review")}
    from analysis_layer.cost.generation_preflight import attach_generation_preflight
    try:
        action = attach_generation_preflight(action, current, context,
            (context or {}).get("effective_config") or config)
    except ValueError as error:
        return {"status": "not_configured", "state": current, "reason": str(error),
                "post_dft_review": action.get("post_dft_review")}
    proposal = build_agent_proposal(action, decision_state, runtime_state=current)
    record_id = record_id_factory(current, invocation_id)
    return {"_proposal_selected": True, "current": current, "decision_state": decision_state,
            "proposal": proposal, "record_id": record_id}

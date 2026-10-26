"""Initial scientific proposal selection and bounded preparation; no execution."""

from copy import deepcopy
from phase_agent.tools.budget.record_budget_usage import record_budget_usage
from phase_agent.tools.policy.execution_policy import build_agent_proposal
from phase_agent.decisions.agent.post_dft_review import valid_post_dft_review


def _post_mc_allowed_tools(allowed, generation, *, post_mc):
    """Apply the post-MC route gate without hiding a new model's first search."""
    tools = list(allowed)
    if not post_mc:
        return tools
    pending_post_refresh_search = (
        generation.get("post_refresh")
        and generation.get("new_branch_count", 0) > 0
        and not generation.get("first_search_completed")
    )
    blocked = {"allocate_mc_bohb", "generate_branches", "run_calculation_stage"}
    if not pending_post_refresh_search:
        blocked.add("prepare_local_batch_files")
    return [
        tool
        for tool in tools
        if tool not in blocked
    ]


def select_initial_proposal(
    *,
    current,
    mode,
    config,
    context,
    decision_state,
    registry,
    agent_client,
    invocation_id,
    propose,
    revise,
    prepare_debug,
    is_verified_second,
    mc_block,
    model_failed,
    record_id_factory,
):
    allowed = [
        name
        for name in (config.get("agent") or {}).get("allowed_tools") or []
        if callable((registry.get(name) or {}).get("handler"))
    ]
    post_recovery_only = bool(
        current.get("post_recovery_next_step_pending")
        or current.get("_recovery_just_approved")
    )
    if post_recovery_only:
        # This is the final proposal boundary after result recovery.  The
        # dialogue layer may enter the scientific workflow with `advance`, so
        # enforce the phase transition again where the action is actually
        # selected.  Relax/MC preparation must wait for a separate DFT/branch
        # proposal approval.
        allowed = [
            tool for tool in allowed if tool in {"select_dft_candidates", "generate_branches"}
        ]
        if not allowed:
            return {
                "status": "not_configured",
                "state": current,
                "reason": "回收后缺少可用的 DFT 或 branch 动作；未生成 Relax/MC 方案。",
                "submitted": False,
            }
    if (
        mode == "interactive"
        and "select_dft_candidates" in allowed
        and (
            (context or {}).get("frozen_selection_only") is True
            or str((context or {}).get("user_message") or "").strip().lower()
            in {"继续", "下一步", "然后呢", "continue", "next"}
        )
    ):
        from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection

        saved = reusable_selection(current)
        if saved is not None:
            if saved.pop("reuse_requires_preview", False):
                from phase_agent.tools.workflows.prepare_dft_proposal import prepare_dft_proposal

                preparation = prepare_dft_proposal(
                    saved["raw_action"],
                    current,
                    decision_state,
                    (context or {}).get("effective_config") or config,
                    agent_client=None,
                    revise=revise,
                )
                if preparation.status == "rejected":
                    return {
                        "status": "rejected",
                        "state": current,
                        "reason": "原 DFT 选点仍未通过当前预检；原方案保留，请修订后再批准："
                        + preparation.error,
                        "submitted": False,
                    }
                saved = {
                    **build_agent_proposal(
                        preparation.action, decision_state, runtime_state=current
                    ),
                    "reuse_source_record_id": saved["reuse_source_record_id"],
                }
            return {
                "_proposal_selected": True,
                "current": current,
                "decision_state": decision_state,
                "proposal": saved,
                "record_id": record_id_factory(current, invocation_id),
            }
    if (context or {}).get("frozen_selection_only"):
        return {
            "status": "saved_selection_changed",
            "state": current,
            "reason": "原 DFT 选点的复用条件已变化；本次停止，未调用模型重新决策或执行任务。",
            "submitted": False,
        }
    from phase_agent.tools.policy.branch_conditions import branch_facts
    from phase_agent.runtime.turn_process import process_event

    facts = branch_facts(current)
    generation = facts["new_model_generation"]
    generation_required = generation["required"] and not generation["ready"]
    decision_state.setdefault("decision_context", {})["branch_conditions"] = facts
    # Saved snapshots can predate a code update without any scientific change.
    # Refresh current execution facts at the actual model boundary as well.
    from phase_agent.analysis.state.workflow_blockers import (
        mc_allocation_facts,
        refresh_result_facts,
    )

    decision_state["decision_context"]["mc_entry_conditions"] = mc_allocation_facts(current)
    decision_state["decision_context"]["refresh_result_facts"] = refresh_result_facts(current)
    from phase_agent.analysis.state.workflow_blockers import convergence_feedback

    decision_state["decision_context"]["latest_convergence_check"] = convergence_feedback(current)
    process_event("科学分支进入条件", facts)
    unified = bool((context or {}).get("unified_dialogue"))
    model_owned = callable(agent_client)
    if unified:
        decision_state.setdefault("decision_context", {})["unified_dialogue"] = True
        # This project dispatches HPC inputs, never local scientific calculations.
        allowed = [name for name in allowed if name != "run_calculation_stage"]
    from phase_agent.decisions.agent.choose_debug_next_action import (
        choose_debug_next_action,
    )

    mc_files_pending = any(
        row.get("stage") == "deep_search"
        and row.get("status") == "pending"
        and not row.get("slurm_batch_id")
        for row in current.get("tasks") or []
    )
    has_mc_tasks = any(row.get("stage") == "deep_search" for row in current.get("tasks") or [])
    continue_requested = str((context or {}).get("user_message") or "").strip().lower() in {
        "继续",
        "下一步",
        "然后呢",
        "continue",
        "next",
    }
    from phase_agent.science.mc.second_round_state import second_round_completed

    effective = (context or {}).get("effective_config") or config
    model = effective.get("mlip") or {}
    version = model.get("version") or model.get("name") or current.get("active_model_version")
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment

    assessment = post_dft_assessment(current, effective)
    decision_state.setdefault("decision_context", {})["post_dft_assessment"] = assessment
    if assessment and (context or {}).get("manager") is not None:
        from phase_agent.analysis.cost.generation_preflight import (
            generation_framework_costs,
        )

        try:
            costs = generation_framework_costs(
                context["manager"],
                context.get("phase_references") or {},
                effective,
                current,
            )
            decision_state["decision_context"]["generation_framework_costs"] = [
                {
                    key: row[key]
                    for key in (
                        "phase",
                        "det_H",
                        "atom_count_upper",
                        "stage_costs",
                        "mc_steps_upper",
                    )
                }
                for row in costs[:64]
            ]
        except ValueError as error:
            decision_state["decision_context"]["generation_cost_warning"] = str(error)
    mc_ids = sorted(
        str(row.get("task_id"))
        for row in current.get("tasks") or []
        if row.get("stage") == "deep_search" and row.get("model_version") == version
    )
    consumed_mc = (current.get("post_dft_decided_mc_tasks") or {}).get(version)
    post_mc = (
        second_round_completed(current, version)
        and not assessment
        and mc_ids != consumed_mc
        and not generation_required
    )
    if post_mc or assessment:
        from phase_agent.science.qbc.post_mc_candidates import post_mc_candidates

        current["qbc_candidates"] = post_mc_candidates(current, version)
        if assessment:
            previous_dft = {
                t.get("structure_id")
                for t in current.get("tasks") or []
                if t.get("stage") in {"dft_single_point", "dft_relax"}
                and t.get("status") != "cancelled"
            }
            prior_relax = {
                t.get("structure_id")
                for t in current.get("tasks") or []
                if t.get("stage") == "dft_relax" and t.get("status") != "cancelled"
            }
            current["qbc_candidates"] = [
                r
                for r in current["qbc_candidates"]
                if r.get("candidate_id") not in previous_dft
                or (
                    r.get("geometry_optimization_evidence")
                    and r.get("candidate_id") not in prior_relax
                )
            ]
        if not current["qbc_candidates"] and not assessment:
            return {
                "status": "not_configured",
                "state": current,
                "reason": "第二轮 MC 已完成；DFT 候选缺少当前相图中唯一匹配的已识别结构与 Ehull/atom。未追加 MC 或回退 Relax。",
            }
    if "select_dft_candidates" in allowed and "qbc_candidates" in current:
        from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence

        refresh_dft_evidence(current, decision_state, effective)
    safe_next = (
        choose_debug_next_action(
            current,
            (context or {}).get("manager"),
            (context or {}).get("effective_config") or config,
            allowed_tools=allowed,
            user_message=(context or {}).get("user_message"),
        )
        if not unified
        and not model_owned
        and mode == "interactive"
        and (mc_files_pending or not callable(agent_client) or continue_requested)
        else None
    )
    if assessment:
        safe_next = None
        allowed = [
            tool
            for tool in allowed
            if tool
            in {
                "generate_branches",
                "update_mlip",
                "select_dft_candidates",
                "check_convergence",
                "pause_search",
                "adjust_strategy",
            }
        ]
    # A completed MC round normally moves the next proposal to DFT/convergence
    # review.  Keep Relax preparation available while a refreshed model still
    # has an unsearched first-generation branch set.
    gated_tools = _post_mc_allowed_tools(allowed, generation, post_mc=post_mc)
    if gated_tools != allowed:
        safe_next = None
        allowed = gated_tools
    from phase_agent.tools.state.approved_direction import (
        action_direction,
        direction_hash,
        TOOLS,
    )

    direction = action_direction(current)
    if direction:
        safe_next = None
        allowed = [tool for tool in allowed if tool in TOOLS[direction["direction"]]]
        decision_state.setdefault("decision_context", {})["approved_direction"] = {
            **deepcopy(direction),
            "instruction": "复用已保存的方向及具体动作，禁止重新选择方向或生成同一动作的新版本。尚未批准时只展示完整执行范围及成本；一次具体动作审批同时确认方向。",
        }
    if (
        not unified
        and not model_owned
        and mode == "interactive"
        and has_mc_tasks
        and not post_mc
        and not assessment
        and not direction
        and (mc_files_pending or continue_requested)
        and not safe_next
    ):
        return mc_block(current, mode, (context or {}).get("effective_config") or config)
    from phase_agent.tools.state.training_pending_validation import (
        training_pending_validation,
    )

    if training_pending_validation(current):
        allowed = [tool for tool in allowed if tool != "update_mlip"]
        if safe_next and safe_next.get("tool") == "update_mlip":
            safe_next = None
        decision_state.setdefault("decision_context", {})["training_already_completed"] = {
            "instruction": "本轮微调已完成且结果回收，不重复相同数据的训练。应比较切换候选模型后刷新并搜索，与针对明确覆盖缺口补采样/DFT、回收新增数据后再微调的收益；不要默认转向独立验证配置修订。"
        }
    from phase_agent.tools.local.training_followup_plan import saved_followup

    if (context or {}).get("manager") is not None:
        from phase_agent.tools.policy.region_sampling import ledger_context

        decision_state.setdefault("decision_context", {})["branch_ledger_facts"] = ledger_context(
            context["manager"]
        )
        from phase_agent.decisions.agent.branch_pool_review import pool_context

        decision_state["decision_context"]["branch_pool_facts"] = pool_context(
            current, context["manager"].data["branches"]
        )
        from phase_agent.tools.relax_mc_batch.plan import registration_facts

        decision_state["decision_context"]["relax_mc_entry_conditions"] = registration_facts(
            current, context["manager"], effective
        )
        if current.get("branch_region_review"):
            decision_state["decision_context"]["branch_region_review"] = deepcopy(
                current["branch_region_review"]
            )

    from phase_agent.analysis.state.scientific_decision_brief import decision_brief

    decision_state.setdefault("decision_context", {})["scientific_decision_brief"] = decision_brief(
        current
    )
    if generation_required:
        # The user requires branch generation before downstream searches in a new epoch.
        allowed = [tool for tool in allowed if tool in {"generate_branches", "pause_search"}]
        safe_next = None
        if direction:
            return {
                "status": "rejected",
                "state": current,
                "reason": "新模型刷新后须先生成新branch并展示结果；旧方向方案需重新审阅。",
            }
    planned_action = saved_followup(direction) if direction else None
    if planned_action:
        planned_action["decision_source"] = "saved_agent_direction_plan"
    action = planned_action or safe_next
    if not action:
        from phase_agent.tools.workflows.relax_proposal_feedback import propose_with_relax_feedback

        action, prior_usage, preflight_error = propose_with_relax_feedback(
            decision_state,
            agent_client=agent_client,
            allowed_tools=allowed,
            config=config,
            propose=propose,
        )
        if prior_usage:
            current = record_budget_usage(current, {"llm_usage": prior_usage})
        if preflight_error:
            if action.get("_llm_usage"):
                current = record_budget_usage(current, {"llm_usage": action["_llm_usage"]})
            return {"status": "rejected", "state": current, "reason": preflight_error}
    from phase_agent.decisions.agent.dialogue_contract import (
        is_dialogue,
        dialogue_result,
    )

    if unified and is_dialogue(action):
        if action.get("_llm_usage"):
            current = record_budget_usage(current, {"llm_usage": action["_llm_usage"]})
        return dialogue_result(action, current)
    if (
        training_pending_validation(current)
        and (action.get("tool") or action.get("action_type")) == "update_mlip"
    ):
        return {
            "status": "not_configured",
            "state": current,
            "reason": "本轮微调已完成，重复训练方案已拦截。请先审阅切换搜索或补采样/DFT的方案。",
        }
    from phase_agent.tools.policy.workflow_policy import initial_direction_errors

    initial_errors = (
        initial_direction_errors(action, current, assessment) if not model_failed(action) else []
    )
    if initial_errors:
        return {"status": "not_configured", "state": current, "reason": "; ".join(initial_errors)}
    if assessment and not model_failed(action):
        if not valid_post_dft_review(action):
            return {
                "status": "not_configured",
                "state": current,
                "reason": "DFT 后方案缺少误差、覆盖和微调/搜索取舍分析；未生成 branch 或训练。请继续重新评估。",
            }
        action["_post_dft_review_version"] = 5
        action["_post_dft_scope_key"] = assessment["scope_key"]
    if model_failed(action):
        if action.get("_llm_usage"):
            current = record_budget_usage(
                current,
                {
                    "llm_usage": action["_llm_usage"],
                    "iteration": current.get("iteration", 0),
                },
            )
        return {
            "status": "not_configured",
            "state": current,
            "post_dft_review": action.get("failed_post_dft_review"),
            "reason": "模型未能返回完整有效方案，未暂停搜索或执行任务。请继续重试。原因："
            + str(action.get("fallback_reason")),
        }
    if (
        not unified
        and not model_owned
        and mode == "interactive"
        and not post_mc
        and not assessment
        and action.get("decision_source") == "rule"
    ):
        recovery = choose_debug_next_action(
            current,
            (context or {}).get("manager"),
            (context or {}).get("effective_config") or config,
            allowed_tools=allowed,
            user_message=(context or {}).get("user_message"),
        )
        if recovery:
            recovery["_llm_usage"] = action.get("_llm_usage")
            recovery["fallback_reason"] = action.get("fallback_reason")
            action = recovery
    if (
        not unified
        and not model_owned
        and mode == "interactive"
        and has_mc_tasks
        and action.get("tool") == "allocate_mc_bohb"
        and not is_verified_second(
            action, current, (context or {}).get("effective_config") or config
        )
    ):
        safe_next = choose_debug_next_action(
            current,
            (context or {}).get("manager"),
            (context or {}).get("effective_config") or config,
            allowed_tools=allowed,
            user_message="继续",
        )
        if safe_next and safe_next.get("parameters", {}).get("round_kind") == "second":
            action = safe_next
        else:
            return mc_block(current, mode, (context or {}).get("effective_config") or config)
    if mode == "interactive" and not unified and not model_owned:
        original_usage = action.get("_llm_usage")
        original_evidence = action.get("evidence_refs")
        action = prepare_debug(
            action, current, context, config, allowed, invocation_id=invocation_id
        )
        if original_usage:
            action["_llm_usage"] = original_usage
        if original_evidence:
            action["evidence_refs"] = original_evidence
    if action.get("_llm_usage"):
        current = record_budget_usage(
            current,
            {
                "llm_usage": action["_llm_usage"],
                "iteration": current.get("iteration", 0),
            },
        )
    mc_intent = current.get("mc_budget_intent") or {}
    if action.get("tool") == "allocate_mc_bohb" and (action.get("parameters") or {}).get(
        "mc_budget"
    ) == mc_intent.get("steps"):
        current.pop("mc_budget_intent", None)
    if mode == "interactive" and action.get("tool") == "select_dft_candidates":
        from phase_agent.tools.workflows.prepare_dft_proposal import (
            prepare_dft_proposal,
        )

        preparation = prepare_dft_proposal(
            action,
            current,
            decision_state,
            (context or {}).get("effective_config") or config,
            agent_client=agent_client,
            revise=revise,
        )
        current = preparation.state
        if preparation.status == "rejected":
            return {
                "status": "rejected",
                "state": current,
                "reason": preparation.error,
                "post_dft_review": preparation.action.get("post_dft_review"),
            }
        action = preparation.action
    if action.get("tool") == "allocate_mc_bohb":
        from phase_agent.tools.workflows.prepare_mc_proposal import prepare_mc_proposal

        action, current, error = prepare_mc_proposal(
            action,
            current,
            decision_state,
            context or {},
            (context or {}).get("effective_config") or config,
            agent_client=agent_client,
            revise=revise,
        )
        if error:
            return {"status": "rejected", "state": current, "reason": error}
    if (action.get("parameters") or {}).get("mode") == "relax_mc_inputs":
        from phase_agent.tools.relax_mc_batch.plan import attach_preview

        try:
            action = attach_preview(
                action,
                current,
                {
                    **(context or {}),
                    "config_version": current.get("confirmed_config_version")
                    or (context or {}).get("config_version"),
                },
                (context or {}).get("effective_config") or config,
            )
        except ValueError as error:
            return {"status": "rejected", "state": current, "reason": str(error)}
    from phase_agent.tools.local.prepare_ordering_search import attach_ordering_preview

    try:
        action = attach_ordering_preview(
            action,
            current,
            context or {},
            (context or {}).get("effective_config") or config,
        )
    except (ValueError, OSError, RuntimeError) as error:
        return {"status": "not_configured", "state": current, "reason": str(error)}
    from phase_agent.analysis.cost.generation_preflight import (
        attach_generation_preflight,
    )

    try:
        action = attach_generation_preflight(
            action, current, context, (context or {}).get("effective_config") or config
        )
    except ValueError as error:
        return {
            "status": "not_configured",
            "state": current,
            "reason": str(error),
            "post_dft_review": action.get("post_dft_review"),
        }
    if direction:
        if (action.get("tool") or action.get("action_type")) not in TOOLS[direction["direction"]]:
            return {
                "status": "not_configured",
                "state": current,
                "reason": "具体方案偏离已批准方向，已拦截；未执行。",
            }
        action["_approved_direction_hash"] = direction_hash(direction)
        action.setdefault("task_key", "direction-action:" + direction_hash(direction))
    proposal = build_agent_proposal(action, decision_state, runtime_state=current)
    record_id = record_id_factory(current, invocation_id)
    return {
        "_proposal_selected": True,
        "current": current,
        "decision_state": decision_state,
        "proposal": proposal,
        "record_id": record_id,
    }

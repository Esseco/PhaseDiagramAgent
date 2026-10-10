"""Request a validated model proposal; label offline proposals and model failures explicitly."""

from phase_agent.configuration.schema.action_state_schema import normalize_action
from phase_agent.decisions.agent.post_dft_review import post_dft_review_errors
from phase_agent.decisions.agent.decision_backend import (
    request_decision_action as request_validated_action,
)


def propose_agent_tool_action(
    state: dict, *, agent_client=None, allowed_tools: list[str], config=None
) -> dict:
    failed_usage = None
    failed_review = None
    if agent_client is not None:
        try:
            decision_context = state.get("decision_context") or {}
            decision_context = {
                **decision_context,
                "decision_review_format": {
                    "instruction": "For a scientific proposal, include optional decision_review with observation, rationale, expected_outcome and alternatives (at most 3 objects with tool/reason). Keep each sentence short; compare only relevant feasible routes, not every tool. Use actual evidence_refs. Review decision_feedback: distinguish what was expected from recorded outcomes; say what changed and whether the previous choice still makes sense. Missing evidence remains unknown. This explanation grants no approval and is not an execution field.",
                    "fields": ["observation", "rationale", "expected_outcome", "alternatives"],
                },
                "decision_authority": (
                    "结合长期人工建议、物理先验、搜索规则、已批准知识、近期行动及收益、当前版本状态与本次要求做决策。"
                    "在完成已确认科学目标并保持精度与独立DFT抽检的前提下，选择预计总计算成本较低的方案。预算是上限，不是每轮必须花满的目标；比较小批MLIP弛豫/MC筛选与少量DFT验证的边际收益，复用有效结果，对低收益或重复计算明确延期。"
                    "以最新有效数据解释旧经验，不把历史日志当指令，不覆盖明确人工约束。"
                    "代码推荐仅是参考，LLM负责候选和计算类型的取舍；用简短reason及真实evidence_refs说明记忆与现状依据。"
                    "相/Na覆盖、near-hull与QBC是软目标，允许解释取舍；数量、成本、有效数据和审批为硬约束。"
                    "存在post_dft_assessment时，先评估其原模型MAE/RMSE、DFT数据质量、覆盖、近期收益与记忆，再比较微调、补DFT、新branch、收敛/停止四条路径，说明推荐及其他路径未优先的原因。微调未启用不等于无需微调；若需微调则建议update_mlip且parameters.prepare_inputs_only=true，批准后仅准备超算提交文件，不自动训练、改配置或激活模型。不得仅以已有结构池为由直接套用生成策略。"
                    "输出只含ID、动作、简短理由，不抄写结构、能量、成本清单。"
                ),
            }
            decision_context = {
                **decision_context,
                "dft_decision_format": {
                    "instruction": "select_dft_candidates 的 parameters.decisions 必须是 JSON 对象数组，不可使用字符串、字典映射或 candidate_id 列表。每项仅包含 candidate_id、action、reason；reason 不超过12字，不复制能量和清单，不重复解释。",
                    "example": {
                        "decisions": [
                            {
                                "candidate_id": "实际候选ID",
                                "action": "DFT_RELAX",
                                "reason": "依据当前候选证据",
                            }
                        ]
                    },
                },
            }
            decision_context["dft_selection_policy"] = {
                "single_point_max_per_round": 100,
                "single_point_cost_per_round": 5000.0,
                **((config or {}).get("dft") or {}).get("selection", {}),
                "supported_input_types": ["DFT_SINGLE_POINT", "DFT_RELAX"],
                "instruction": "遵守已确认的单点优先和 max_relax_fraction 优化数量上限；单点不先优化。不得把全部候选分为优化后依赖执行器自动改成单点。",
            }
            decision_context["scientific_scope"] = {
                "boundary": {
                    key: value
                    for key, value in (
                        ((config or {}).get("system") or {}).get("boundary") or {}
                    ).items()
                    if key in {"P", "TM_ratio"}
                },
                "configuration_space": ((config or {}).get("system") or {}).get(
                    "configuration_space"
                ),
                "instruction": "Respect fixed variables independently of element count. Fixed T preserves TM occupancy from phase references, not atomic positions. Single TM has no TM exchange freedom but still has physical interactions. Single phase has no inter-phase competition. Never use disabled strategies to fill quotas.",
            }
            decision_context["early_branch_planning"] = {
                "confirmed_run": dict((config or {}).get("run") or {}),
                "instruction": (
                    "前期缺少branch与有效结果时，应先建立充分合法覆盖；以confirmed_run的total_quota和batch_size为规划起点，"
                    "结合用户扩大前期覆盖的偏好、相/Na/H空间、去重、剩余预算及Relax/MC预估安排数量。"
                    "不得无依据退回小样本批次；明显缩小数量须说明实际预算或合法空间等证据。"
                    "候选数与入选上限须区分，入选不是保证数量；不机械花满预算。"
                    "branch方案不附带假设性DFT任务或成本；DFT以后独立分析审批。"
                ),
            }
            from phase_agent.decisions.agent.generation_plan import configured_generation_strategies

            decision_context["enabled_generation_strategies"] = configured_generation_strategies(
                config or {}
            )
            decision_context["generation_strategy_constraint"] = (
                "generation_plan和quotas只能使用enabled_generation_strategies中的策略。"
                "单相体系不能分配competing_phase；不得用禁用策略补足数量。"
                "修改已有方案时重新分析合法分配，不静默转换策略或保留冲突汇总。"
            )
            if decision_context.get("post_dft_assessment"):
                from phase_agent.decisions.agent.post_dft_review import review_output_instruction

                decision_context["post_dft_review_format"] = (
                    "Required exception to the compact action field list: "
                    + review_output_instruction()
                )
                decision_context["generation_allocation_format"] = (
                    "For generate_branches, parameters.generation_plan is REQUIRED: at most 12 objects "
                    "with strategy, quota, phase (allowed phase or all), optional na_min/na_max in Na/O2, "
                    "strategy MUST be exactly coverage/composition/competing_phase/periodic_extension/tm_ordering; "
                    "quota MUST be a positive JSON integer, not text or a budget. "
                    "optional positive integer max_det_H per allocation, and reason under160 characters. "
                    "Use generation_framework_costs to compare atom counts and downstream Relax/MC "
                    "cost BEFORE selecting expansion sizes and quantities. Prefer affordable smaller cells; "
                    "do not expand first to discover cost. Cost/time are estimates, not authorizations. "
                    "Derive allocations from coverage, valid hull/QBC, "
                    "memory and recent strategy gains; absence of a stable phase alone is not a gap. "
                    "Include justified exploration. Supply allocation quotas in generation_plan; the program derives "
                    "parameters.quotas and total_quota when omitted. Explicit aggregate counts must agree. Choose batch_size within confirmed limit; defaults are ceilings, not "
                    "mandatory counts. New exception: generation_plan is an allowed parameter. "
                    "Never claim P3/lowNa targeting with a generic unbounded allocation."
                )
            if decision_context.get("unified_dialogue"):
                from phase_agent.decisions.agent.dialogue_contract import dialogue_output_contract

                decision_context["dialogue_output_contract"] = dialogue_output_contract()
            from phase_agent.decisions.agent.round_budget_review import round_budget_instruction

            decision_context["round_budget_review_format"] = round_budget_instruction()
            if "generate_branches" in allowed_tools:
                from phase_agent.decisions.agent.generation_contracts import GenerationAllocation

                decision_context["generation_allocation_contract"] = {
                    "item_schema": GenerationAllocation.model_json_schema(),
                    "constraints": [
                        "generation_plan contains 1..12 allocations; quota is a positive JSON integer",
                        "generation_plan is the allocation source; omit redundant parameters.quotas and total_quota for automatic derivation",
                        "if aggregate counts are explicitly supplied, they must equal the sums of generation_plan",
                        "phase must belong to the confirmed allowed phases or be all",
                        "first batch cannot use strategies requiring existing parent branches",
                    ],
                }
            if decision_context.get("training_already_completed") or decision_context.get(
                "training_validation_planning"
            ):
                decision_context["decision_authority"] = (
                    "本轮微调已经完成并回收；当前是微调后的策略判断。不得重复update_mlip。"
                    "比较换新模型、旧模型补DFT、接口核查或其他策略，沿用已记录的Agent模型审阅理由。"
                    "只返回allowed_tools中的动作。核查或配置修改用adjust_strategy；补DFT用select_dft_candidates。"
                    "用简短reason说明依据，不让用户做科学选择。"
                )
            action = request_validated_action(
                agent_client,
                {
                    "mode": "autonomous_search",
                    "unified_dialogue": bool(decision_context.get("unified_dialogue")),
                    "state": state,
                    "decision_context": decision_context,
                    "allowed_tools": allowed_tools,
                    "instruction": "For a scientific action only, choose exactly one registered tool. Return one compact JSON object with tool, task_key, target_ids, parameters, budget, reason, expected_purpose and evidence_refs; include required post_dft_review when post_dft_assessment exists. You may also include the optional decision_review explanation object described in decision_context.decision_review_format. budget must be one non-negative JSON number in relative_cost units, never an object. Keep reason and expected_purpose under 100 Chinese characters each; never copy the state into the answer. For generate_branches, parameters may contain only total_quota, quotas, batch_size, initial_states_per_branch, seed, max_det_H and generation_plan. Follow generation_allocation_format for the required post-DFT plan. Preserve an explicit user det(H) limit in max_det_H as a positive JSON integer; omit the field when no limit is supplied instead of returning null or text. The limit applies to this generation batch and must be shown in the proposal. On the first batch, use coverage across allowed phases and prefer smaller H; parent-dependent strategies cannot run until branches exist. Do not claim a code fix through action parameters. restart_failed_task may target only a real task_id listed in decision_context.retryable_tasks. A webui record_id or failed generate_branches action is not a scientific task and cannot be restarted with that tool. For allocate_mc_bohb, select existing branch_id values from decision_context.available_branches in target_ids; put focus_regions, exploration_fraction, mc_budget and dft_budget in parameters. Hyperband controls fidelity and promotions after Relax screening. Batch MC planning and input generation use allocate_mc_bohb or prepare_local_batch_files, never run_calculation_stage. run_calculation_stage requires exactly one structure_id and is not a batch action. If all current MC tasks are complete and the confirmed policy disables a second segment, assess convergence or the next stage instead of proposing another MC batch. For select_dft_candidates, use existing qbc_candidates only and categorical decisions DFT_SINGLE_POINT, DFT_RELAX, DEFER or REJECT. Never invent energies, Ehull, scores, uncertainty or convergence.",
                },
            )
            from phase_agent.decisions.agent.dialogue_contract import is_dialogue

            if is_dialogue(action) and decision_context.get("unified_dialogue"):
                return action
            if not isinstance(action, dict):
                raise TypeError("agent action must be dict")
            tool = action.get("tool") or action.get("action_type")
            if tool is None and isinstance(action.get("action"), dict):
                envelope = action
                action = dict(envelope["action"])
                if envelope.get("_llm_usage"):
                    action["_llm_usage"] = envelope["_llm_usage"]
                tool = action.get("tool") or action.get("action_type")
            if not isinstance(tool, str) or tool not in allowed_tools:
                raise ValueError(f"Agent 返回未注册的动作：{tool!r}")
            from phase_agent.decisions.agent.action_contracts import action_contract_errors

            envelope_errors = action_contract_errors(action)
            if envelope_errors:
                raise ValueError("Invalid LLM action: " + "; ".join(envelope_errors))
            from phase_agent.decisions.agent.post_dft_review import valid_post_dft_review

            if decision_context.get("post_dft_assessment") and not valid_post_dft_review(action):
                raise ValueError(
                    "DFT 后决策缺少有效分析：" + "; ".join(post_dft_review_errors(action))
                )
            if (action.get("tool") or action.get("action_type")) == "restart_failed_task":
                retryable = {
                    row.get("task_id") for row in decision_context.get("retryable_tasks") or []
                }
                targets = action.get("target_ids") or []
                if len(targets) != 1 or targets[0] not in retryable:
                    raise ValueError("restart_failed_task requires a retryable scientific task_id")
            if (action.get("tool") or action.get("action_type")) == "generate_branches":
                unsupported = set(action.get("parameters") or {}) - {
                    "total_quota",
                    "quotas",
                    "batch_size",
                    "initial_states_per_branch",
                    "seed",
                    "max_det_H",
                    "generation_plan",
                }
                if unsupported:
                    raise ValueError(
                        f"unsupported generate_branches parameters: {sorted(unsupported)}"
                    )
                action = _apply_generation_defaults(action, state, config or {})
                if decision_context.get("post_dft_assessment") or (
                    action.get("parameters") or {}
                ).get("generation_plan"):
                    from phase_agent.decisions.agent.generation_plan import validate_generation_plan

                    validate_generation_plan(action["parameters"])
            action = _ensure_task_key(action, state)
            from phase_agent.analysis.state.decision_evidence_catalog import (
                check_decision_evidence_refs,
            )

            evidence_check = check_decision_evidence_refs(
                action.get("evidence_refs") or [], decision_context.get("evidence_catalog") or {}
            )
            return normalize_action(
                {
                    **action,
                    "decision_source": "llm_agent",
                    "decision_context": decision_context,
                    "evidence_reference_check": evidence_check,
                }
            )
        except Exception as error:
            failed_review = getattr(error, "post_dft_review", None)
            failed_usage = getattr(error, "llm_usage", None)
            returned_action = locals().get("action")
            if (
                failed_review is None
                and isinstance(returned_action, dict)
                and decision_context.get("post_dft_assessment")
            ):
                if not post_dft_review_errors(returned_action, include_parameters=False):
                    failed_review = returned_action["post_dft_review"]
            if failed_usage is None and isinstance(returned_action, dict):
                failed_usage = returned_action.get("_llm_usage")
            fallback_reason = f"llm_failed: {type(error).__name__}: {error}"
    else:
        fallback_reason = "llm_not_configured"
    if _needs_initial_generation(state) and "generate_branches" in allowed_tools:
        config_version = state.get("config_version") or "unversioned"
        action = {
            "tool": "generate_branches",
            "task_key": f"bootstrap-branches:{config_version}",
            "target_ids": [],
            "parameters": {},
            "budget": 0.0,
            "reason": "首轮尚无 branch、任务或相图结果；先按已确认边界生成候选 branch。",
            "expected_purpose": "建立首轮合法 branch 候选池，供后续结构初始化与 Relax 筛选。",
            "decision_source": "rule",
            "fallback_reason": fallback_reason,
            "_llm_usage": failed_usage,
        }
        return normalize_action(_apply_generation_defaults(action, state, config or {}))
    tool = "pause_search" if "pause_search" in allowed_tools else "check_convergence"
    return normalize_action(
        {
            "tool": tool,
            "target_ids": [],
            "parameters": {},
            "budget": 0.0,
            "reason": "rule fallback",
            "decision_source": "rule",
            "fallback_reason": fallback_reason,
            "_llm_usage": failed_usage,
            "failed_post_dft_review": failed_review,
        }
    )


def _needs_initial_generation(state: dict) -> bool:
    context = state.get("decision_context") or {}
    hull = state.get("current_convex_hull") or {}
    has_hull_entries = any((item or {}).get("stable_entries") for item in hull.values())
    has_scientific_work = any(
        (
            state.get("available_branches"),
            context.get("available_branches"),
            (state.get("short_term_memory") or {}).get("task_status"),
            has_hull_entries,
        )
    )
    history = state.get("search_history") or []
    has_successful_generation = any(
        row.get("action_type") == "generate_branches" and row.get("status") == "completed"
        for row in history
    )
    return not has_scientific_work and not has_successful_generation


def _apply_generation_defaults(action: dict, state: dict, config: dict) -> dict:
    """Keep Agent proposals within the user-confirmed generation scale."""
    from copy import deepcopy

    result = deepcopy(action)
    params = result.get("parameters")
    if not isinstance(params, dict):
        params = {}
        result["parameters"] = params
    run = config.get("run") or config
    strategy_config = config.get("round_strategy") or {}
    total_default = int(run.get("total_quota", config.get("total_quota", 300)))
    total_limit = int(strategy_config.get("generation_quota_total", total_default))
    total_limit = max(total_default, total_limit)
    params["total_quota"] = min(total_limit, max(1, int(params.get("total_quota", total_default))))
    batch_default = int(run.get("batch_size", config.get("batch_size", 96)))
    params["batch_size"] = min(
        params["total_quota"], max(1, int(params.get("batch_size", batch_default)))
    )
    initial_default = int(
        run.get("initial_states_per_branch", config.get("initial_states_per_branch", 4))
    )
    params["initial_states_per_branch"] = min(
        3, max(1, int(params.get("initial_states_per_branch", initial_default)))
    )
    invalid_h_cap = False
    if "max_det_H" in params:
        raw_cap = params["max_det_H"]
        if raw_cap is None or raw_cap == "":
            params.pop("max_det_H")
        elif type(raw_cap) is int and raw_cap > 0:
            pass
        elif type(raw_cap) is float and raw_cap.is_integer() and raw_cap > 0:
            params["max_det_H"] = int(raw_cap)
        elif isinstance(raw_cap, str):
            import re

            match = re.fullmatch(r"\s*(?:≤|<=)?\s*(\d+)(?:\.0+)?\s*", raw_cap)
            if match and int(match.group(1)) > 0:
                params["max_det_H"] = int(match.group(1))
            else:
                invalid_h_cap = True
                params.pop("max_det_H")
        else:
            invalid_h_cap = True
            params.pop("max_det_H")
    h_generation = (config.get("system_config") or config.get("system") or {}).get(
        "H_generation"
    ) or {}
    if _needs_initial_generation(state) and h_generation.get("enabled"):
        configured_cap = h_generation.get("first_round_max_det_H")
        size_max = h_generation.get("size_max")
        if configured_cap is not None:
            if type(configured_cap) is not int or configured_cap <= 0:
                raise ValueError(
                    "system.H_generation.first_round_max_det_H must be a positive integer"
                )
            params["max_det_H"] = min(params.get("max_det_H", configured_cap), configured_cap)
        elif type(size_max) is int and size_max > 0:
            params["max_det_H"] = min(params.get("max_det_H", size_max), size_max)

    if invalid_h_cap:
        result["reason"] = "Agent 给出的 H 上限格式无效，已改用确认配置的 H 边界；请核对本轮上限。"

    strategy = strategy_config.get("rule_default") or {}
    configured = strategy.get("generation_quotas") or {}
    enabled = set(
        (config.get("generation_actions") or {}).get("enabled")
        or ("coverage", "composition", "competing_phase", "tm_ordering", "periodic_extension")
    )

    def valid_quotas(source):
        if not isinstance(source, dict):
            return {}
        return {
            name: value
            for name, value in source.items()
            if name in enabled
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        }

    quotas = valid_quotas(params.get("quotas")) or valid_quotas(configured)
    if params.get("generation_plan"):
        # An explicit plan is authoritative; do not fill or reorder its quotas.
        from phase_agent.decisions.agent.generation_plan import validate_generation_plan

        validate_generation_plan(params)
        return result
    quota_limit = params["total_quota"]
    if _needs_initial_generation(state):
        quotas = {"coverage": params["total_quota"]}
    else:
        capped = {}
        remaining = quota_limit
        for name, value in quotas.items():
            if remaining <= 0:
                break
            capped[name] = min(value, remaining)
            remaining -= capped[name]
        quotas = capped
        quota_total = sum(quotas.values())
        if quota_total < params["total_quota"]:
            quotas["coverage"] = quotas.get("coverage", 0) + params["total_quota"] - quota_total
        params["total_quota"] = max(params["total_quota"], sum(quotas.values()))
    params["quotas"] = quotas or {"coverage": params["total_quota"]}
    return result


def _ensure_task_key(action: dict, state: dict) -> dict:
    """Program owns idempotency keys; an LLM omission must not reject a valid action."""
    updated = dict(action)
    tool = updated.get("tool") or updated.get("action_type")
    if tool not in {"check_convergence", "pause_search"} and not updated.get("task_key"):
        snapshot = state.get("snapshot_id") or f"state-{int(state.get('snapshot_index', 0)):06d}"
        config = state.get("config_version") or "unversioned"
        updated["task_key"] = f"{tool}:{config}:{snapshot}"
    return updated

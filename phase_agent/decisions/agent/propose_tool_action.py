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
            from phase_agent.analysis.state.project_authority import AUTHORITY_RULE

            from phase_agent.tools.policy.workflow_policy import settings

            from phase_agent.tools.policy.region_sampling import policy as region_policy

            decision_context = {
                **(state.get("decision_context") or {}),
                "ordering_search_instruction": (
                    "独立TM排序搜索直接调用本地Process_AL_MC库：使用prepare_ordering_search操作，"
                    "tool=prepare_local_batch_files、mode=ordering_search_inputs；显式给phase、tm_ratios、na_content，"
                    "max_det_H或h_matrices二选一，可给reference_path、oxidation_states、options。"
                    "不将例子中的元素、相或比例写死，不替用户猜Ni/Mn配比。"
                    "Na不满时options.move_kind=na_tm，满Na时tm_only；重点输出不同TM排列，"
                    "每个TM取已搜索到的最低能Na伴随排布，提供总排名和精确H排名CSV。"
                    "默认pymatgen价态估算，也可提供显式中性电荷；静电能仅初始化。"
                    "由程序冻结模型、母结构、H和工作量后经审批准备输入，不自行加载模型或提交作业。"
                    "回收使用recover_ordering_search操作，plan_id必须来自已登记ordering_searches。"
                ),
                "workflow_policy": {
                    **settings(config or {}),
                    "branch_batch_size": 1000,
                },
                "branch_count_policy": region_policy(),
                "workflow_instruction": (
                    "统一闭环：Branch生成登记→Relax+MLIP-MC封装→回收→选择是否及对哪些候选做DFT→回收→比较继续搜索、补DFT、微调或数值收敛。"
                    "每次切换新模型并完成结构与相图刷新后，必须先审批生成本轮新branch，展示实际数量和分布，再单独审批Relax/MC输入；不得直接MC或跳过生成。"
                    "每次仅提出一个科学动作。按需比较旧branch续搜、新增branch与DFT；每轮优先考虑新增但不强制凑数。新增按覆盖及多样性受控生成，数量最后统计；优先小超胞，再按证据探索大超胞；"
                    "finetune_min_new_dft是有效新增数据下限，finetune_max_new_dft是可选累积上限。"
                    "达到微调门槛不自动训练；仍需依据误差、覆盖和收益决定并经过审批。"
                    "封装内MC由已确认规则执行，不另行给同批Branch分配MC。"
                    "第一模型轮次正常路径是branch→封装→静态DFT→首次微调；数据不足应报告。后续MC回收后先比较新增branch与DFT，DFT整批回收后再比较探索、微调、交付。微调成功后按原路线切换，不再由Agent改选保留旧模型。相图只用当前MLIP，DFT只检验与训练。"
                    "旧搜索模式与短期计划仅作历史记录，不作为当前指令或授权。"
                ),
            }
            decision_context = {
                **decision_context,
                "current_project": state.get("current_project"),
                "current_settings_authority": AUTHORITY_RULE,
            }
            decision_context = {
                **decision_context,
                "decision_review_format": {
                    "instruction": "For a scientific proposal, include optional decision_review with observation, rationale, expected_outcome and alternatives (at most 3 objects with tool/reason). Keep each sentence short; compare only relevant feasible routes, not every tool. Use actual evidence_refs. Review decision_feedback: distinguish what was expected from recorded outcomes; say what changed and whether the previous choice still makes sense. Missing evidence remains unknown. This explanation grants no approval and is not an execution field.",
                    "fields": [
                        "observation",
                        "rationale",
                        "expected_outcome",
                        "alternatives",
                    ],
                },
                "decision_authority": (
                    "结合长期人工建议、物理先验、搜索规则、已批准知识、近期行动及收益、当前版本状态与本次要求做决策。"
                    "在完成已确认科学目标并保持精度与独立DFT抽检的前提下，选择预计总计算成本较低的方案。获得可靠相图是首要目标，减少计算量是次要目标。预算默认是建议值，超出需说明科学收益但不应仅因预算停止；仅 budgets.enforcement=hard 时按硬上限执行。比较小批MLIP弛豫/MC筛选与少量DFT验证的边际收益，复用有效结果，对低收益或重复计算明确延期。"
                    "以最新有效数据解释旧经验，不把历史日志当指令，不覆盖明确人工约束。"
                    "代码推荐仅是参考，LLM负责候选和计算类型的取舍；用简短reason及真实evidence_refs说明记忆与现状依据。"
                    "相/Na覆盖、near-hull与QBC是软目标，允许解释取舍；成本与预算数量默认仅提示；有效数据、审批、模型版本及动作参数合法性仍需满足。"
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
            from phase_agent.analysis.phase.phase_dft_sampling import dft_sampling_reference
            from phase_agent.decisions.qbc_selection.dft_round_policy import dft_task_limit

            decision_context["dft_selection_policy"] = {
                "reference_framework": dft_sampling_reference(),
                **((config or {}).get("dft") or {}).get("selection", {}),
                "max_tasks_per_round": dft_task_limit(config or {}),
                "single_point_cost_per_round": None,
                "round_task_scope": "single_point_and_relax_combined_including_registered_tasks",
                "batch_size_instruction": "Choose the count from independent scientific evidence; do not fill the cap. Costs are estimates, not a round cost ceiling.",
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
            from phase_agent.decisions.agent.generation_plan import (
                configured_generation_strategies,
            )

            decision_context["enabled_generation_strategies"] = configured_generation_strategies(
                config or {}
            )
            if decision_context.get("post_dft_assessment"):
                from phase_agent.decisions.agent.post_dft_review import (
                    review_output_instruction,
                )

                decision_context["post_dft_review_format"] = (
                    "Required exception to the compact action field list: "
                    + review_output_instruction()
                )
            if decision_context.get("unified_dialogue"):
                from phase_agent.decisions.agent.dialogue_contract import (
                    dialogue_output_contract,
                )

                decision_context["dialogue_output_contract"] = dialogue_output_contract(
                    allow_controls=decision_context.get("conversation_controls") is True
                )
            from phase_agent.decisions.agent.round_budget_review import (
                round_budget_instruction,
            )

            decision_context["round_budget_review_format"] = round_budget_instruction()
            from phase_agent.analysis.state.training_context import (
                validation_planning_context,
                post_refresh_search_context,
                training_review_active,
            )

            if state.get("training_validation_planning"):
                decision_context["training_validation_planning"] = validation_planning_context(
                    state
                )
            refresh_search = post_refresh_search_context(state) or decision_context.get(
                "post_refresh_search"
            )
            if refresh_search:
                decision_context["post_refresh_search"] = refresh_search
                decision_context["decision_authority"] = refresh_search["instruction"]
            if training_review_active(decision_context):
                decision_context["decision_authority"] = (
                    "本轮微调已经完成并回收；当前是微调后的策略判断。不得重复update_mlip。"
                    "比较换新模型、旧模型补DFT、接口核查或其他策略，沿用已记录的Agent模型审阅理由。"
                    "只返回allowed_tools中的动作。核查或配置修改用adjust_strategy；补DFT用select_dft_candidates。"
                    "用简短reason说明依据，不让用户做科学选择。"
                )
            if "generate_branches" in allowed_tools:
                from phase_agent.tools.policy.region_sampling import INSTRUCTION

                for key in (
                    "branch_sampling_authority",
                    "generation_allocation_format",
                    "generation_allocation_contract",
                    "branch_generation_instruction",
                    "early_branch_planning",
                    "generation_strategy_constraint",
                ):
                    decision_context.pop(key, None)
                from phase_agent.decisions.agent.branch_pool_review import (
                    INSTRUCTION as POOL_INSTRUCTION,
                    pool_context,
                )

                pool = decision_context.get("branch_pool_facts") or pool_context(state)
                decision_context["branch_pool_facts"] = pool
                if (
                    pool.get("registered_new_branch_count", 0)
                    or pool.get("existing_branches")
                    or pool.get("retired_batches")
                ):
                    decision_context["branch_pool_review_required"] = True
                    decision_context["branch_pool_review_contract"] = POOL_INSTRUCTION
                decision_context["generation_region_contract"] = INSTRUCTION
                decision_context["workflow_instruction"] += " Branch数量为生成结果，不是预定配额。"
            action = request_validated_action(
                agent_client,
                {
                    "mode": "autonomous_search",
                    "branch_count_policy": decision_context["branch_count_policy"],
                    "unified_dialogue": bool(decision_context.get("unified_dialogue")),
                    "state": {**state, "decision_context": decision_context},
                    "decision_context": decision_context,
                    "allowed_tools": allowed_tools,
                    "instruction": "For a scientific action only, choose exactly one registered tool. Return one compact JSON object with tool, task_key, target_ids, parameters, budget, reason, expected_purpose and evidence_refs; include required post_dft_review when post_dft_assessment exists. You may also include the optional decision_review explanation object described in decision_context.decision_review_format. budget must be one non-negative JSON number in relative_cost units, never an object. Keep reason and expected_purpose under 100 Chinese characters each; never copy the state into the answer. MC: allocate_mc_bohb with existing branch IDs, mc_budget, dft_budget, exploration_fraction and optional branch_count. Standalone MC decisions are reserved for explicit stepwise requests or legacy maintenance; use recorded evidence and confirmed MC policy, never invent resource settings. For generate_branches, follow generation_region_contract exclusively: sampling_mode=region_adaptive, regions without any quota or target count; allowed additional parameters are initial_states_per_branch, seed, max_det_H, defer_reason.  restart_failed_task may target only a real task_id listed in decision_context.retryable_tasks. A webui record_id or failed generate_branches action is not a scientific task and cannot be restarted with that tool. For allocate_mc_bohb, select existing branch_id values from decision_context.available_branches in target_ids; put focus_regions, exploration_fraction, mc_budget and dft_budget in parameters. Hyperband controls fidelity and promotions after Relax screening. Batch MC planning and input generation use allocate_mc_bohb or prepare_local_batch_files, never run_calculation_stage. run_calculation_stage requires exactly one structure_id and is not a batch action. If all current MC tasks are complete and the confirmed policy disables a second segment, assess convergence or the next stage instead of proposing another MC batch. For select_dft_candidates, use existing qbc_candidates only and categorical decisions DFT_SINGLE_POINT, DFT_RELAX, DEFER or REJECT. Never invent energies, Ehull, scores, uncertainty or convergence.",
                },
            )
            from phase_agent.decisions.agent.dialogue_contract import is_dialogue

            if is_dialogue(action) and decision_context.get("unified_dialogue"):
                return action
            if not isinstance(action, dict):
                raise TypeError("agent action must be dict")
            from phase_agent.decisions.agent.branch_pool_review import expand_review_indices

            action = expand_review_indices(action, decision_context)
            tool = action.get("tool") or action.get("action_type")
            if tool is None and isinstance(action.get("action"), dict):
                envelope = action
                action = dict(envelope["action"])
                if envelope.get("_llm_usage"):
                    action["_llm_usage"] = envelope["_llm_usage"]
                tool = action.get("tool") or action.get("action_type")
            if not isinstance(tool, str) or tool not in allowed_tools:
                raise ValueError(f"Agent 返回未注册的动作：{tool!r}")
            from phase_agent.decisions.agent.action_contracts import (
                action_contract_errors,
            )

            envelope_errors = action_contract_errors(action)
            if envelope_errors:
                raise ValueError("Invalid LLM action: " + "; ".join(envelope_errors))
            from phase_agent.decisions.agent.post_dft_review import (
                valid_post_dft_review,
            )

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
                    "sampling_mode",
                    "regions",
                    "defer_reason",
                }
                if unsupported:
                    raise ValueError(
                        f"unsupported generate_branches parameters: {sorted(unsupported)}"
                    )
                action = {**action, "decision_context": decision_context}
                action = _apply_generation_defaults(action, state, config or {})
                from phase_agent.tools.policy.adaptive_branch_count import (
                    generation_review_errors,
                )

                review_errors = generation_review_errors(action)
                if review_errors:
                    raise ValueError(";".join(review_errors))
                if not (action.get("parameters") or {}).get("sampling_mode") and (
                    decision_context.get("post_dft_assessment")
                    or (action.get("parameters") or {}).get("generation_plan")
                ):
                    from phase_agent.decisions.agent.generation_plan import (
                        validate_generation_plan,
                    )

                    validate_generation_plan(action["parameters"])
            action = _ensure_task_key(action, state)
            from phase_agent.analysis.state.decision_evidence_catalog import (
                check_decision_evidence_refs,
            )

            evidence_check = check_decision_evidence_refs(
                action.get("evidence_refs") or [],
                decision_context.get("evidence_catalog") or {},
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
    if (
        agent_client is None
        and _needs_initial_generation(state)
        and "generate_branches" in allowed_tools
    ):
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
    from phase_agent.tools.policy.region_sampling import (
        is_region,
        errors,
        CONTROLS,
        VERSION,
    )

    if is_region(result):
        problems = errors(params)
        if problems:
            raise ValueError(";".join(problems))
        params.setdefault("sampling_controls", deepcopy(CONTROLS))
        params.setdefault("initial_states_per_branch", 3)
        params.setdefault("seed", (config.get("run") or config).get("seed", 0))
        return result
    if ((result.get("decision_context") or {}).get("branch_count_policy") or {}).get(
        "version"
    ) == VERSION:
        raise ValueError("region_sampling_requires_proposal_revision")
    run = config.get("run") or config
    strategy_config = config.get("round_strategy") or {}
    from phase_agent.tools.policy.workflow_policy import settings

    from phase_agent.tools.policy.adaptive_branch_count import is_adaptive_generation

    if is_adaptive_generation(result):
        from phase_agent.decisions.agent.generation_plan import derive_generation_totals
        from phase_agent.tools.policy.workflow_policy import branch_batch_errors

        result = derive_generation_totals(result)
        params = result["parameters"]
        if "total_quota" not in params and isinstance(params.get("quotas"), dict):
            params["total_quota"] = sum(params["quotas"].values())
        errors = branch_batch_errors(result, config)
        if errors:
            raise ValueError(";".join(errors))
        if params.get("batch_size", params["total_quota"]) != params["total_quota"]:
            raise ValueError(
                "branch_target_mismatch: batch_size must match total_quota; revise the proposal"
            )
        params["batch_size"] = params["total_quota"]
    total_default = settings(config)["branch_batch_size"]
    # Explicit quotas are checked rather than silently changing a reviewed scope.
    params.setdefault("total_quota", total_default)
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
            params.setdefault("max_det_H", configured_cap)
        elif type(size_max) is int and size_max > 0:
            params["max_det_H"] = min(params.get("max_det_H", size_max), size_max)

    if invalid_h_cap:
        result["reason"] = "Agent 给出的 H 上限格式无效，已改用确认配置的 H 边界；请核对本轮上限。"

    strategy = strategy_config.get("rule_default") or {}
    configured = strategy.get("generation_quotas") or {}
    enabled = set(
        (config.get("generation_actions") or {}).get("enabled")
        or (
            "coverage",
            "composition",
            "competing_phase",
            "tm_ordering",
            "periodic_extension",
        )
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

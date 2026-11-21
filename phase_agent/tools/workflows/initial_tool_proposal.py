"""Initial scientific proposal selection and bounded preparation; no execution."""

from copy import deepcopy
from phase_agent.tools.budget.record_budget_usage import record_budget_usage
from phase_agent.policies.execution_policy import build_agent_proposal
from phase_agent.agents.common.post_dft_review import valid_post_dft_review


def _deterministic_cold_start_action(current, effective_config, allowed, *, context=None):
    """Build deterministic cold-start or first-finetune proposals without 4-agent LLM calls."""
    from phase_agent.evidence.parent_first_recovery import (
        initial_perturbation_direct_ready,
        small_parent_structure_ids,
    )

    workflow_cfg = (effective_config or {}).get("workflow") or {}
    if workflow_cfg.get("direct_cold_start_without_llm") is False:
        return None
    manager = (context or {}).get("manager")
    manager_data = (getattr(manager, "data", None) or {}) if manager is not None else {}
    if manager is not None and initial_perturbation_direct_ready(
        current, manager_data, allowed=allowed
    ):
        small_parents = small_parent_structure_ids(manager_data)
        dft_params = dict((effective_config.get("dft") or {}).get("parameters") or {})
        mlip_cfg = effective_config.get("mlip") or {}
        direct_cfg = effective_config.get("direct") or {}
        params = {
            "mode": "parent_first_recovery_inputs",
            "scope_id": "initial_direct",
            "round_evidence_digest": "initial_direct_v1",
            "confirmed": True,
            "initial_parents": small_parents,
            "dft_parameters": dft_params,
            "initial_setup": True,
            "small_supercell_only": True,
            "sampling_method": "direct",
            "direct_target": 100,
            "eval_mlip_on_selected_frames": True,
            "include_baseline_hull_and_errors": True,
        }
        model_sha = mlip_cfg.get("model_sha256")
        if model_sha:
            params["model_sha256"] = model_sha
        f_path = direct_cfg.get("feature_model_path") or mlip_cfg.get("feature_model_path")
        f_sha = direct_cfg.get("feature_model_sha256") or mlip_cfg.get("feature_model_sha256")
        if f_path:
            params["feature_model_path"] = f_path
        if f_sha:
            params["feature_model_sha256"] = f_sha
        return {
            "tool": "prepare_local_batch_files",
            "action_type": "prepare_local_batch_files",
            "operation": "prepare_parent_first_recovery",
            "target_ids": [],
            "parameters": params,
            "reasoning": "新项目冷启动：跳过多 Agent 讨论，直接生成小超胞母帧 MH-1 弛豫→微扰→特征差<=1筛选→reference-aware DIRECT→DFT 一体化待审批提案。",
            "decision_source": "deterministic_cold_start_policy",
        }
    initial_direct_done = (current.get("initial_direct_sampling") or {}).get(
        "status"
    ) == "completed"
    valid_dft_count = sum(
        1
        for row in current.get("dft_dataset_records") or []
        if isinstance(row, dict) and row.get("status", "completed") == "completed"
    )
    if (
        initial_direct_done
        and valid_dft_count >= 50
        and not current.get("remote_finetune_jobs")
        and not current.get("generation_history")
        and "update_mlip" in allowed
    ):
        return {
            "tool": "update_mlip",
            "action_type": "update_mlip",
            "operation": "prepare_mlip_training",
            "target_ids": [],
            "parameters": {
                "action": "RETRAIN_MLIP",
                "prepare_inputs_only": True,
                "handoff_scope": ["manifest"],
            },
            "reasoning": "初始微扰 DIRECT DFT 标签已回收并达标，跳过多 Agent 讨论，直接生成首轮微调待审批提案。",
            "decision_source": "deterministic_first_finetune_policy",
        }
    return None


def _deterministic_generation_action(decision_state, effective_config, current=None):
    """Build deterministic generate_branches proposal using dynamic strategy quotas."""
    from phase_agent.agents.strategic.strategy.choose_generation_strategy import (
        choose_generation_strategy,
        compute_strategy_efficiency,
    )

    context = (decision_state or {}).get("decision_context") or {}
    pool = context.get("branch_pool_facts") or {}
    metrics = dict(context.get("exploration_metrics") or {})
    if "coverage_and_marginal_matrix" not in metrics and pool.get("coverage_and_marginal_matrix"):
        metrics["coverage_and_marginal_matrix"] = pool["coverage_and_marginal_matrix"]
    if "strategy_efficiency" not in metrics and current is not None:
        metrics["strategy_efficiency"] = compute_strategy_efficiency(
            current, pool.get("existing_branches") or []
        )
    total_quota = int(
        ((effective_config or {}).get("exploration") or {}).get("branches_per_round", 10)
    )
    plan = choose_generation_strategy(
        metrics, total_quota=max(1, total_quota), config=effective_config
    )
    return {
        "tool": "generate_branches",
        "action_type": "generate_branches",
        "operation": "generate_branches",
        "target_ids": [],
        "parameters": {"quotas": plan["quotas"]},
        "reasoning": "新模型刷新后需先生成候选 branch，跳过多 Agent 讨论并按相图覆盖度/边际收益矩阵与策略效率动态配额直出提案。",
        "decision_source": "deterministic_generation_policy",
    }


def _deterministic_post_generation_relax_mc_action(
    current, decision_state, effective_config, allowed, *, context=None
):
    """Go straight from finished branch generation to prepare_relax_mc_batch without extra LLM calls."""
    workflow_cfg = (effective_config or {}).get("workflow") or {}
    if workflow_cfg.get("direct_relax_mc_after_generation") is False:
        return None
    if (
        "prepare_local_batch_files" not in allowed
        or (current.get("dedup_gate") or {}).get("status") != "ready"
    ):
        return None
    manager = (context or {}).get("manager")
    if manager is None:
        return None
    branches = (getattr(manager, "data", None) or {}).get("branches") or {}
    if not branches:
        return None
    dctx = (decision_state or {}).get("decision_context") or {}
    conflicts = {
        row.get("branch_id")
        for row in ((dctx.get("relax_mc_entry_conditions") or {}).get("conflicts") or [])
        if row.get("branch_id")
    }
    version = current.get("active_model_version")
    searched = {
        row.get("branch_id")
        for row in current.get("tasks") or []
        if row.get("stage") == "deep_search" and row.get("model_version") == version
    }
    latest_gen_ids = []
    for hist in reversed(current.get("generation_history") or []):
        if hist.get("model_version") in (None, version) and hist.get("registered_ids"):
            reg = set(hist["registered_ids"])
            latest_gen_ids = [
                bid
                for bid, b in sorted(branches.items())
                if reg.intersection(b.get("structure_ids") or [])
                and bid not in conflicts
                and bid not in searched
            ]
            if latest_gen_ids:
                break
    target_ids = latest_gen_ids or [
        bid for bid in sorted(branches) if bid not in conflicts and bid not in searched
    ]
    if not target_ids:
        return None
    small_supercell_only = all(int(branches[bid].get("det_H") or 1) <= 6 for bid in target_ids)
    params = {"mode": "relax_mc_inputs"}
    if small_supercell_only:
        params["bundle_post_mc_dft"] = True
        params["max_dft_frames"] = 100
    return {
        "tool": "prepare_local_batch_files",
        "action_type": "prepare_local_batch_files",
        "operation": "prepare_relax_mc_batch",
        "target_ids": target_ids,
        "parameters": params,
        "reasoning": "Branch 采样已完成，跳过额外专家讨论直接生成 Relax-MC 待审批提案。",
        "decision_source": "deterministic_post_generation_relax_mc_policy",
    }


def _has_unsearched_latest_generation(current, version, context):
    """Return True when the latest generation under `version` still has unsearched branches."""
    manager = (context or {}).get("manager")
    branches = (getattr(manager, "data", None) or {}).get("branches") or {}
    if not branches:
        return False
    searched = {
        row.get("branch_id")
        for row in current.get("tasks") or []
        if row.get("stage") == "deep_search" and row.get("model_version") == version
    }
    for hist in reversed(current.get("generation_history") or []):
        if hist.get("model_version") in (None, version) and hist.get("registered_ids"):
            reg = set(hist["registered_ids"])
            return any(
                reg.intersection(b.get("structure_ids") or []) and bid not in searched
                for bid, b in branches.items()
            )
    return False


def _post_dft_allowed_tools(allowed, *, awaiting_first_search=False):
    from phase_agent.agents.common.post_dft_review import REVIEW_TOOLS

    return [
        tool
        for tool in allowed
        if tool in REVIEW_TOOLS or (awaiting_first_search and tool == "prepare_local_batch_files")
    ]


def _post_mc_allowed_tools(allowed, generation, *, post_mc, allow_continue_exploration=False):
    """Apply the post-MC route gate without hiding a new model's first search."""
    tools = list(allowed)
    if not post_mc:
        return tools
    pending_post_refresh_search = (
        generation.get("post_refresh")
        and generation.get("new_branch_count", 0) > 0
        and not generation.get("first_search_completed")
    )
    blocked = {"allocate_mc_bohb", "run_calculation_stage"}
    if not allow_continue_exploration:
        blocked.add("generate_branches")
        if not pending_post_refresh_search:
            blocked.add("prepare_local_batch_files")
    return [tool for tool in tools if tool not in blocked]


def _post_dft_candidate_pool(candidates, state, *, frozen_dft_action=False):
    """Keep frozen selection and deferral frames available for identity validation."""
    if frozen_dft_action:
        return list(candidates)
    previous_dft = {
        t.get("structure_id")
        for t in state.get("tasks") or []
        if t.get("stage") in {"dft_single_point", "dft_relax"} and t.get("status") != "cancelled"
    }
    prior_relax = {
        t.get("structure_id")
        for t in state.get("tasks") or []
        if t.get("stage") == "dft_relax" and t.get("status") != "cancelled"
    }
    return [
        r
        for r in candidates
        if r.get("candidate_id") not in previous_dft
        or (r.get("geometry_optimization_evidence") and r.get("candidate_id") not in prior_relax)
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
    from phase_agent.agents.reliability.extrapolation_round import required_direct_round

    required = required_direct_round(current)
    if required:
        from phase_agent.evidence.parent_first_recovery import valid_parent_first_cycle
        from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled

        cycle = current.get("extrapolation_recovery_cycle") or {}
        batch = (current.get("parent_first_recovery_batches") or {}).get(
            cycle.get("batch_id")
        ) or {}
        if (
            cycle.get("scope_id") == required["scope_id"]
            and batch.get("round_evidence_digest") == required["evidence_digest"]
            and valid_parent_first_cycle(current)
            and not (
                cycle.get("policy_version") == "epoch_refresh_v1"
                and recovery_dft_failure_settled(current, cycle)
            )
        ):
            required = None
    if required:
        parent_batches = [
            b
            for b in (current.get("parent_first_recovery_batches") or {}).values()
            if b.get("scope_id") == required["scope_id"]
            and b.get("round_evidence_digest") == required["evidence_digest"]
            and b.get("status") in {"prepared", "submitted", "running", "waiting_dft_results"}
        ]
        if parent_batches:
            return {
                "status": "waiting_results",
                "state": current,
                "submitted": False,
                "reason": "母帧先弛豫恢复任务已登记，等待原批次结果或处理提交状态；不重复准备。",
            }
        if not callable((registry.get("prepare_local_batch_files") or {}).get("handler")):
            return {
                "status": "not_configured",
                "state": current,
                "submitted": False,
                "reason": "可靠性要求外推恢复，但采样处理器不可用。",
            }
        if not callable(agent_client) or not callable(propose):
            return {
                "status": "not_configured",
                "state": current,
                "submitted": False,
                "reason": "可靠性要求外推恢复，等待 Strategic 提出路线。",
            }
        batch = current.get("perturbation_recovery_batch") or {}
        if batch:
            if (
                batch.get("scope_id") != required["scope_id"]
                or batch.get("round_evidence_digest") != required["evidence_digest"]
            ):
                return {
                    "status": "not_configured",
                    "state": current,
                    "submitted": False,
                    "reason": "微扰任务与当前外推轮次不一致；保留原任务等待核对。",
                }
            task_map = {t.get("task_id"): t for t in current.get("tasks") or []}
            ids = batch.get("task_ids") or []
            if not ids or any(
                i not in task_map or task_map[i].get("status") not in {"completed", "failed"}
                for i in ids
            ):
                return {
                    "status": "waiting_results",
                    "state": current,
                    "submitted": False,
                    "reason": "等待已批准的 MH-1 微扰弛豫任务回收；已完成任务不重算。",
                }
            current = deepcopy(current)
            current["perturbation_recovery_batch"]["status"] = "results_received"
            from phase_agent.tools.workflows.perturbation_direct import perturbation_candidates

            try:
                candidates, excluded = perturbation_candidates(
                    current,
                    current["perturbation_recovery_batch"],
                    scope_id=required["scope_id"],
                    round_digest=required["evidence_digest"],
                )
            except (ValueError, OSError, KeyError) as error:
                return {
                    "status": "not_configured",
                    "state": current,
                    "submitted": False,
                    "reason": "MH-1 回收证据尚未通过核验：" + str(error),
                }
        from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy

        policy = (
            epoch_recovery_policy(current, required)
            if not batch
            else {"status": "sampling_required", "route": "perturbation"}
        )
        if policy["status"] != "sampling_required":
            reasons = {
                "attempt_limit": "本 epoch 已补充5次，合格结构不足50个；暂停并保留结果。",
                "unknown": "外推比例、轮次或补充记录不完整，需先核对。",
                "waiting_results": "补充批次尚有任务未结束，等待原结果，不重复提交。",
                "complete": "补充数量已达标，进入现有微调流程。",
                "clear": "当前外推比例未达到补充门槛。",
            }
            return {
                "status": "not_configured",
                "state": current,
                "submitted": False,
                "reason": reasons.get(policy["status"], "外推补充状态未知。"),
                "extrapolation_policy": policy,
            }
        evidence = decision_state.setdefault("decision_context", {})
        evidence["extrapolation_sampling_policy"] = policy
        evidence["mandatory_direct_round"] = deepcopy(required)
        evidence.setdefault("reliability_extrapolation_rounds", {"rounds": [deepcopy(required)]})
        if batch:
            evidence["perturbation_recovery_ready"] = {
                "candidate_count": len(candidates),
                "excluded": excluded,
            }
            evidence["mandatory_direct_instruction"] = (
                "已批准微扰结构微调且 MH-1 任务已回收；对核验通过的末态使用成功 DFT 参考进行 DIRECT，"
                "提出 sample_direct_extrapolation，recovery_route=perturbation，绑定本轮 scope 和证据摘要。"
            )
        else:
            evidence["mandatory_direct_instruction"] = (
                "可靠性确认本轮大量外推失败；按 extrapolation_sampling_policy.route 提出对应方案。"
                "新恢复仅允许微扰路线：Agent 提供已验证小扩胞母帧和 DFT 参考，"
                "由 HPC 先用原始 MH-1 弛豫母帧，再微扰、合理性检查、相对弛豫母帧特征差<=1筛选、DIRECT 和恢复DFT；微扰后不再MLIP弛豫；使用 prepare_parent_first_recovery。"
                "批准范围须绑定本轮 scope 和证据摘要，并明确 DFT 目标。"
            )
            manager = (context or {}).get("manager")
            if manager is not None:
                from phase_agent.evidence.perturbation_recovery import (
                    build_perturbation_recovery_pool,
                )

                try:
                    evidence["perturbation_parent_pool"] = build_perturbation_recovery_pool(
                        current, manager.data, max_det_H=6
                    )
                except (ValueError, OSError, KeyError) as error:
                    evidence["perturbation_parent_pool"] = {
                        "status": "unavailable",
                        "reason": str(error),
                    }
        action = propose(
            decision_state,
            agent_client=agent_client,
            allowed_tools=["prepare_local_batch_files"],
            config=config,
        )
        params = action.get("parameters") or {}
        direct = (
            action.get("operation") == "sample_direct_extrapolation"
            and params.get("mode") == "direct_extrapolation"
            and params.get("recovery_route", "extrapolation")
            == ("perturbation" if batch else "extrapolation")
        )
        perturbation = not batch and (action.get("operation"), params.get("mode")) in {
            ("prepare_parent_first_recovery", "parent_first_recovery_inputs"),
            # Old saved proposals remain reviewable; their execution is disabled.
            ("prepare_perturbation_recovery", "perturbation_recovery_inputs"),
        }
        if (
            (policy["route"] == "extrapolation" and not direct)
            or (policy["route"] == "perturbation" and not perturbation and not (batch and direct))
            or action.get("tool") != "prepare_local_batch_files"
            or params.get("scope_id") != required["scope_id"]
            or params.get("round_evidence_digest") != required["evidence_digest"]
        ):
            return {
                "status": "not_configured",
                "state": current,
                "submitted": False,
                "reason": "Strategic 尚未确认与可靠性证据绑定的外推恢复方案。",
            }
        if not batch:
            action = deepcopy(action)
            action.setdefault("parameters", {})["extrapolation_policy_version"] = "epoch_refresh_v1"
        proposal = build_agent_proposal(action, decision_state, runtime_state=current)
        return {
            "_proposal_selected": True,
            "current": current,
            "decision_state": decision_state,
            "proposal": proposal,
            "record_id": record_id_factory(current, invocation_id),
        }
    recovery = current.get("extrapolation_recovery_cycle") or {}
    if recovery.get("status") == "awaiting_reviewed_dft_labels":
        from phase_agent.evidence.extrapolation_recovery import recovery_context

        facts = recovery_context(current)
        decision_state.setdefault("decision_context", {})["extrapolation_recovery"] = facts
        jobs = (current.get("remote_finetune_jobs") or {}).values()
        label_ids = facts["labels"].get("label_task_ids") or []

        def covers_recovery(job):
            if job.get("original_model_version") != recovery.get("invalid_model_version"):
                return False
            if job.get("status") in {
                "inputs_prepared",
                "awaiting_remote_training",
                "running",
                "submitted",
            }:
                return True  # Existing in-flight training must be reconciled, not duplicated.
            data_ids = ((job.get("report") or {}).get("dataset_scope") or {}).get("data_ids") or []
            return bool(label_ids) and all(
                any(str(value).startswith(tid + ":frame:") for value in data_ids)
                for tid in label_ids
            )

        training_exists = any(covers_recovery(job) for job in jobs)
        if facts["labels"]["status"] != "complete" and recovery.get("recovery_approval_record_id"):
            from hashlib import sha256
            from pathlib import Path

            expected = {r.get("structure_sha256") for r in recovery.get("selected") or []}
            matching = []
            for task in current.get("tasks") or []:
                if task.get("stage") not in {"dft_single_point", "dft_relax"}:
                    continue
                try:
                    digest = sha256(Path(task["structure_path"]).read_bytes()).hexdigest()
                except (KeyError, OSError, TypeError):
                    continue
                if digest in expected:
                    matching.append(task)
            if matching:
                return {
                    "status": "waiting_results",
                    "state": current,
                    "submitted": False,
                    "reason": "恢复选中帧的 DFT 任务已登记；等待有效结果或处理失败任务，不重复创建。",
                }
            if (
                not callable(agent_client)
                or not callable(propose)
                or "select_dft_candidates" not in allowed
            ):
                return {
                    "status": "not_configured",
                    "state": current,
                    "submitted": False,
                    "reason": "等待 Strategic 为恢复选中帧提出 DFT 计算方式。",
                }
            decision_state.setdefault("decision_context", {})["recovery_dft_instruction"] = (
                "对本轮 DIRECT 的全部选中帧提出 DFT 单点或弛豫计算；不得加入其他帧，不追加 GS。"
                "此范围属于已批准的外推恢复路线。"
            )
            action = propose(
                decision_state,
                agent_client=agent_client,
                allowed_tools=["select_dft_candidates"],
                config=config,
            )
            decisions = (action.get("parameters") or {}).get("decisions") or []
            expected_ids = {r.get("candidate_id") for r in recovery.get("selected") or []}
            if (
                action.get("tool") != "select_dft_candidates"
                or not isinstance(decisions, list)
                or len(decisions) != len(expected_ids)
                or any(not isinstance(r, dict) for r in decisions)
                or {r.get("candidate_id") for r in decisions} != expected_ids
                or any(r.get("action") not in {"DFT_SINGLE_POINT", "DFT_RELAX"} for r in decisions)
            ):
                return {
                    "status": "not_configured",
                    "state": current,
                    "submitted": False,
                    "reason": "Strategic 的 DFT 方案未精确覆盖恢复选中帧；未创建任务。",
                }
            from phase_agent.tools.workflows.prepare_dft_proposal import prepare_dft_proposal

            preparation = prepare_dft_proposal(
                action,
                current,
                decision_state,
                (context or {}).get("effective_config") or config,
                agent_client=None,
                revise=revise,
            )
            if preparation.status == "rejected":
                return {
                    "status": "rejected",
                    "state": preparation.state,
                    "reason": preparation.error,
                }
            proposal = build_agent_proposal(
                preparation.action, decision_state, runtime_state=preparation.state
            )
            return {
                "_proposal_selected": True,
                "current": preparation.state,
                "decision_state": decision_state,
                "proposal": proposal,
                "record_id": record_id_factory(preparation.state, invocation_id),
            }
        if facts["labels"]["status"] == "complete" and not training_exists:
            if not callable(agent_client) or not callable(propose) or "update_mlip" not in allowed:
                return {
                    "status": "not_configured",
                    "state": current,
                    "submitted": False,
                    "reason": "DIRECT 选中结构的 DFT 标签已完成，等待 Strategic 提出微调方案。",
                }
            action = propose(
                decision_state,
                agent_client=agent_client,
                allowed_tools=["update_mlip"],
                config=config,
            )
            params = action.get("parameters") or {}
            if (
                action.get("operation") != "prepare_mlip_training"
                or action.get("tool") != "update_mlip"
                or params.get("action") != "RETRAIN_MLIP"
                or params.get("prepare_inputs_only") is not True
            ):
                return {
                    "status": "not_configured",
                    "state": current,
                    "submitted": False,
                    "reason": "等待 Strategic 确认 DIRECT 补数后的微调方案；未启动其他搜索。",
                }
            proposal = build_agent_proposal(action, decision_state, runtime_state=current)
            return {
                "_proposal_selected": True,
                "current": current,
                "decision_state": decision_state,
                "proposal": proposal,
                "record_id": record_id_factory(current, invocation_id),
            }
    post_recovery_only = bool(
        current.get("post_recovery_next_step_pending") or current.get("_recovery_just_approved")
    )
    if post_recovery_only:
        # This is the final proposal boundary after result recovery.  The
        # dialogue layer may enter the scientific workflow with `advance`, so
        # enforce the phase transition again where the action is actually
        # selected. Use the recovered stage's next-step scope, including the
        # full post-DFT review. New scientific work still requires approval.
        from phase_agent.policies.recovery_transition import post_recovery_tools

        stage_tools = post_recovery_tools(current)
        allowed = [tool for tool in allowed if tool in stage_tools]
        if not allowed:
            return {
                "status": "not_configured",
                "state": current,
                "reason": "回收后缺少本轮阶段可用的科学建议动作；未生成 Relax/MC 方案或执行新任务。",
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
    from phase_agent.policies.branch_conditions import branch_facts
    from phase_agent.runtime.turn_process import process_event

    facts = branch_facts(current)
    generation = facts["new_model_generation"]
    generation_required = generation["required"] and not generation["ready"]
    decision_state.setdefault("decision_context", {})["branch_conditions"] = facts
    # Saved snapshots can predate a code update without any scientific change.
    # Refresh current execution facts at the actual model boundary as well.
    from phase_agent.evidence.workflow_blockers import (
        mc_allocation_facts,
        refresh_result_facts,
    )

    decision_state["decision_context"]["mc_entry_conditions"] = mc_allocation_facts(current)
    decision_state["decision_context"]["refresh_result_facts"] = refresh_result_facts(current)
    from phase_agent.evidence.task_failure_facts import task_failure_facts

    decision_state["decision_context"]["retryable_tasks"] = [
        {key: row.get(key) for key in ("task_id", "stage", "status")}
        for row in task_failure_facts(current)["actionable_failures"]
        if row.get("task_id")
    ]
    from phase_agent.evidence.workflow_blockers import convergence_feedback

    decision_state["decision_context"]["latest_convergence_check"] = convergence_feedback(current)
    process_event("科学分支进入条件", facts)
    unified = bool((context or {}).get("unified_dialogue"))
    model_owned = callable(agent_client)
    if unified:
        decision_state.setdefault("decision_context", {})["unified_dialogue"] = True
        # This project dispatches HPC inputs, never local scientific calculations.
        allowed = [name for name in allowed if name != "run_calculation_stage"]
    from phase_agent.agents.strategic.choose_debug_next_action import (
        choose_debug_next_action,
    )

    mc_files_pending = any(
        row.get("stage") == "deep_search"
        and row.get("status") == "pending"
        and not row.get("slurm_batch_id")
        for row in current.get("tasks") or []
    )
    from phase_agent.science.mc.second_round_state import has_relevant_mc_tasks

    has_mc_tasks = has_relevant_mc_tasks(current)
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
    from phase_agent.evidence.post_dft_assessment import post_dft_assessment

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
    has_unsearched_new_gen = _has_unsearched_latest_generation(current, version, context)
    post_mc = (
        second_round_completed(current, version)
        and not assessment
        and mc_ids != consumed_mc
        and not generation_required
        and not has_unsearched_new_gen
    )
    allow_hull_branching = bool(
        effective.get("workflow", {}).get("allow_post_mc_hull_branching", True)
        and not getattr(agent_client, "_frozen_dft_action", None)
    )
    max_post_mc_continuations = int(
        effective.get("workflow", {}).get("max_post_mc_continuations", 2)
    )
    gen_under_model = [
        h
        for h in current.get("generation_history") or []
        if h.get("model_version") in (None, version)
    ]
    dft_under_model = [
        r for r in current.get("dft_dataset_records") or [] if r.get("model_version") == version
    ]
    continuation_cycles_without_dft = max(0, len(gen_under_model) - 1) if not dft_under_model else 0
    continuation_limit_reached = (
        post_mc and continuation_cycles_without_dft >= max_post_mc_continuations
    )
    if post_mc or assessment:
        from phase_agent.science.qbc.post_mc_candidates import post_mc_candidates

        current["qbc_candidates"] = post_mc_candidates(current, version)
        if assessment:
            current["qbc_candidates"] = _post_dft_candidate_pool(
                current["qbc_candidates"],
                current,
                frozen_dft_action=bool(getattr(agent_client, "_frozen_dft_action", None)),
            )
        if (
            not current["qbc_candidates"]
            and not assessment
            and not (post_recovery_only and "generate_branches" in allowed)
            and not (
                allow_hull_branching
                and not continuation_limit_reached
                and "generate_branches" in allowed
            )
        ):
            return {
                "status": "not_configured",
                "state": current,
                "reason": "第二轮 MC 已完成；DFT 候选缺少当前相图中唯一匹配的已识别结构与 Ehull/atom。未追加 MC 或回退 Relax。",
            }
    if "select_dft_candidates" in allowed and "qbc_candidates" in current:
        from phase_agent.tools.workflows.dft_proposal_evidence import refresh_dft_evidence

        refresh_dft_evidence(current, decision_state, effective)
    remote_sampling = current.get("remote_dft_sampling") or {}
    accepted_sampling = [
        row
        for row in remote_sampling.values()
        if isinstance(row, dict) and row.get("status") == "accepted" and row.get("selected")
    ]
    if accepted_sampling and "select_dft_candidates" in allowed and not assessment:
        latest_rs = accepted_sampling[-1]
        decision_state.setdefault("decision_context", {})["remote_dft_sampling_ready"] = {
            "bundle_id": latest_rs.get("bundle_id"),
            "selected_count": latest_rs.get("selected_count"),
            "gs_count": latest_rs.get("gs_count"),
            "kmeans_count": latest_rs.get("kmeans_count"),
            "instruction": "远端采样与验收已完成；若无额外阻塞，可直接基于已验收的 remote_dft_sampling 提出 select_dft_candidates。",
        }
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
        awaiting_first_search = (
            generation.get("post_refresh")
            and generation.get("ready")
            and not generation.get("first_search_completed")
        )
        allowed = _post_dft_allowed_tools(allowed, awaiting_first_search=awaiting_first_search)
    # A completed MC round normally moves the next proposal to DFT/convergence
    # review.  Keep Relax preparation available while a refreshed model still
    # has an unsearched first-generation branch set.
    gated_tools = _post_mc_allowed_tools(
        allowed,
        generation,
        post_mc=post_mc and not post_recovery_only,
        allow_continue_exploration=allow_hull_branching and not continuation_limit_reached,
    )
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
        from phase_agent.policies.region_sampling import ledger_context

        decision_state.setdefault("decision_context", {})["branch_ledger_facts"] = ledger_context(
            context["manager"]
        )
        from phase_agent.agents.exploration.branch_pool_review import pool_context

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

    from phase_agent.evidence.scientific_decision_brief import decision_brief

    decision_state.setdefault("decision_context", {})["scientific_decision_brief"] = decision_brief(
        current
    )
    from phase_agent.evidence.parent_first_recovery import initial_perturbation_direct_ready
    from phase_agent.tools.workflows.model_refresh_state import refresh_active

    manager_data = (
        getattr((context or {}).get("manager"), "data", None)
        if (context or {}).get("manager") is not None
        else None
    )
    if initial_perturbation_direct_ready(current, manager_data, allowed=allowed):
        decision_state.setdefault("decision_context", {})["initial_perturbation_direct_ready"] = (
            True
        )
    if (
        not safe_next
        and not direction
        and not generation_required
        and not assessment
        and not refresh_active(current)
        and not training_pending_validation(current)
        and not (unified and (context or {}).get("user_message") and not continue_requested)
    ):
        safe_next = _deterministic_cold_start_action(current, effective, allowed, context=context)
    if (
        (
            (
                generation.get("post_refresh")
                and generation.get("ready")
                and not generation.get("first_search_completed")
            )
            or has_unsearched_new_gen
        )
        and not assessment
        and not post_mc
    ):
        decision_state.setdefault("decision_context", {})["skip_reliability_before_first_mc"] = True
        if (
            not safe_next
            and not direction
            and not generation_required
            and not refresh_active(current)
            and not training_pending_validation(current)
            and not (unified and (context or {}).get("user_message") and not continue_requested)
        ):
            safe_next = _deterministic_post_generation_relax_mc_action(
                current, decision_state, effective, allowed, context=context
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
        if (
            "generate_branches" in allowed
            and effective.get("workflow", {}).get("direct_generation_without_llm", True)
            and not (unified and (context or {}).get("user_message") and not continue_requested)
        ):
            safe_next = _deterministic_generation_action(decision_state, effective, current=current)
    if (
        post_mc
        and not assessment
        and not safe_next
        and not direction
        and not generation_required
        and allow_hull_branching
        and not (unified and (context or {}).get("user_message") and not continue_requested)
    ):
        matrix = (decision_state.get("decision_context") or {}).get("branch_pool_facts", {}).get(
            "coverage_and_marginal_matrix"
        ) or []
        unexplored_count = sum(1 for row in matrix if row.get("exploration_status") == "unexplored")
        high_gain_count = sum(1 for row in matrix if row.get("exploration_status") == "high_gain")
        saturated_count = sum(1 for row in matrix if row.get("exploration_status") == "saturated")
        near_hull_count = sum(
            1
            for row in current.get("qbc_candidates") or []
            if isinstance(row, dict)
            and isinstance(row.get("ehull"), (int, float))
            and row["ehull"] <= 0.05
        )
        has_remote_dft = bool(
            (decision_state.get("decision_context") or {}).get("remote_dft_sampling_ready")
        )
        has_qbc = bool(current.get("qbc_candidates"))
        decision_state.setdefault("decision_context", {})["post_mc_hull_decision_facts"] = {
            "unexplored_count": unexplored_count,
            "high_gain_count": high_gain_count,
            "saturated_count": saturated_count,
            "has_remote_dft_sampling": has_remote_dft,
            "qbc_candidate_count": len(current.get("qbc_candidates") or []),
            "near_hull_candidate_count": near_hull_count,
            "continuation_cycles_without_dft": continuation_cycles_without_dft,
            "max_post_mc_continuations": max_post_mc_continuations,
            "continuation_limit_reached": continuation_limit_reached,
            "arbitration_policy": "urgent_dft > continue_exploration > normal_dft",
            "decision_guidance": (
                "Exploration只评估凸包轮廓是否仍需补探索及目标区域；Reliability只评估是否属于urgent_dft（基态反转/外推风险）或normal_dft；"
                "Strategic按urgent_dft>continue_exploration>normal_dft一轮定音；连续补探索达上限后直接进select_dft_candidates。"
            ),
        }
        if (
            not continuation_limit_reached
            and (
                (unexplored_count > 0 and not has_remote_dft)
                or (not has_qbc and not has_remote_dft)
            )
            and "generate_branches" in allowed
        ):
            safe_next = _deterministic_generation_action(decision_state, effective, current=current)
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
    from phase_agent.agents.common.dialogue_contract import (
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
    from phase_agent.policies.workflow_policy import initial_direction_errors

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
    from phase_agent.tools.local.prepare_mc_energy_repair import (
        attach_preview as attach_energy_repair_preview,
    )

    try:
        action = attach_energy_repair_preview(
            action, current, context or {}, (context or {}).get("effective_config") or config
        )
    except (ValueError, OSError) as error:
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

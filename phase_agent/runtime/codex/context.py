"""Agent-equivalent, read-only decision context for Codex proposals."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from phase_agent.evidence.requests import project_summary, summary_payload
from phase_agent.graphs.dialogue.evidence import conversation_facts, progress_signature
from phase_agent.runtime.project_context import current_config_session, handler_project_context
from phase_agent.policies.recovery_transition import post_recovery_tools
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import read_json


_CONTEXT_TOPICS = (
    "mc",
    "dft",
    "tasks",
    "convergence",
    "budget",
    "history",
    "model",
    "configuration",
    "plans",
)

HPC_APPROVAL_POLICY = {
    "upload_requires_explicit_user_approval": True,
    "submit_requires_explicit_user_approval": True,
    "approval_scope": "具体清单；路径、文件、命令或资源变化需重新批准",
    "data_transfer": {
        "scope": "all_submissions_and_returns_including_codex_takeover",
        "required_data_only": True,
        "compressed_archive_required": True,
        "reuse_repository_packaging_and_collection": True,
        "inspect_archive_inventory_before_transfer": True,
        "required_inventory": [
            "path",
            "purpose",
            "file_count",
            "compressed_bytes",
            "uncompressed_bytes",
            "sha256",
        ],
        "remote_only": [
            "raw_logs",
            "trajectories",
            "unreferenced_pool_structures",
            "models_and_large_checkpoints_unneeded_locally",
        ],
        "preserve_required": "保留本地校验、分析、训练和已验证续跑协议实际读取的数据与身份标记；不能只按后缀删减科学数据。",
        "legacy_archive": "旧包也先核对清单；含非必需文件时复用现有裁剪/导出入口，不整目录下载，不以已压缩为由整包回收。入口不支持时先报告缺口。",
        "approval_does_not_waive_minimization": True,
        "preserve_remote_results_and_do_not_recompute": True,
    },
    "required_manifest": [
        "local_task_path_and_files",
        "server",
        "remote_upload_path",
        "remote_workdir",
        "submit_command",
        "resources_and_environment",
        "output_path",
        "overwrite_or_reuse",
    ],
    "execution_authority": "Codex/MCP 只能准备和核验；不得自行上传、提交、取消、删除或覆盖 HPC 任务",
}


def _json_digest(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_codex_rule_audit(action, state, decision_context):
    """Attach a deterministic, read-only audit of the scientific selection rules."""
    tool = action.get("tool") or action.get("action_type")
    if tool == "generate_branches":
        params = action.get("parameters") or {}
        errors = []
        if params.get("sampling_mode") != "region_adaptive":
            errors.append("sampling_mode")
        for key in ("quota", "quotas", "total_quota", "batch_size", "generation_plan"):
            if key in params:
                errors.append(key)
        regions = params.get("regions")
        if regions is not None and not isinstance(regions, list):
            errors.append("regions")
        from phase_agent.policies.initial_coverage import review
        from phase_agent.policies.region_sampling import errors as region_errors
        from phase_agent.agents.exploration.generation_plan import (
            configured_generation_strategies,
            disabled_generation_allocations,
        )

        errors.extend(region_errors(params))
        config = state.get("confirmed_config") or {}
        errors.extend(
            "disabled_strategy:" + name
            for name in disabled_generation_allocations(
                params, configured_generation_strategies(config)
            )
        )
        initial_errors, initial = review(params, decision_context.get("initial_coverage") or {})
        errors.extend(initial_errors)
        return {
            "initial_coverage": initial,
            "status": "audited" if not errors else "invalid",
            "errors": errors,
            "sampling_mode": params.get("sampling_mode"),
            "region_count": len(regions or []),
            "evidence_sources": ["branch_pool", "coverage", "n novelty", "resource_guard"],
        }
    if tool != "select_dft_candidates":
        return {"status": "audited", "tool": tool, "evidence_sources": ["current_state", "policy"]}
    candidates = list(state.get("qbc_candidates") or [])
    by_id = {row.get("candidate_id"): row for row in candidates}
    decisions = (action.get("parameters") or {}).get("decisions") or []
    selected = [
        row.get("candidate_id")
        for row in decisions
        if row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    ]
    audit_ids = set(state.get("dft_audit_candidate_ids") or [])
    dh_groups = {}
    for row in candidates:
        evidence = row.get("dft_sampling_evidence") or {}
        group = (
            evidence.get("D_candidate_id") or ("H:" + str(evidence.get("geometry_key")))
            if ((evidence.get("D_candidate_id")) or ((evidence.get("H_single_max") or 0) > 1e-8))
            else None
        )
        if group:
            dh_groups.setdefault(group, []).append(row.get("candidate_id"))
    ranked = []
    try:
        from phase_agent.agents.reliability.selection.select_dft_candidates import (
            select_dft_candidates,
        )

        rows = []
        for row in candidates:
            item = dict(row)
            signal = row.get("phase_quality_signal") or {}
            item["hull_impact"] = row.get(
                "hull_impact", signal.get("distance_to_current_hull_mev_per_O2", 0) or 0
            )
            rows.append(item)
        ranked = [
            row.get("candidate_id")
            for row in select_dft_candidates(
                rows, batch_size=max(1, len(selected)), seed=0, cover_phases=True
            )["selected_candidates"]
        ]
    except (KeyError, TypeError, ValueError):
        ranked = []
    return {
        "status": "audited",
        "selected_count": len(selected),
        "selected_ids": selected,
        "unknown_selected_ids": [cid for cid in selected if cid not in by_id],
        "independent_audit_selected": [cid for cid in selected if cid in audit_ids],
        "policy_ranked_candidate_ids": ranked,
        "policy_overlap_count": len(set(selected) & set(ranked)),
        "d_h_group_count": len(dh_groups),
        "d_h_groups_with_decision": sum(
            any(
                cid in selected
                or any(
                    d.get("candidate_id") == cid and d.get("action") in {"DEFER", "REJECT"}
                    for d in decisions
                )
                for cid in ids
            )
            for ids in dh_groups.values()
        ),
        "evidence_sources": ["ehull", "qbc", "diversity", "coverage", "d_h", "independent_audit"],
        "rule_note": "D/H and QBC are diagnostic evidence; they are not numerically summed into a physical error probability.",
    }


def ensure_codex_round_budget_review(action, decision_context):
    """Keep Codex proposals inside the Agent's mandatory budget-review contract.

    The Agent model normally supplies this explanation itself.  Codex supplies a
    structured action, so fill only the missing, non-execution explanation from
    the action's declared budget; all scientific and approval validation remains
    in the existing workflow.
    """
    if not (decision_context.get("round_budget_evidence") or {}).get("required"):
        return action
    if action.get("round_budget_review") is not None:
        return action
    tool = action.get("tool") or action.get("action_type")
    if tool not in {"select_dft_candidates", "generate_branches"}:
        return action
    choice = "supplement_dft" if tool == "select_dft_candidates" else "new_search"
    budget = action.get("budget")
    estimated = budget if type(budget) in (int, float) and budget >= 0 else None
    selected_benefit = (
        "补充当前候选的独立校准证据" if choice == "supplement_dft" else "扩展当前模型的结构探索覆盖"
    )
    alternative_benefit = (
        "扩展当前模型的结构探索覆盖" if choice == "supplement_dft" else "补充当前候选的独立校准证据"
    )
    action = deepcopy(action)
    action["round_budget_review"] = {
        "choice": choice,
        "reason": selected_benefit + "；先完成本次审阅动作。",
        "uncertainty": "两条路径的下游收益仍需新结果验证。",
        "revisit_when": "本次结果回收并更新相图后重新比较两条路径。",
        "alternatives": [
            {
                "path": "supplement_dft",
                "expected_benefit": "补充近凸包与独立代表的校准证据。",
                "benefit_status": "hypothesis",
                "expected_relative_cost": estimated if choice == "supplement_dft" else None,
                "cost_basis": "按当前动作 budget 记录；完整下游收益未知。",
            },
            {
                "path": "new_search",
                "expected_benefit": "扩展当前模型的结构与成分覆盖。",
                "benefit_status": "hypothesis",
                "expected_relative_cost": estimated if choice == "new_search" else None,
                "cost_basis": "需要重新生成并回收 Relax/MC，当前未估算完整下游成本。",
            },
        ],
    }
    return action


def ensure_codex_post_dft_review(action, decision_context):
    """Provide the required explanatory post-DFT review without an LLM retry."""
    if action.get("post_dft_review") is not None:
        return action
    if not (
        decision_context.get("round_budget_review_required")
        or decision_context.get("post_dft_assessment")
    ):
        return action
    tool = action.get("tool") or action.get("action_type")
    choices = {
        "generate_branches": "search",
        "select_dft_candidates": "supplement_dft",
        "restart_failed_task": "retry_failed",
        "update_mlip": "finetune",
        "adjust_strategy": "revise_strategy",
        "check_convergence": "convergence",
        "pause_search": "stop",
    }
    choice = choices.get(tool)
    if choice is None:
        return action
    action = deepcopy(action)
    action["post_dft_review"] = {
        "choice": choice,
        "stop_status": "continue" if choice != "stop" else "blocked",
        "finetune_recommendation": "defer" if choice != "finetune" else "now",
        "error_assessment": "当前模型误差与独立校准证据仍需新结果验证。",
        "coverage_assessment": "当前候选池已形成，但覆盖与代表性仍有未知项。",
        "finetune_assessment": "先完成本次证据补充，再依据新增标签评估微调。",
        "search_assessment": "已有当前模型搜索结果；是否扩展搜索由本次审阅动作决定。",
        "reference_assessment": "依据当前模型相图、回收结果和候选池快照。",
        "round_findings": "回收结果已登记，下一步只推进已结构化动作。",
        "limitations": "尚未观察到新动作的真实 DFT 结果或下游收益。",
        "dft_assessment": "本动作用于补充当前候选的独立校准证据。",
        "convergence_assessment": "当前证据不足以宣称科学收敛。",
    }
    return action


def _allowed_tools(state, session):
    config = (session.get("confirmed_snapshot") or {}).get("config") or session.get("config") or {}
    allowed = [
        name
        for name in (config.get("agent") or {}).get("allowed_tools") or []
        if name != "run_calculation_stage"
    ]
    stage_tools = post_recovery_tools(state)
    if stage_tools:
        allowed = [name for name in allowed if name in stage_tools]
    return allowed


def build_codex_context(handler) -> dict:
    """Build a serializable snapshot from the same facts used by Agent dialogue.

    This function is read-only. It does not call a model, write state, create an
    approval, or execute a scientific operation. The returned context version
    binds a later structured proposal to this exact project observation.
    """
    state = read_json(handler.state_path, {}) or {}
    session = current_config_session(handler)
    project = handler_project_context(handler, state)
    summary = conversation_facts(state, project)
    summary["current_project"] = project_summary(summary["current_project"])
    refresh = summary.get("refresh_result_facts") or {}
    summary["refresh_result_facts"] = {
        key: deepcopy(value)
        for key, value in refresh.items()
        if not isinstance(value, (dict, list))
    }
    # Keep the context endpoint bounded: the Agent's dialogue already stores
    # the prepared evidence in state, while rebuilding every historical index
    # here can block a local control request for minutes on large Cr runs.
    evidence_keys = {
        "mc_entry_conditions",
        "available_branches",
        "current_phase_diagram",
        "branch_pool_facts",
        "qbc_candidates",
        "post_dft_assessment",
        "training_excluded_data_ids",
        "dft_magnetic_diagnostics",
        "task_stage_evidence",
        "task_round_evidence",
        "retryable_tasks",
        "latest_convergence_check",
        "round_budget_evidence",
        "calculation_cost_reference",
        "recent_experience",
        "decision_feedback",
        "long_term_memory",
        "relevant_approved_knowledge",
        "model_refresh",
        "model_round_summary",
        "training_validation_planning",
        "post_refresh_search",
        "new_model_generation",
        "post_dft_assessment",
        "confirmed_configuration",
        "active_model",
        "convergence_rules",
    }
    evidence = {key: deepcopy(state.get(key)) for key in evidence_keys if key in state}
    from phase_agent.persistence.memory.decision_context import decision_memory_context

    memory_context = decision_memory_context(state)
    evidence["memory_context"] = memory_context
    evidence["relevant_approved_knowledge"] = memory_context["approved_knowledge"]
    evidence.setdefault("refresh_result_facts", deepcopy(summary.get("refresh_result_facts") or {}))
    from phase_agent.agents.exploration.branch_pool_review import (
        INSTRUCTION as BRANCH_REVIEW_INSTRUCTION,
    )
    from phase_agent.agents.exploration.branch_pool_review import pool_context

    manager = handler.workflow_kwargs.get("manager")
    branches = manager.data["branches"] if manager is not None else None
    branch_pool = pool_context(state, branches)
    if branches is not None:
        summary.setdefault("pool_counts", {})["branches"] = len(branches)
    evidence["branch_pool_facts"] = branch_pool
    from phase_agent.evidence.project_authority import AUTHORITY_RULE

    decision_context = {
        **evidence,
        "unified_dialogue": True,
        "conversation_controls": True,
        "current_settings_authority": AUTHORITY_RULE,
        "saved_facts_notice": "Codex 只能基于本快照形成方案；方案仍需原审批。",
        "context_topics": list(_CONTEXT_TOPICS),
    }
    allowed = _allowed_tools(state, session)
    payload_state = {
        **summary,
        **{key: deepcopy(state[key]) for key in evidence_keys if key in state},
    }
    payload = {
        "mode": "autonomous_search",
        "state": payload_state,
        "decision_context": decision_context,
        "allowed_tools": allowed,
    }
    prepared, _sources = summary_payload(payload)
    from phase_agent.tools.dispatch.scientific_operations import with_operation_contracts

    prepared = with_operation_contracts(prepared)
    from phase_agent.evidence.decision_evidence_catalog import decision_evidence_catalog

    references = deepcopy(decision_evidence_catalog(decision_context).get("references") or {})
    prepared.setdefault("decision_context", {})["evidence_reference_index"] = {
        "references": references,
        "instruction": "Use only these exact references; they identify the read-only snapshot and do not grant execution authority.",
    }
    persisted_refs = (
        ((state.get("current_state_snapshot") or {}).get("decision_context") or {})
        .get("evidence_catalog", {})
        .get("references", {})
    )
    prepared["decision_context"]["evidence_reference_index"]["references"].update(
        {key: {"kind": row.get("kind", "persisted")} for key, row in persisted_refs.items()}
    )
    prepared["decision_context"].setdefault("evidence_catalog", {})["references"] = deepcopy(
        prepared["decision_context"]["evidence_reference_index"]["references"]
    )
    prepared["decision_context"]["branch_pool_facts"] = branch_pool
    from phase_agent.policies.initial_coverage import contract

    cfg = (session.get("confirmed_snapshot") or {}).get("config") or session.get("config") or {}
    from phase_agent.agents.permissions import ROLES, role_permissions
    from phase_agent.agents.strategic.human_review import authorization_settings

    prepared["decision_context"]["agent_permissions"] = {
        role: role_permissions(role, registered_tools=allowed) for role in ROLES
    }
    confirmed_policy = (
        ((session.get("confirmed_snapshot") or {}).get("config") or {})
        if session.get("status") == "confirmed"
        else {}
    )
    prepared["decision_context"]["execution_authorization_policy"] = {
        "source": "confirmed_config" if session.get("status") == "confirmed" else "none",
        **authorization_settings(confirmed_policy),
    }
    initial = contract(manager, handler.workflow_kwargs.get("phase_references") or {}, cfg)
    prepared["decision_context"]["initial_coverage"] = initial
    prepared["decision_context"]["branch_pool_review_contract"] = BRANCH_REVIEW_INSTRUCTION
    prepared["decision_context"]["branch_pool_review_required"] = bool(
        branch_pool.get("existing_branches") or branch_pool.get("registered_new_branch_count")
    )
    # Build the same derived evidence used by the normal Agent proposal path.
    # The durable state often does not persist this derived object verbatim,
    # while post-DFT review still requires the two-path budget comparison.
    from phase_agent.evidence.round_budget_evidence import round_budget_evidence

    budget_evidence = round_budget_evidence(state)
    prepared["decision_context"]["round_budget_review_required"] = bool(
        budget_evidence.get("required")
    )
    prepared["decision_context"]["round_budget_evidence"] = deepcopy(budget_evidence)
    prepared["required_contracts"] = {
        "branch_pool_review": prepared["decision_context"].get("branch_pool_review_required"),
        "round_budget_review": prepared["decision_context"].get("round_budget_review_required"),
        "generate_branches": {
            "sampling_mode": "region_adaptive",
            "initial_coverage": initial,
            "regions": "required when supplementing the registered pool",
            "relax_only_x_values": "include 0 for Na=0 Relax-only groups",
        },
        "scientific_decision": "required for generate_branches/select_dft_candidates/update_mlip/check_convergence/pause_search; use round_direction for model updates and post-DFT decisions, otherwise branch_sampling or dft_sampling",
    }
    from phase_agent.policies.model_generation import new_model_generation_facts

    generation = new_model_generation_facts(state)
    prepared["decision_context"]["new_model_generation"] = generation
    if generation["required"] and not generation["ready"]:
        prepared["required_next_tool"] = "generate_branches"
        prepared["required_next_reason"] = (
            "当前模型刷新已完成，但没有发现与当前模型绑定的新branch登记；"
            "必须先生成新branch，再进入Relax/MC或DFT选择。"
        )
    else:
        prepared["required_next_tool"] = None
        prepared["required_next_reason"] = None
    context_version = progress_signature(state)
    state_version = build_status_summary(
        state, config_version=state.get("confirmed_config_version")
    )["summary_id"]
    from phase_agent.runtime.codex.team import native_team_workflow

    return {
        "codex_team_workflow": native_team_workflow(),
        "context_version": context_version,
        "state_version": state_version,
        "config_version": state.get("confirmed_config_version")
        or (state.get("confirmed_config") or {}).get("version")
        or (session.get("confirmed_snapshot") or {}).get("config_version"),
        "model_version": state.get("active_model_version")
        or (state.get("active_model") or {}).get("version"),
        "project": deepcopy(prepared.get("state", {}).get("current_project") or project),
        "state": prepared.get("state") or {},
        "decision_context": prepared.get("decision_context") or {},
        "allowed_tools": allowed,
        "scientific_operations": prepared.get("scientific_operations") or [],
        "module_operations": prepared.get("module_operations") or [],
        "operation_contract_rule": prepared.get("operation_contract_rule"),
        "required_next_tool": prepared.get("required_next_tool"),
        "required_next_reason": prepared.get("required_next_reason"),
        "required_contracts": prepared.get("required_contracts") or {},
        "hpc_approval_policy": deepcopy(HPC_APPROVAL_POLICY),
        "context_digest": _json_digest(
            {
                "context_version": context_version,
                "state_version": state_version,
                "allowed_tools": allowed,
                "scientific_operations": prepared.get("scientific_operations") or [],
                "module_operations": prepared.get("module_operations") or [],
                "decision_context": prepared.get("decision_context") or {},
            }
        ),
    }

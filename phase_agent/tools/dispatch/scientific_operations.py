"""Serializable operation contracts; route through existing approved tools only."""

from copy import deepcopy


def _spec(tool, purpose, *, parameters=None, stage=None, targets="structure_ids", outputs=()):
    return {
        "tool": tool,
        "purpose": purpose,
        "parameters": parameters or {},
        "stage": stage,
        "target_kind": targets,
        "outputs": list(outputs),
        "version_bindings": ["config_version", "model_version"],
        "execution": "existing_approval_validation_dispatch",
        "completion": "Inspect returned status and task results; prepared is not computed.",
    }


OPERATIONS = {
    "prepare_parent_first_recovery": _spec(
        "prepare_local_batch_files",
        "微扰结构微调：冻结母帧和成功 DFT 参考，HPC 先用原始 MH-1 弛豫，再微扰、合理性检查、相对弛豫母帧M3GNet特征差<=1筛选、DIRECT和恢复DFT；微扰后不再MLIP弛豫。必须绑定轮次、母帧、目标及 DFT 参数。",
        parameters={"mode": "parent_first_recovery_inputs"},
        targets="none",
        outputs=("parent_first_recovery_batch", "handoff"),
    ),
    "prepare_perturbation_recovery": _spec(
        "prepare_local_batch_files",
        "旧本地先微扰入口已停用；微扰结构微调应将母帧和成功 DFT 参考交给 HPC，先用原始 MH-1 弛豫，再微扰、DIRECT 和 DFT；新链路使用 prepare_parent_first_recovery",
        parameters={"mode": "perturbation_recovery_inputs"},
        targets="none",
        outputs=("perturbation_mh1_inputs",),
    ),
    "sample_direct_extrapolation": _spec(
        "prepare_local_batch_files",
        "仅用于历史冻结方案恢复；新恢复禁止选择此入口，使用 prepare_parent_first_recovery。外推风险证据保留供母帧和常规DFT选点。",
        parameters={"mode": "direct_extrapolation"},
        targets="none",
        outputs=("direct_sampling",),
    ),
    "prepare_training_handoff": _spec(
        "prepare_local_batch_files",
        "单独审批训练交接输入；parameters需job_key及kind=manifest或validation，不提交计算",
        parameters={"mode": "training_handoff_inputs"},
        targets="none",
        outputs=("handoffs",),
    ),
    "prepare_ordering_search": _spec(
        "prepare_local_batch_files",
        "调用Process_AL_MC规划并准备独立TM/Na+TM搜索；需phase、tm_ratios、na_content，以及max_det_H或h_matrices二选一，可传oxidation_states与options",
        parameters={"mode": "ordering_search_inputs"},
        targets="none",
        outputs=("ordering_search_preview", "input_archive", "workload"),
    ),
    "recover_ordering_search": _spec(
        "prepare_local_batch_files",
        "读取已登记独立搜索结果；parameters含plan_id和results_archive，外部回传必须是最小校验压缩包，结果不自动写入主相图或启动DFT",
        parameters={"mode": "ordering_search_results"},
        targets="none",
        outputs=("global_ranking_csv", "all_joint_csv", "diagnostics_csv"),
    ),
    "prepare_relax_mc_batch": _spec(
        "prepare_local_batch_files",
        "诊断并准备一次上传的Relax+最多三段MC封装",
        parameters={"mode": "relax_mc_inputs"},
        targets="branch_ids",
        outputs=("upload_archive", "workload", "batch_diagnosis"),
    ),
    "prepare_relax_mc_resume": _spec(
        "prepare_local_batch_files",
        "识别HPC启动/I/O故障并准备原封装缺失项补算交接；必须通过远端resume预检与原有提交审批，保留已完成结果",
        parameters={"mode": "relax_mc_resume"},
        targets="none",
        outputs=("recovery_plan", "next_action"),
    ),
    "recover_relax_mc_batch": _spec(
        "prepare_local_batch_files",
        "核验并回收已登记封装的结果与相图",
        parameters={"mode": "relax_mc_results"},
        targets="none",
        outputs=("result_summary", "candidate_pool", "phase_diagrams"),
    ),
    "generate_branches": _spec(
        "generate_branches",
        "生成并登记候选 branch",
        targets="optional_parent_branch_ids",
        outputs=("branch_ids", "structure_ids"),
    ),
    "prepare_mlip_relax": _spec(
        "prepare_local_batch_files",
        "准备 MLIP 弛豫输入",
        parameters={"mode": "relax_inputs"},
        stage="relax_and_feature",
        outputs=("task_ids", "upload_paths"),
    ),
    "allocate_mlip_mc": _spec(
        "allocate_mc_bohb",
        "筛选 branch 并分配 MC 任务",
        stage="deep_search",
        targets="branch_ids",
        outputs=("task_ids", "allocation"),
    ),
    "prepare_mlip_mc": _spec(
        "prepare_local_batch_files",
        "准备已分配的 MC 输入",
        parameters={"mode": "mc_inputs"},
        stage="deep_search",
        targets="all_pending_mc_tasks",
        outputs=("upload_paths",),
    ),
    "select_dft_candidates": _spec(
        "select_dft_candidates",
        "依据已有 QBC 与 Ehull 证据选择 DFT",
        outputs=("task_ids", "validated_selection"),
    ),
    "restart_failed_task": _spec(
        "restart_failed_task",
        "重试已登记失败任务；MC 阈值修复可批量绑定同轮已核验目标，保留历史并重新审批",
        targets="retryable_task_ids",
        outputs=("retry_task_id", "retry_input_path"),
    ),
    "prepare_dft_inputs": _spec(
        "prepare_local_batch_files",
        "准备已批准 DFT 任务输入",
        parameters={"mode": "dft_inputs"},
        targets="registered_task_ids",
        outputs=("upload_paths",),
    ),
    "prepare_mlip_training": _spec(
        "update_mlip",
        "使用合格 DFT 数据准备远端微调；可在已审核动作中明确选择验证通过后自动切换并单独审批刷新",
        parameters={"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
        targets="none",
        outputs=("training_handoff",),
    ),
    "activate_mlip_candidate": _spec(
        "update_mlip",
        "审批后激活已评估候选模型",
        parameters={"action": "ACTIVATE_CANDIDATE_MODEL"},
        targets="none",
        outputs=("active_model", "model_refresh"),
    ),
    "prepare_mc_energy_repair": _spec(
        "prepare_local_batch_files",
        "对已登记 MC 末态以同一主模型进行单点，修复相图能量口径；保留历史原结果",
        parameters={"mode": "mc_energy_repair_inputs"},
        stage="relax_and_feature",
        targets="task_ids",
        outputs=("energy_repair_preview", "upload_paths"),
    ),
    "prepare_model_refresh": _spec(
        "prepare_local_batch_files",
        "按已审核刷新方案准备新模型计算输入",
        parameters={"mode": "model_refresh_inputs"},
        stage="relax_and_feature",
        outputs=("task_ids", "refresh_plan"),
    ),
    "check_convergence": _spec(
        "check_convergence", "评估已有数据的收敛条件", targets="none", outputs=("convergence",)
    ),
}


def operation_catalog(allowed_tools=None, *, registry=None):
    from phase_agent.agents.permissions import advisory_roles

    allowed = set(allowed_tools) if allowed_tools is not None else None
    return [
        {
            "operation": name,
            **deepcopy(spec),
            "advisory_roles": list(advisory_roles(name)),
            "proposal_roles": ["strategic"],
            "execution_roles": ["simulation"],
        }
        for name, spec in OPERATIONS.items()
        if (allowed is None or spec["tool"] in allowed)
        and (registry is None or callable((registry.get(spec["tool"]) or {}).get("handler")))
    ]


# Existing new-proposal policy; historical frozen plans are not reinterpreted.
NEW_PROPOSAL_DISABLED = {"sample_direct_extrapolation", "prepare_perturbation_recovery"}


def new_operation_errors(action):
    current = action
    matches = [
        name
        for name in NEW_PROPOSAL_DISABLED
        if current.get("operation") == name
        or (
            (current.get("tool") or current.get("action_type")) == OPERATIONS[name]["tool"]
            and all(
                (current.get("parameters") or {}).get(k) == v
                for k, v in OPERATIONS[name]["parameters"].items()
            )
        )
    ]
    return ["new_recovery_requires_parent_first_perturbation"] if matches else []


def operation_readiness(facts):
    """Cheap known prerequisites, never full input verification or authorization."""
    context = facts.get("decision_context") or {}
    gate = context.get("mc_entry_conditions") or {}
    rows = [*(facts.get("scientific_operations") or []), *(facts.get("module_operations") or [])]
    result = {}
    for row in rows:
        name = row["operation"]
        reasons = new_operation_errors(row)
        state = "unsupported_new_proposal" if reasons else "unknown"
        sources = ["policies.execution_policy.build_agent_proposal"] if reasons else []
        if not reasons and name in {"allocate_mlip_mc", "prepare_mlip_mc"} and gate.get("required"):
            state = "known_prerequisites_passed" if gate.get("ready") else "waiting_dependency"
            reasons = (
                []
                if gate.get("ready")
                else ["MC requires recovered Relax results and current-model MLIP hull"]
            )
            sources = ["decision_context.mc_entry_conditions"]
        result[name] = {
            "registered": True,
            "project_readiness": state,
            "reasons": reasons,
            "sources": sources,
            "implementation": "not_verified_by_catalog",
            "input_readiness": "not_fully_verified",
            "authorization": "not_evaluated; existing approval policy still applies",
        }
    return result


def bind_operation(action):
    """Bind an optional explicit operation, rejecting ambiguous caller overrides."""
    name = action.get("operation")
    if name is None:
        return deepcopy(action)
    if not isinstance(name, str) or name not in OPERATIONS:
        raise ValueError("unknown_scientific_operation")
    spec = OPERATIONS[name]
    current = deepcopy(action)
    for key in ("tool", "action_type"):
        if current.get(key) not in (None, spec["tool"]):
            raise ValueError("operation_tool_conflict")
        current[key] = spec["tool"]
    if spec["stage"] is not None:
        if current.get("stage") not in (None, spec["stage"]):
            raise ValueError("operation_stage_conflict")
        current["stage"] = spec["stage"]
    if spec["target_kind"] == "all_pending_mc_tasks" and current.get("target_ids"):
        raise ValueError("operation_scope_conflict:all_pending_mc_tasks")
    parameters = deepcopy(current.get("parameters") or {})
    for key, value in spec["parameters"].items():
        if key in parameters and parameters[key] != value:
            raise ValueError("operation_parameter_conflict:" + key)
        parameters[key] = value
    current["parameters"] = parameters
    return current


def operation_binding_errors(action, *, config_version, model_version):
    bindings = action.get("operation_context")
    if bindings is None:
        return []
    if not isinstance(bindings, dict):
        return ["invalid_operation_context"]
    errors = []
    expected = {"config_version": config_version, "model_version": model_version}
    for key, value in bindings.items():
        if key == "relax_only_x_values":
            if (
                action.get("tool") or action.get("action_type")
            ) != "generate_branches" or value != [0]:
                errors.append("invalid_operation_binding:relax_only_x_values")
        elif key == "human_constraint":
            if not isinstance(value, str) or not value.strip():
                errors.append("invalid_operation_binding:human_constraint")
        elif key not in expected:
            errors.append("unknown_operation_binding:" + key)
        elif value is None or value != expected[key]:
            errors.append("operation_binding_mismatch:" + key)
    return errors


def with_operation_contracts(payload):
    current = dict(payload)
    if current.get("allowed_tools"):
        current["scientific_operations"] = operation_catalog(current["allowed_tools"])
        brief = (current.get("decision_context") or {}).get("scientific_decision_brief") or {}
        if (brief.get("policy") or {}).get("schema_version") == 1:
            decisions = {
                "generate_branches",
                "select_dft_candidates",
                "prepare_mlip_training",
                "activate_mlip_candidate",
                "check_convergence",
            }
            current["module_operations"] = [
                row for row in current["scientific_operations"] if row["operation"] not in decisions
            ]
            current["scientific_operations"] = [
                row for row in current["scientific_operations"] if row["operation"] in decisions
            ]
            current["scientific_decision_points"] = [
                "branch_sampling",
                "dft_sampling",
                "round_direction",
            ]
        current["operation_contract_rule"] = (
            "You may set operation to one listed scientific_operations name. Keep tool and "
            "fixed parameters consistent with its contract. This selects exactly one existing "
            "approved tool, never bypasses approval. Optional operation_context binds current "
            "config_version/model_version; do not invent versions. Preparation, waiting and "
            "dispatch completion are not proof of scientific result completion. Recovery and "
            "hull updates run in the existing observation nodes before the next decision."
        )
    return current

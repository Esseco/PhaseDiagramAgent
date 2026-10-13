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
    "prepare_ordering_search": _spec(
        "prepare_local_batch_files",
        "调用Process_AL_MC规划并准备独立TM/Na+TM搜索；需phase、tm_ratios、na_content，以及max_det_H或h_matrices二选一，可传oxidation_states与options",
        parameters={"mode": "ordering_search_inputs"},
        targets="none",
        outputs=("ordering_search_preview", "input_archive", "workload"),
    ),
    "recover_ordering_search": _spec(
        "prepare_local_batch_files",
        "读取已登记独立搜索结果；parameters含plan_id和可选results_directory，结果不自动写入主相图或启动DFT",
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
    "prepare_dft_inputs": _spec(
        "prepare_local_batch_files",
        "准备已批准 DFT 任务输入",
        parameters={"mode": "dft_inputs"},
        targets="registered_task_ids",
        outputs=("upload_paths",),
    ),
    "prepare_mlip_training": _spec(
        "update_mlip",
        "使用合格 DFT 数据准备远端微调",
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
    allowed = set(allowed_tools) if allowed_tools is not None else None
    return [
        {"operation": name, **deepcopy(spec)}
        for name, spec in OPERATIONS.items()
        if (allowed is None or spec["tool"] in allowed)
        and (registry is None or callable((registry.get(spec["tool"]) or {}).get("handler")))
    ]


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
        if key not in expected:
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

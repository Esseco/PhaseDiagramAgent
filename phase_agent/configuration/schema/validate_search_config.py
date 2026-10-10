"""Check missing, conflicts and ambiguity without launching work."""


def validate_search_config(config: dict, *, stage="full") -> dict:
    if stage not in {"full", "startup"}:
        raise ValueError("unknown_configuration_stage")
    missing, conflicts, ambiguities = [], [], []
    required = [
        "system",
        "frozen_parameters",
        "generation_actions",
        "calculation",
        "budgets",
        "dft",
        "convergence",
        "agent",
    ]
    missing.extend(name for name in required if name not in config)
    system = config.get("system") or {}
    if "boundary" in system:
        boundary = system.get("boundary") or {}
        if not boundary.get("P"):
            missing.append("system.boundary.P")
        if not boundary.get("TM_ratio"):
            missing.append("system.boundary.TM_ratio")
    if not system.get("constraints"):
        missing.append("system.constraints")
    if not system.get("branch_schema"):
        missing.append("system.branch_schema")
    from phase_agent.configuration.schema.validate_configuration_space import (
        validate_configuration_space,
    )

    conflicts.extend(validate_configuration_space(system)["errors"])
    calculation = config.get("calculation") or {}
    if not calculation.get("mlip_version"):
        missing.append("calculation.mlip_version")
    dft = config.get("dft") or {}
    if not dft.get("parameters") and dft.get("parameter_source") != "atomate_defaults":
        missing.append("dft.parameters")
    convergence = config.get("convergence") or {}
    if convergence.get("hull_change_tolerance", convergence.get("hull_tolerance")) is None:
        missing.append("convergence.hull_change_tolerance")
    if (
        convergence.get(
            "final_energy_mae_tolerance", convergence.get("final_frame_error_tolerance")
        )
        is None
    ):
        missing.append("convergence.final_energy_mae_tolerance")
    epochs = convergence.get("stable_model_update_epochs", convergence.get("stable_rounds"))
    if epochs is None:
        missing.append("convergence.stable_model_update_epochs")
    elif isinstance(epochs, bool) or int(epochs) <= 0:
        conflicts.append("convergence_stable_model_update_epochs_must_be_positive")
    for field in ("hull_change_tolerance", "final_energy_mae_tolerance"):
        value = convergence.get(
            field,
            convergence.get(
                {
                    "hull_change_tolerance": "hull_tolerance",
                    "final_energy_mae_tolerance": "final_frame_error_tolerance",
                }[field]
            ),
        )
        if value is not None and (isinstance(value, bool) or float(value) < 0):
            conflicts.append(f"convergence.{field}_must_be_nonnegative")
    mc = config.get("mc_policy")
    if mc is not None:
        tiers = mc.get("tiers") or []
        if not tiers:
            missing.append("mc_policy.tiers")
        for index, tier in enumerate(tiers):
            for field in ("max_mc_steps", "patience_steps", "min_improvement"):
                if tier.get(field) is None:
                    missing.append(f"mc_policy.tiers[{index}].{field}")
        if not 0 <= float(mc.get("random_exploration_fraction", 0)) <= 1:
            conflicts.append("mc_policy.random_exploration_fraction_invalid")
    finetune = config.get("mlip_finetune") or {}
    if finetune.get("enabled"):
        validation = finetune.get("validation") or {}
        for field in (
            "max_energy_mae",
            "max_critical_failure_fraction",
            "max_near_hull_ranking_reversals",
        ):
            if validation.get(field) is None:
                missing.append(f"mlip_finetune.validation.{field}")
        if validation.get("max_energy_mae") is not None and float(validation["max_energy_mae"]) < 0:
            conflicts.append("mlip_finetune.validation.max_energy_mae_must_be_nonnegative")
        failure_limit = validation.get("max_critical_failure_fraction")
        if failure_limit is not None and not 0 <= float(failure_limit) <= 1:
            conflicts.append("mlip_finetune.validation.max_critical_failure_fraction_invalid")
        reversals = validation.get("max_near_hull_ranking_reversals")
        if reversals is not None and (isinstance(reversals, bool) or int(reversals) < 0):
            conflicts.append("mlip_finetune.validation.max_near_hull_ranking_reversals_invalid")
        refresh = finetune.get("refresh_validation") or {}
        for field in ("max_failure_fraction", "max_near_hull_ranking_reversals"):
            if refresh.get(field) is None:
                missing.append(f"mlip_finetune.refresh_validation.{field}")
    allocator = calculation.get("mc_allocator")
    if allocator not in {"agent_tools", "bohb", "fixed"}:
        conflicts.append("calculation.mc_allocator_invalid")
    enabled = set((config.get("generation_actions") or {}).get("enabled") or [])
    known = set((system.get("generation") or {}).get("enabled_strategies") or [])
    if known and not enabled <= known:
        conflicts.append("generation_action_not_registered_for_system")
    frozen = config.get("frozen_parameters") or []
    if len(frozen) != len(set(frozen)):
        conflicts.append("duplicate_frozen_parameter")
    if (config.get("budgets") or {}).get("total_relative_cost") is None:
        ambiguities.append("budgets.total_relative_cost")
    deferred = []
    if stage == "startup":
        later = ("convergence", "mlip_finetune", "mc_policy", "budgets.total_relative_cost")
        deferred = sorted({item for item in missing + ambiguities if item.startswith(later)})
        missing = [item for item in missing if item not in deferred]
        ambiguities = [item for item in ambiguities if item not in deferred]
    return {
        "deferred_stage_fields": deferred,
        "validation_stage": stage,
        "valid": not missing and not conflicts and not ambiguities,
        "missing": sorted(set(missing)),
        "conflicts": sorted(set(conflicts)),
        "ambiguities": sorted(set(ambiguities)),
        "hard_constraints": [
            "system.constraints",
            "frozen_parameters",
            "budgets",
            "dft.parameters",
            "convergence",
        ],
        "adjustable_suggestions": [
            "branch_partition_suggestions",
            "generation_actions.quotas",
            "mc_policy.agent_adjustable",
        ],
    }

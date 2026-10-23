"""Check missing, conflicts and ambiguity without launching work."""

from math import isfinite


def validate_search_config(config: dict, *, stage="full") -> dict:
    if stage not in {"full", "startup"}:
        raise ValueError("unknown_configuration_stage")
    shape_errors = _shape_errors(config)
    if shape_errors:
        return _report([], shape_errors, [], stage)
    missing, conflicts, ambiguities = [], [], []
    if (config.get("budgets") or {}).get("enforcement", "advisory") not in ("advisory", "hard"):
        conflicts.append("budgets.enforcement_invalid")
    try:
        from phase_agent.tools.policy.workflow_policy import settings

        settings(config)
    except (ValueError, TypeError) as error:
        conflicts.append(str(error))
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
        h_by_phase = boundary.get("H", {})
        if not isinstance(h_by_phase, dict):
            conflicts.append("system.boundary.H 必须按相列出超胞矩阵")
        else:
            from phase_agent.science.structures.boundary_utils import det_H

            for phase, matrices in h_by_phase.items():
                if not isinstance(matrices, list):
                    conflicts.append(f"system.boundary.H.{phase} 必须是矩阵列表")
                    continue
                for index, matrix in enumerate(matrices):
                    try:
                        det_H(matrix)
                    except (TypeError, ValueError, OverflowError) as error:
                        conflicts.append(f"system.boundary.H.{phase}[{index}]：{error}")
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
    elif not _integer(epochs, minimum=1):
        conflicts.append("convergence.stable_model_update_epochs_must_be_positive_integer")
    for field, minimum in (("stable_feedback_cycles", 1), ("minimum_recent_dft_checks", 0)):
        if field in convergence and not _integer(convergence[field], minimum=minimum):
            conflicts.append(f"convergence.{field}_invalid_integer")
    for field in (
        "hull_change_tolerance",
        "final_energy_mae_tolerance",
        "final_energy_rmse_tolerance",
        "final_force_mae_tolerance",
        "final_force_rmse_tolerance",
        "phase_competition_gap_ev_per_atom",
    ):
        value = convergence.get(
            field,
            convergence.get(
                {
                    "hull_change_tolerance": "hull_tolerance",
                    "final_energy_mae_tolerance": "final_frame_error_tolerance",
                }.get(field, field)
            ),
        )
        if value is not None and (not _number(value) or float(value) < 0):
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
                elif (
                    not _integer(tier[field], minimum=1)
                    if field != "min_improvement"
                    else not _number(tier[field]) or float(tier[field]) < 0
                ):
                    conflicts.append(f"mc_policy.tiers[{index}].{field}_invalid")
        fraction = mc.get("random_exploration_fraction", 0)
        if not _number(fraction) or not 0 <= float(fraction) <= 1:
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
        if validation.get("max_energy_mae") is not None and (
            not _number(validation["max_energy_mae"]) or float(validation["max_energy_mae"]) < 0
        ):
            conflicts.append("mlip_finetune.validation.max_energy_mae_must_be_nonnegative")
        failure_limit = validation.get("max_critical_failure_fraction")
        if failure_limit is not None and (
            not _number(failure_limit) or not 0 <= float(failure_limit) <= 1
        ):
            conflicts.append("mlip_finetune.validation.max_critical_failure_fraction_invalid")
        reversals = validation.get("max_near_hull_ranking_reversals")
        if reversals is not None and not _integer(reversals, minimum=0):
            conflicts.append("mlip_finetune.validation.max_near_hull_ranking_reversals_invalid")
        refresh = finetune.get("refresh_validation") or {}
        for field in ("max_failure_fraction", "max_near_hull_ranking_reversals"):
            if refresh.get(field) is None:
                missing.append(f"mlip_finetune.refresh_validation.{field}")
            elif (
                not _integer(refresh[field], minimum=0)
                if field == "max_near_hull_ranking_reversals"
                else not _number(refresh[field]) or not 0 <= float(refresh[field]) <= 1
            ):
                conflicts.append(f"mlip_finetune.refresh_validation.{field}_invalid")
    allocator = calculation.get("mc_allocator")
    if allocator not in ("agent_tools", "bohb", "fixed"):
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
    else:
        total = config["budgets"]["total_relative_cost"]
        if not _number(total) or float(total) < 0:
            conflicts.append("budgets.total_relative_cost_must_be_nonnegative")
    return _report(missing, conflicts, ambiguities, stage)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    try:
        return isfinite(float(value))
    except (ValueError, OverflowError):
        return False


def _integer(value, *, minimum):
    return _number(value) and float(value).is_integer() and float(value) >= minimum


def _shape_errors(config):
    """Check containers before using dict/set operations on user-edited JSON."""
    if not isinstance(config, dict):
        return ["config_must_be_object"]
    errors = []
    objects = (
        "system",
        "system.boundary",
        "system.constraints",
        "system.generation",
        "system.branch_schema",
        "calculation",
        "budgets",
        "dft",
        "convergence",
        "generation_actions",
        "agent",
        "agent.adaptive",
        "workflow",
        "run",
        "qbc",
        "qbc.retrain",
        "mc_policy",
        "mlip_finetune",
        "mlip_finetune.training",
        "mlip_finetune.validation",
        "mlip_finetune.refresh_validation",
    )

    def value_at(path):
        value = config
        for key in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    for path in objects:
        value = value_at(path)
        if value is not None and not isinstance(value, dict):
            errors.append(f"{path}_must_be_object")
    for path in (
        "frozen_parameters",
        "generation_actions.enabled",
        "system.generation.enabled_strategies",
    ):
        value = value_at(path)
        if value is not None and (
            not isinstance(value, list) or any(not isinstance(item, str) for item in value)
        ):
            errors.append(f"{path}_must_be_string_list")
    tiers = value_at("mc_policy.tiers")
    if tiers is not None and (
        not isinstance(tiers, list) or any(not isinstance(item, dict) for item in tiers)
    ):
        errors.append("mc_policy.tiers_must_be_object_list")
    return errors


def _report(missing, conflicts, ambiguities, stage):
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

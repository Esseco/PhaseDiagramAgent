"""Safely migrate a persisted run to an approved configuration revision."""

from copy import deepcopy
from math import isfinite


_APPROVED_MC_POLICY_PATHS = {
    "mc_policy.second_segment_enabled",
    "budgets.cost_model.mc_step_cost_factor",
    "qbc.budget_limits.cost_model.mc_step_cost_factor",
    "qbc.cost_model.mc_step_cost_factor",
}
# Legacy snapshots predate mc_step_cost_factor and used one MLIP relaxation
# per MC step, i.e. an effective factor of 1.0.
_LEGACY_MC_STEP_COST_FACTOR = 1.0


def authorize_budget_extension(state, confirmed_snapshot, *, user_approved=False):
    from config_layer.runtime.authorize_generation_policy_revision import (
        authorize_second_mc_segment_revision,
    )
    second_mc = authorize_second_mc_segment_revision(state, confirmed_snapshot)
    if second_mc["status"] != "not_applicable":
        return second_mc
    current = deepcopy(state)
    new_version = confirmed_snapshot["config_version"]
    old_version = current.get("confirmed_config_version")
    if old_version in {None, new_version}:
        current["confirmed_config_version"] = new_version
        current.setdefault("confirmed_config", deepcopy(confirmed_snapshot["config"]))
        return {"status": "unchanged", "state": current}
    if _scientifically_empty(current):
        current["confirmed_config_version"] = new_version
        current["confirmed_config"] = deepcopy(confirmed_snapshot["config"])
        current.setdefault("config_migrations", []).append({
            "type": "empty_run_rebind", "from": old_version, "to": new_version,
        })
        return {"status": "empty_run_rebound", "state": current}
    if not user_approved:
        return {
            "status": "approval_required", "state": current,
            "from": old_version, "to": new_version,
        }
    old = current.get("confirmed_config")
    new = deepcopy(confirmed_snapshot["config"])
    requested = current.get("requested_config_revision") or {}
    if (requested.get("source_config_version") == old_version
            and requested.get("patch") == {"mlip_finetune.enabled": True}
            and (old or {}).get("mlip_finetune", {}).get("enabled") is False
            and new.get("mlip_finetune", {}).get("enabled") is True
            and _changed_paths(old, new) == ["mlip_finetune.enabled"]):
        from execution_layer.state.task_waiting import awaiting_task_result
        if any(awaiting_task_result(t) for t in current.get("tasks") or []) or current.get("pending_execution_policies"):
            return {"status": "rejected_active_work", "state": current, "from": old_version, "to": new_version}
        current["confirmed_config_version"] = new_version
        current["confirmed_config"] = new
        current.setdefault("config_migrations", []).append({"type": "approved_finetune_enablement",
                                                             "from": old_version, "to": new_version})
        current.pop("requested_config_revision", None)
        return {"status": "extended", "state": current, "from": old_version, "to": new_version}
    normalized_old, compatibility, compatibility_rejection = (
        _normalize_legacy_mace_relax_defaults(current, old or {}, new, old_version)
    )
    comparison_old = normalized_old if normalized_old is not None else (old or {})
    policy_changes = _approved_mc_policy_changes(comparison_old, new)
    if policy_changes is None:
        policy_changes = []
    elif policy_changes and _has_active_work(current):
        return {"status": "rejected_active_work", "state": current,
                "from": old_version, "to": new_version,
                "changed_fields": [row["field"] for row in policy_changes]}
    compared_old = _without_migratable_budget_limits(comparison_old)
    compared_new = _without_migratable_budget_limits(new)
    if policy_changes:
        for row in policy_changes:
            parts = tuple(row["field"].split("."))
            _remove_path(compared_old, parts)
            _remove_path(compared_new, parts)
    changed_fields = _changed_paths(
        compared_old, compared_new,
    )
    if not old or changed_fields or (compatibility_rejection and normalized_old is None):
        return {
            "status": "rejected_non_budget_change", "state": current,
            "from": old_version, "to": new_version,
            "changed_fields": changed_fields,
            "compatibility_rejection": compatibility_rejection,
        }
    if not _non_decreasing(
        _migratable_budget_limits(old), _migratable_budget_limits(new)
    ):
        return {
            "status": "rejected_budget_decrease", "state": current,
            "from": old_version, "to": new_version,
        }
    current["confirmed_config_version"] = new_version
    current["confirmed_config"] = new
    budget_changes = _budget_limit_changes(old, new)
    if compatibility:
        migration_type = "approved_migration_with_verified_legacy_mace_defaults"
    elif policy_changes and budget_changes:
        migration_type = "approved_mc_policy_and_budget_revision"
    elif policy_changes:
        migration_type = "approved_mc_policy_revision"
    else:
        migration_type = "approved_budget_extension"
    migration_record = {
        "type": migration_type, "from": old_version, "to": new_version,
        "budget_changes": budget_changes,
        "policy_changes": policy_changes,
    }
    if compatibility:
        migration_record["compatibility"] = deepcopy(compatibility)
    current.setdefault("config_migrations", []).append({
        **migration_record,
    })
    result = {
        "status": "extended", "state": current, "from": old_version,
        "to": new_version, "budget_changes": budget_changes,
    }
    if compatibility:
        result["compatibility"] = compatibility
    if policy_changes:
        result["policy_changes"] = deepcopy(policy_changes)
    return result


def _approved_mc_policy_changes(old, new):
    """Return only validated MC-segment and cost-estimate policy changes."""
    changes = []
    for field in sorted(_APPROVED_MC_POLICY_PATHS):
        before, after = _get_path(old, field), _get_path(new, field)
        if before == after:
            continue
        if field == "mc_policy.second_segment_enabled":
            if before is not False or after is not True:
                return None
            changes.append({"field": field, "from": before, "to": after})
            continue
        before_is_implicit = not _path_exists(old, field)
        effective_before = (
            _LEGACY_MC_STEP_COST_FACTOR if before_is_implicit else before
        )
        if not (_positive_finite_number(effective_before)
                and _positive_finite_number(after)):
            return None
        changes.append({
            "field": field,
            "from": float(effective_before),
            "to": float(after),
            "from_basis": "implicit_default" if before_is_implicit else "configured",
        })
    if changes and not _mc_cost_factors_consistent(new):
        return None
    return changes


def _positive_finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(float(value)) and float(value) > 0)


def _mc_cost_factors_consistent(config):
    fields = (
        "budgets.cost_model.mc_step_cost_factor",
        "qbc.budget_limits.cost_model.mc_step_cost_factor",
        "qbc.cost_model.mc_step_cost_factor",
    )
    values = []
    for field in fields:
        value = _get_path(config, field)
        if not _path_exists(config, field):
            value = _LEGACY_MC_STEP_COST_FACTOR
        elif not _positive_finite_number(value):
            return False
        values.append(float(value))
    return not values or all(abs(value - values[0]) <= 1e-12 for value in values[1:])


def _has_active_work(state):
    active = {"pending", "planned", "reserved", "submitted", "running", "queued"}
    tasks = state.get("tasks") or []
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    if any((row or {}).get("status") in active for row in tasks):
        return True
    if any((row or {}).get("status") in active
           for row in (state.get("budget_reservations") or {}).values()):
        return True
    return bool(state.get("pending_execution_policies"))


def _remove_path(value, path):
    cursor = value
    for part in path[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor.pop(path[-1], None)


def _path_exists(value, path):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _without_migratable_budget_limits(config):
    value = deepcopy(config)
    budgets = value.get("budgets")
    if isinstance(budgets, dict):
        budgets.pop("total_relative_cost", None)
        for limits in (budgets.get("stage_limits") or {}).values():
            if isinstance(limits, dict):
                limits.pop("max_cost", None)
        llm_limits = budgets.get("llm_limits")
        if isinstance(llm_limits, dict):
            for key in ("max_calls", "max_calls_per_iteration", "max_cost",
                        "max_input_tokens", "max_output_tokens"):
                llm_limits.pop(key, None)
        for key in ("stage_limits", "llm_limits"):
            if budgets.get(key) == {}:
                budgets.pop(key)
        if not budgets:
            value.pop("budgets")
    strategy = value.get("round_strategy")
    if isinstance(strategy, dict):
        strategy.pop("maximum_mc_budget", None)
        strategy.pop("maximum_dft_budget", None)
        if not strategy:
            value.pop("round_strategy", None)
    return value


def _non_decreasing(old, new):
    if isinstance(old, dict) and isinstance(new, dict) and set(old) != set(new):
        return False
    for key, value in old.items():
        replacement = new.get(key)
        if isinstance(value, dict):
            if not isinstance(replacement, dict) or not _non_decreasing(value, replacement): return False
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if replacement is None or float(replacement) < float(value): return False
        elif replacement != value:
            return False
    return True


def _migratable_budget_limits(config):
    """Only expose resource caps; structural bounds and policy stay immutable."""
    limits = {"budgets": {}, "round_strategy": {}}
    budgets = config.get("budgets") or {}
    if "total_relative_cost" in budgets:
        limits["budgets"]["total_relative_cost"] = budgets["total_relative_cost"]
    stage_limits = {
        stage: {"max_cost": row["max_cost"]}
        for stage, row in (budgets.get("stage_limits") or {}).items()
        if isinstance(row, dict) and "max_cost" in row
    }
    if stage_limits:
        limits["budgets"]["stage_limits"] = stage_limits
    llm_fields = ("max_calls", "max_calls_per_iteration", "max_cost",
                  "max_input_tokens", "max_output_tokens")
    llm_limits = {
        key: (budgets.get("llm_limits") or {})[key]
        for key in llm_fields if key in (budgets.get("llm_limits") or {})
    }
    if llm_limits:
        limits["budgets"]["llm_limits"] = llm_limits
    strategy = config.get("round_strategy") or {}
    round_limits = {
        key: strategy[key] for key in ("maximum_mc_budget", "maximum_dft_budget")
        if key in strategy
    }
    if round_limits:
        limits["round_strategy"] = round_limits
    if not limits["budgets"]:
        limits.pop("budgets")
    if not limits["round_strategy"]:
        limits.pop("round_strategy")
    return limits


def _normalize_legacy_mace_relax_defaults(state, old, new, old_version):
    """Recognize only the exact mh-1 defaults evidenced in old Relax results.

    Historical task records retain their original config labels. The migration
    audit reports dtype and worker revision as unrecorded, rather than inventing
    historical provenance.
    """
    expected_parameters = {
        "fmax": 0.05,
        "mace_default_dtype": "float64",
        "relax_cell": True,
        "relax_steps": 150,
    }
    old_mlip = deepcopy(old.get("mlip") or {})
    new_mlip = new.get("mlip") or {}
    if (old_mlip.get("name") != "mace-mh-1"
            or new_mlip.get("name") != "mace-mh-1"
            or old_mlip.get("mace_head") not in (None, "")
            or old_mlip.get("relax_parameters") not in (None, {})):
        return None, None, None
    if (new_mlip.get("mace_head") != "omat_pbe"
            or new_mlip.get("relax_parameters") != expected_parameters):
        return None, None, None

    relax_stages = {"relax_and_feature", "mlip_relax", "relax_screen"}
    tasks = state.get("tasks") or []
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    relax_tasks = [task for task in tasks if isinstance(task, dict)
                   and task.get("stage") in relax_stages]
    if not relax_tasks:
        return None, None, "没有可核对的旧 Relax 任务结果"

    for task in relax_tasks:
        output = task.get("outputs") or task.get("result") or {}
        parameters = task.get("parameters") or {}
        steps = output.get("relax_steps_used")
        model_version = str(task.get("model_version") or "")
        if (task.get("status") != "completed"
                or "mh-1" not in model_version.lower()
                or output.get("status") not in (None, "completed")
                or output.get("mace_head") != "omat_pbe"
                or not _numeric_matches(output.get("fmax_target_ev_per_angstrom"), 0.05)
                or output.get("cell_relaxed") is not True
                or not isinstance(steps, int) or isinstance(steps, bool)
                or not 0 <= steps <= 150):
            return None, None, "旧 Relax 任务未全部完成，或结果元数据与 mh-1 默认设置不一致"
        for key, expected in {
            "mace_head": "omat_pbe", "fmax": 0.05, "relax_cell": True,
            "relax_steps": 150, "mace_default_dtype": "float64",
        }.items():
            if key in parameters and not _values_match(parameters[key], expected):
                return None, None, f"旧任务参数 {key} 与新配置不一致"

    old_mlip["mace_head"] = "omat_pbe"
    old_mlip["relax_parameters"] = expected_parameters
    normalized_old = deepcopy(old)
    normalized_old["mlip"] = old_mlip
    dtype_recorded = all(
        "mace_default_dtype" in (task.get("parameters") or {})
        for task in relax_tasks
    )
    compatibility = {
        "basis": "verified_legacy_mace_mh_1_worker_defaults",
        "model": "mace-mh-1",
        "relax_task_count": len(relax_tasks),
        "verified_result_fields": {
            "mace_head": "omat_pbe",
            "fmax_target_ev_per_angstrom": 0.05,
            "cell_relaxed": True,
            "relax_steps_used_max": max(
                (task.get("outputs") or task.get("result") or {})["relax_steps_used"]
                for task in relax_tasks
            ),
        },
        "unrecorded_fields": ([] if dtype_recorded else [
            "mlip.relax_parameters.mace_default_dtype",
        ]) + ["worker_code_revision"],
        "legacy_config_version": old_version,
    }
    return normalized_old, compatibility, None


def _numeric_matches(value, expected):
    try:
        return abs(float(value) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _values_match(value, expected):
    if isinstance(expected, float):
        return _numeric_matches(value, expected)
    return value == expected


def _budget_limit_changes(old, new):
    changes = []
    before = _migratable_budget_limits(old)
    after = _migratable_budget_limits(new)
    for field in _changed_paths(before, after):
        left, right = _get_path(before, field), _get_path(after, field)
        changes.append({"field": field, "from": left, "to": right})
    return changes


def _get_path(value, path):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return deepcopy(value)


def _changed_paths(old, new, prefix=""):
    """Return compact paths for changed non-budget config values."""
    if isinstance(old, dict) and isinstance(new, dict):
        changed = []
        for key in sorted(set(old) | set(new)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old or key not in new:
                changed.append(path)
            else:
                changed.extend(_changed_paths(old[key], new[key], path))
        return changed
    return [] if old == new else [prefix]


def _scientifically_empty(state):
    """A rejected/debug-only state may adopt a new confirmed snapshot safely."""
    if state.get("tasks") or state.get("branch_candidates"):
        return False
    reservations = (state.get("budget_reservations") or {}).values()
    if any(item.get("status") in {"reserved", "submitted", "running", "completed"}
           for item in reservations):
        return False
    usage = state.get("budget_usage") or {}
    if float(usage.get("total_relative_cost") or 0.0) > 0:
        return False
    return not (usage.get("stages") or {})

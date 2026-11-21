"""Narrow fixed protocol, not another executor or authorization policy."""

from copy import deepcopy
from phase_agent.configuration.schema.action_state_schema import normalize_action
from phase_agent.policies.file_approval import proposal_hash


def dft_input_record(action, state):
    """None means unclassified; blocked means a known, deterministic prerequisite."""
    action = normalize_action(action)
    if (
        action["tool"] != "prepare_local_batch_files"
        or action.get("parameters") != {"mode": "dft_inputs"}
        or not action.get("target_ids")
    ):
        return None
    ids = action["target_ids"]
    requested = set(ids)
    rows = state.get("tasks") or []
    tasks = {row.get("task_id"): row for row in rows}
    errors = []
    if sum(row.get("task_id") in requested for row in rows) != len(set(ids) & set(tasks)):
        errors.append("duplicate_registered_task_ids")
    if len(set(ids)) != len(ids):
        errors.append("duplicate_task_ids")
    if not state.get("confirmed_config_version") or not state.get("active_model_version"):
        errors.append("unknown_config_or_model_version")
    for key in ids:
        task = tasks.get(key)
        if task is None:
            errors.append("unregistered_task:" + key)
        elif not task.get("model_version"):
            errors.append("unknown_task_model:" + key)
        elif (
            task.get("stage") not in {"dft_relax", "dft_single_point"}
            or task.get("status") != "pending"
            or task.get("slurm_batch_id")
        ):
            errors.append("task_not_pending_dft:" + key)
    return {
        "protocol": "registered_dft_inputs/v1",
        "handler": "program",
        "status": "blocked" if errors else "passed",
        "target_ids": deepcopy(ids),
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "task_digest": proposal_hash([tasks.get(key) for key in ids]),
        "checks": [
            "explicit_registered_task_scope",
            "pending_dft_without_submission",
            "known_config_and_model_versions",
        ],
        "unmet_conditions": errors,
        "execution_checks": "not_executed",
        "scientific_completion": "unknown",
        "authorization": "deferred_to_existing_approval_and_tool_validation",
    }


def initial_direct_record(action, state, *, config=None, manager=None):
    """Deterministic preflight for initial parent-first perturbation DIRECT."""
    import hashlib
    from pathlib import Path, PurePosixPath
    import re

    action = normalize_action(action)
    params = action.get("parameters") or {}
    mode = params.get("mode") or params.get("bundle_type")
    is_initial_direct = params.get("scope_id") == "initial_direct" or (
        params.get("sampling_method") == "direct" and "initial_parents" in params
    )
    if (
        action["tool"] != "prepare_local_batch_files"
        or mode != "parent_first_recovery_inputs"
        or not is_initial_direct
    ):
        return None
    errors = []
    if not state.get("confirmed_config_version") or not state.get("active_model_version"):
        errors.append("unknown_config_or_model_version")
    if state.get("dft_dataset_records") or state.get("initial_direct_sampling"):
        errors.append("initial_direct_already_completed")
    if not isinstance(params.get("initial_parents"), list) or not params.get("initial_parents"):
        errors.append("initial_parents_required")
    if config is not None:
        if params.get("confirmed") is not True:
            errors.append("confirmed_true_required")
        if params.get("round_evidence_digest") != "initial_direct_v1":
            errors.append("invalid_round_evidence_digest")
        if params.get("dft_parameters") != ((config.get("dft") or {}).get("parameters") or {}):
            errors.append("dft_parameters_mismatch")
        mlip_cfg = config.get("mlip") or {}
        direct_cfg = config.get("direct") or {}
        f_path = (
            params.get("feature_model_path")
            or direct_cfg.get("feature_model_path")
            or mlip_cfg.get("feature_model_path")
        )
        f_sha = (
            params.get("feature_model_sha256")
            or direct_cfg.get("feature_model_sha256")
            or mlip_cfg.get("feature_model_sha256")
        )
        if not (
            Path(str(f_path or "")).is_absolute() or PurePosixPath(str(f_path or "")).is_absolute()
        ):
            errors.append("missing_or_invalid_feature_model_path")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(f_sha or "")):
            errors.append("missing_or_invalid_feature_model_sha256")
        model_sha = params.get("model_sha256") or mlip_cfg.get("model_sha256")
        if not model_sha:
            try:
                from phase_agent.tools.workflows.comparison_model_registry import (
                    original_round_model,
                )

                _, model_sha, _ = original_round_model(state, {}, "mace-mh-1")
            except (ValueError, KeyError, TypeError):
                model_sha = None
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(model_sha or "")):
            errors.append("missing_or_invalid_model_sha256")
        if (
            manager is not None
            and isinstance(params.get("initial_parents"), list)
            and params.get("initial_parents")
        ):
            try:
                from phase_agent.tools.workflows.parent_first_handoff import (
                    _resolve_initial_parents,
                )

                resolved = _resolve_initial_parents(
                    state, getattr(manager, "data", {}) or {}, params["initial_parents"]
                )
                structures = (getattr(manager, "data", {}) or {}).get("structures") or {}
                for item in resolved:
                    sid = (
                        item.get("structure_id") or item.get("record_id")
                        if isinstance(item, dict)
                        else item
                    )
                    row = structures.get(sid) or (item if isinstance(item, dict) else {})
                    src = Path(row.get("structure_path") or row.get("source_path") or "")
                    if not src.is_file():
                        errors.append(f"missing_parent_file:{sid}")
                    elif (
                        row.get("structure_sha256")
                        and hashlib.sha256(src.read_bytes()).hexdigest() != row["structure_sha256"]
                    ):
                        errors.append(f"parent_sha256_mismatch:{sid}")
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"invalid_initial_parents:{exc}")
        elif isinstance(params.get("initial_parents"), list):
            for idx, item in enumerate(params["initial_parents"]):
                if isinstance(item, dict) and not item.get("structure_path"):
                    errors.append(f"invalid_initial_parents:missing_structure_path_at_{idx}")
                elif isinstance(item, dict) and item.get("structure_path"):
                    src = Path(str(item["structure_path"]))
                    if not src.is_file():
                        errors.append(f"missing_parent_file:{item.get('record_id', idx)}")
    return {
        "protocol": "initial_perturbation_direct/v1",
        "handler": "program",
        "status": "blocked" if errors else "passed",
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "parent_count": len(params.get("initial_parents") or []),
        "checks": [
            "initial_scope_without_prior_dft",
            "explicit_initial_parents",
            "known_config_and_model_versions",
        ],
        "unmet_conditions": errors,
        "execution_checks": "not_executed",
        "scientific_completion": "unknown",
        "authorization": "deferred_to_existing_approval_and_tool_validation",
    }


def parent_first_input_record(action, state, *, config=None, manager=None):
    """Preflight for both initial_direct and extrapolation parent_first_recovery."""
    rec = initial_direct_record(action, state, config=config, manager=manager)
    if rec is not None:
        return rec
    from pathlib import Path, PurePosixPath
    import re

    action = normalize_action(action)
    params = action.get("parameters") or {}
    mode = params.get("mode") or params.get("bundle_type")
    if action["tool"] != "prepare_local_batch_files" or mode != "parent_first_recovery_inputs":
        return None
    if action.get("operation") not in {None, "", "prepare_parent_first_recovery"}:
        return None

    errors = []
    if not state.get("confirmed_config_version") or not state.get("active_model_version"):
        errors.append("unknown_config_or_model_version")
    if not isinstance(params.get("parent_record_ids"), list) or not params.get("parent_record_ids"):
        errors.append("parent_record_ids_required")
    if config is not None:
        if params.get("confirmed") is not True:
            errors.append("confirmed_true_required")
        if params.get("dft_parameters") != ((config.get("dft") or {}).get("parameters") or {}):
            errors.append("dft_parameters_mismatch")
        mlip_cfg = config.get("mlip") or {}
        direct_cfg = config.get("direct") or {}
        if params.get("sampling_method") == "direct":
            f_path = (
                params.get("feature_model_path")
                or direct_cfg.get("feature_model_path")
                or mlip_cfg.get("feature_model_path")
            )
            f_sha = (
                params.get("feature_model_sha256")
                or direct_cfg.get("feature_model_sha256")
                or mlip_cfg.get("feature_model_sha256")
            )
            if not (
                Path(str(f_path or "")).is_absolute()
                or PurePosixPath(str(f_path or "")).is_absolute()
            ):
                errors.append("missing_or_invalid_feature_model_path")
            if not re.fullmatch(r"[a-fA-F0-9]{64}", str(f_sha or "")):
                errors.append("missing_or_invalid_feature_model_sha256")
        model_sha = params.get("model_sha256") or mlip_cfg.get("model_sha256")
        if not model_sha:
            try:
                from phase_agent.tools.workflows.comparison_model_registry import (
                    original_round_model,
                )

                _, model_sha, _ = original_round_model(state, {}, "mace-mh-1")
            except (ValueError, KeyError, TypeError):
                model_sha = None
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(model_sha or "")):
            errors.append("missing_or_invalid_model_sha256")
    return {
        "protocol": "parent_first_recovery_inputs/v1",
        "handler": "program",
        "status": "blocked" if errors else "passed",
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "parent_count": len(params.get("parent_record_ids") or []),
        "checks": [
            "confirmed_parent_first_parameters",
            "known_config_and_model_versions",
        ],
        "unmet_conditions": errors,
        "execution_checks": "not_executed",
        "scientific_completion": "unknown",
        "authorization": "deferred_to_existing_approval_and_tool_validation",
    }


_DETERMINISTIC_LOCAL_BATCH_MODES = frozenset(
    {
        "relax_mc_inputs",
        "relax_inputs",
        "mc_inputs",
        "ordering_search_inputs",
        "training_handoff_inputs",
        "mc_energy_repair_inputs",
        "model_refresh_inputs",
    }
)


def local_batch_input_record(action, state, *, config=None, manager=None):
    """Deterministic preflight for all prepare_local_batch_files modes across rounds."""
    rec = dft_input_record(action, state)
    if rec is not None:
        return rec
    rec = parent_first_input_record(action, state, config=config, manager=manager)
    if rec is not None:
        return rec
    action = normalize_action(action)
    params = action.get("parameters") or {}
    mode = params.get("mode") or params.get("bundle_type")
    if (
        action["tool"] != "prepare_local_batch_files"
        or mode not in _DETERMINISTIC_LOCAL_BATCH_MODES
    ):
        return None
    errors = []
    if not state.get("confirmed_config_version") or not state.get("active_model_version"):
        errors.append("unknown_config_or_model_version")
    if mode == "training_handoff_inputs" and params.get("confirmed") is not True:
        errors.append("confirmed_true_required")
    return {
        "protocol": f"{mode}/v1",
        "handler": "program",
        "status": "blocked" if errors else "passed",
        "config_version": state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "checks": ["known_config_and_model_versions", "deterministic_local_batch_scope"],
        "unmet_conditions": errors,
        "execution_checks": "not_executed",
        "scientific_completion": "unknown",
        "authorization": "deferred_to_existing_approval_and_tool_validation",
    }


def validate_program(record, action, state):
    current = parent_first_input_record(action, state) or dft_input_record(action, state)
    if current is None or current["status"] != "passed":
        raise ValueError("simulation_program_stale_or_blocked; return to Strategic")
    return current

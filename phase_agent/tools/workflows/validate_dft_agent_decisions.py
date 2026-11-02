"""Validate categorical DFT decisions and apply deterministic hard triggers."""

from phase_agent.tools.budget.estimate_stage_cost import estimate_dft_cost
from phase_agent.tools.budget.check_budget import check_budget, budget_is_hard
from phase_agent.agents.reliability.dft_contracts import dft_contract_errors


def validate_dft_agent_decisions(
    proposal: dict,
    metrics: list[dict],
    state: dict,
    *,
    config: dict,
    config_version: str,
    remaining_budget: float,
) -> dict:
    errors, accepted, rejected = [], [], []
    # Budget checks read existing usage; only new local reservations are added.
    budget_state = {
        "budget_usage": state.get("budget_usage") or {},
        "budget_reservations": dict(state.get("budget_reservations") or {}),
    }
    if str(proposal.get("source") or "").startswith("llm_agent") and _contains_forbidden_numeric(
        {
            "decisions": proposal.get("decisions"),
            "global_action": proposal.get("global_action"),
            "reason": proposal.get("reason"),
        }
    ):
        errors.append("agent_supplied_scientific_numeric_value")
    by_id = {item["candidate_id"]: item for item in metrics}
    seen = set()
    spent = 0.0
    decisions = proposal.get("decisions", [])
    if not isinstance(decisions, list) or any(not isinstance(item, dict) for item in decisions):
        return {
            "valid": False,
            "errors": ["invalid_dft_decisions_format"],
            "accepted": [],
            "rejected": [],
            "global_action": proposal.get("global_action"),
        }
    contract_errors = dft_contract_errors(decisions)
    if contract_errors:
        if str(proposal.get("source") or "").startswith("llm_agent"):
            for item in decisions:
                extra = set(item) - {"candidate_id", "action", "reason"}
                if extra:
                    errors.append(f"agent_decision_fields_forbidden:{','.join(sorted(extra))}")
        return {
            "valid": False,
            "errors": errors + contract_errors,
            "accepted": [],
            "rejected": [],
            "global_action": proposal.get("global_action"),
        }
    relax_limit = (
        max(
            1,
            int(
                len(decisions)
                * float((config.get("selection_policy") or {}).get("max_relax_fraction", 0.10))
            ),
        )
        if decisions
        else 0
    )
    relax_accepted = 0
    for item in decisions:
        extra_fields = set(item) - {"candidate_id", "action", "reason"}
        if str(proposal.get("source") or "").startswith("llm_agent") and extra_fields:
            errors.append(f"agent_decision_fields_forbidden:{','.join(sorted(extra_fields))}")
            continue
        candidate_id, action = item.get("candidate_id"), item.get("action")
        if candidate_id not in by_id:
            errors.append(f"unknown_candidate:{candidate_id}")
            continue
        if candidate_id in seen:
            errors.append(f"duplicate_candidate_decision:{candidate_id}")
            continue
        seen.add(candidate_id)
        if action not in set(config.get("allowed_actions") or []):
            errors.append(f"invalid_action:{action}")
            continue
        metric = by_id[candidate_id]
        if action == "DFT_RELAX" and relax_accepted >= relax_limit:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "action": action,
                    "reason": "dft_relax_reserved_for_limited_geometry_checks",
                }
            )
            continue
        if action == "DFT_RELAX" and not _relax_geometry_justified(metric, config):
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "action": action,
                    "reason": "dft_relax_requires_hull_and_geometry_evidence",
                }
            )
            continue
        hard = _hard_action(metric, config)
        if metric.get("duplicate_of"):
            if action not in {"REJECT", "DEFER"}:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "action": action,
                        "reason": "duplicate_safety_rule",
                    }
                )
                continue
            hard = "duplicate_safety_rule"
        elif hard and action in {"DEFER", "REJECT"}:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "action": action,
                    "reason": "configured_uncertainty_rule_requires_revision",
                    "required_action": hard,
                }
            )
            continue
        task_key = f"{config_version}:{candidate_id}:{action}"
        if task_key in (state.get("effective_decisions") or {}):
            rejected.append(
                {"candidate_id": candidate_id, "action": action, "reason": "duplicate_task_key"}
            )
            continue
        cost = estimate_dft_cost(action, metric, config)
        if budget_is_hard(config.get("budget_limits")) and spent + cost > remaining_budget:
            rejected.append(
                {"candidate_id": candidate_id, "action": action, "reason": "dft_budget"}
            )
            continue
        spent += cost
        stage = {"DFT_SINGLE_POINT": "dft_single_point", "DFT_RELAX": "dft_relax"}.get(action)
        budget_warnings = []
        if stage and config.get("budget_limits"):
            check = check_budget(
                budget_state,
                {"stage": stage, "relative_cost": cost, "tasks": 1},
                config["budget_limits"],
            )
            if not check["allowed"]:
                spent -= cost
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "action": action,
                        "reason": "dft_budget",
                        "details": check["reasons"],
                    }
                )
                continue
            budget_warnings = check.get("warnings", [])
            budget_state.setdefault("budget_reservations", {})[task_key] = {
                "stage": stage,
                "reserved_cost": cost,
                "status": "reserved",
            }
        accepted.append(
            {
                "candidate_id": candidate_id,
                "branch_id": metric.get("branch_id"),
                "action": action,
                "task_key": task_key,
                "relative_cost": cost,
                "reason": item.get("reason"),
                "hard_trigger": hard,
                "budget_warnings": budget_warnings,
                "config_version": config_version,
            }
        )
        if action == "DFT_RELAX":
            relax_accepted += 1
    audit_ids = set(state.get("dft_audit_candidate_ids") or [])
    selected_ids = {
        r["candidate_id"] for r in accepted if r["action"] in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    }
    if selected_ids and audit_ids and not (audit_ids & selected_ids):
        errors.append("independent_dft_audit_required")
    if selected_ids:
        errors.extend(_ground_state_selection_errors(state, selected_ids))
    for row in accepted:
        row["independent_audit"] = row["candidate_id"] in audit_ids
    global_action = proposal.get("global_action", "CONTINUE_DATA_COLLECTION")
    if global_action not in {"RETRAIN_MLIP", "CONTINUE_DATA_COLLECTION"}:
        errors.append("invalid_global_action")
    return {
        "valid": not errors,
        "errors": errors,
        "accepted": accepted,
        "rejected": rejected,
        "reserved_cost": spent,
        "global_action": global_action,
        "source": proposal.get("source"),
    }


def _hard_action(metric, config):
    rule = config.get("extreme_uncertainty") or {}
    for key in ("f_std_max", "energy_std"):
        threshold = rule.get(key)
        value = metric.get(key)
        if threshold is not None and value is not None and float(value) >= float(threshold):
            return rule.get("hard_action", "DFT_SINGLE_POINT")
    return None


def _contains_forbidden_numeric(value, path=""):
    forbidden = {
        "energy",
        "ehull",
        "qbc",
        "uncertainty",
        "f_std_max",
        "f_std_p95",
        "energy_std",
        "converged",
        "score",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if any(token in str(key).lower() for token in forbidden) and isinstance(
                child, (int, float)
            ):
                return True
            if _contains_forbidden_numeric(child, f"{path}.{key}"):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_numeric(item, path) for item in value)
    return False


def _relax_geometry_justified(metric, config):
    from math import isfinite

    evidence = metric.get("geometry_optimization_evidence") or {}
    gap = metric.get("predicted_Ehull")
    residual = evidence.get("residual_force_ev_per_angstrom")
    tolerance = evidence.get("force_tolerance_ev_per_angstrom")
    values = (gap, residual, tolerance)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in values
    ):
        return False
    return (
        0 <= gap <= float(config.get("near_hull_threshold", 0.1))
        and bool(evidence.get("source_task_id"))
        and tolerance > 0
        and residual > tolerance
    )


def _ground_state_selection_errors(state, selected_ids):
    """Require each unvalidated current GS and its actual frame in a DFT batch."""
    from phase_agent.analysis.phase.phase_dft_candidates import bind_ground_state_frames
    from phase_agent.analysis.convergence.independent_evidence import geometry_key
    from phase_agent.analysis.feedback.dft_comparison_tables import build_comparison_tables

    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("status") != "completed" or snapshot.get("model_version") != state.get(
        "active_model_version"
    ):
        return []
    rows = [r for r in snapshot.get("entries") or [] if r.get("is_stable")]
    if not rows:
        return []
    try:
        bound = bind_ground_state_frames(state, state.get("qbc_candidates") or [])
    except (ValueError, OSError) as error:
        return [str(error)]
    existing = {r.get("candidate_id"): r for r in state.get("qbc_candidates") or []}
    records = [
        r
        for r in state.get("dft_dataset_records") or []
        if r.get("checks_passed") is True
        and r.get("model_version") == snapshot.get("model_version")
    ]
    energies, _ = build_comparison_tables(
        records, state.get("dft_mlip_comparisons") or [], round_name="ground_state"
    )
    valid_tasks = {r["task_id"] for r in energies if r.get("comparison_status") == "completed"}
    checked = {key for r in records if r.get("task_id") in valid_tasks if (key := geometry_key(r))}
    errors = []
    for row in bound:
        if not row.get("ground_state_record_ids"):
            continue
        sid = row["candidate_id"]
        if geometry_key({"structure": row["structure_path"]}) in checked:
            continue
        if sid not in selected_ids:
            errors.append("ground_state_selection_required:" + sid)
            continue
        candidate = existing.get(sid) or {}
        path = candidate.get("structure_path")
        try:
            from hashlib import sha256
            from pathlib import Path

            actual = sha256(Path(path).read_bytes()).hexdigest() if path else None
        except OSError:
            actual = None
        if actual != row["ground_state_frame_sha256"]:
            errors.append("ground_state_frame_binding_required:" + sid)
    return errors

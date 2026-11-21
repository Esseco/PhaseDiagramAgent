"""Read-only, evidence-bound DFT recalculation review."""

import hashlib
import json
import math
from collections import Counter
from copy import deepcopy

from phase_agent.evidence.dft_failure_policy import diagnose_failure_batch

PROFILE = "dft_recalculation"
STRATEGIES = {
    "original",
    "mixing_parameters",
    "vasp_symmetry_single_gpu",
    "nelm_200",
    "nelm_200_slow_mixing",
}


def review(state, target_ids, client=None):
    if (
        not isinstance(target_ids, list)
        or not 1 <= len(target_ids) <= 1000
        or any(not isinstance(x, str) or not x for x in target_ids)
        or len(set(target_ids)) != len(target_ids)
    ):
        raise ValueError("dft_recalculation_requires_unique_task_ids")
    tasks = state.get("tasks") or []
    selected_ids = set(target_ids)
    selected = [task for task in tasks if task.get("task_id") in selected_ids]
    if len(selected) != len(target_ids) or len({t.get("task_id") for t in selected}) != len(
        target_ids
    ):
        raise ValueError("dft_recalculation_target_missing_or_duplicate")
    selected.sort(key=lambda t: t["task_id"])
    if any(
        t.get("stage") not in {"dft_single_point", "dft_relax"}
        or t.get("status") not in {"failed", "timeout"}
        for t in selected
    ):
        raise ValueError("dft_recalculation_requires_failed_terminal_dft")
    attempts = sorted(
        [deepcopy(t) for t in tasks if t.get("retry_of") in selected_ids],
        key=lambda t: str(t.get("task_id")),
    )
    evidence = {
        "config_version": state.get("confirmed_config_version"),
        "active_model_version": state.get("active_model_version"),
        "tasks": deepcopy(selected),
        "prior_attempts": attempts,
    }
    if not evidence["config_version"] or not evidence["active_model_version"]:
        raise ValueError("dft_recalculation_version_unknown")
    digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    totals, unknown = [], []
    rows = []
    for t in selected:
        cost = t.get("planned_relative_cost", t.get("budget"))
        valid = (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
        )
        if valid:
            totals.append(cost)
        else:
            unknown.append(t["task_id"])
        rows.append(
            {
                k: deepcopy(t.get(k))
                for k in (
                    "task_id",
                    "task_key",
                    "stage",
                    "status",
                    "structure_id",
                    "model_version",
                    "config_version",
                    "error",
                    "failure_reason",
                    "workflow_error",
                    "input_path",
                    "result_path",
                    "planned_relative_cost",
                    "budget",
                    "retry_index",
                )
            }
        )
    reservations = [
        r
        for r in (state.get("budget_reservations") or {}).values()
        if r.get("status") in {"reserved", "submitted", "running"}
    ]
    observations = {
        "failure_evidence_digest": digest,
        "failure_evidence": rows,
        "failed_count": len(rows),
        "statuses": dict(Counter(t["status"] for t in selected)),
        "prior_attempts": [
            {k: t.get(k) for k in ("task_id", "retry_of", "retry_index", "status")}
            for t in attempts
        ],
        "config_version": evidence["config_version"],
        "active_model_version": evidence["active_model_version"],
        "budget_limits": deepcopy((state.get("confirmed_config") or {}).get("budgets")),
        "budget_usage": deepcopy((state.get("budget_usage") or {}).get("stages")),
        "active_reservation_count": len(reservations),
        "estimated_relative_cost": sum(totals) if not unknown else None,
        "cost_basis": "sum of original task reservations; changed settings can alter actual cost",
        "unknown_cost_task_ids": unknown,
    }
    diagnosis = diagnose_failure_batch(
        observations,
        client,
        extra_instruction=(
            " Review recalculation feasibility and unknown evidence. Identify which failures "
            "have support for retry, risks, inputs, resources, estimated costs and assumptions. "
            "Do not select VASP parameters or grant authorization. Existing retry profiles "
            "are options only; human and Strategic choose the exact frozen action."
        ),
    )
    return {
        "review": observations,
        **diagnosis,
        "next": "human_decides_whether_to_propose_recalculation",
        "execution_authorized": False,
    }


def validate_action(action, state):
    params = action.get("parameters") or {}
    targets = action.get("target_ids") or []
    if params.get("repair_profile") != PROFILE:
        return
    allowed = {
        "repair_profile",
        "strategies",
        "failure_evidence_digest",
        "diagnosis",
        "max_retries",
        "prepare_inputs_after_approval",
    }
    if set(params) - allowed:
        raise ValueError("dft_recalculation_unapproved_parameters")
    strategies = params.get("strategies")
    if (
        not isinstance(strategies, dict)
        or set(strategies) != set(targets)
        or any(
            not isinstance(value, str) or value not in STRATEGIES for value in strategies.values()
        )
    ):
        raise ValueError("dft_recalculation_strategies_required")
    maximum = params.get("max_retries", 1)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("dft_recalculation_retry_limit_invalid")
    if params.get("prepare_inputs_after_approval"):
        raise ValueError("dft_recalculation_input_preparation_requires_separate_approval")
    if (
        "prepare_inputs_after_approval" in params
        and type(params["prepare_inputs_after_approval"]) is not bool
    ):
        raise ValueError("dft_recalculation_prepare_inputs_must_be_boolean")
    from phase_agent.agents.simulation.contracts import SimulationDiagnosis

    diagnosis = SimulationDiagnosis.model_validate(params.get("diagnosis"))
    if "execution.operation_outcome" not in diagnosis.evidence:
        raise ValueError("dft_recalculation_diagnosis_missing_evidence")
    result = review(state, targets)
    if params.get("failure_evidence_digest") != result["review"]["failure_evidence_digest"]:
        raise ValueError("dft_recalculation_evidence_changed")
    if result["review"]["prior_attempts"]:
        raise ValueError("dft_recalculation_existing_retry_requires_review")
    tasks = {t["task_id"]: t for t in state.get("tasks") or [] if t.get("task_id") in targets}
    if any(
        t.get("model_version") != state.get("active_model_version")
        or t.get("config_version") != state.get("confirmed_config_version")
        for t in tasks.values()
    ):
        raise ValueError("dft_recalculation_original_model_or_config_mismatch")
    if any(
        strategies[t] in {"nelm_200", "nelm_200_slow_mixing"}
        and tasks[t].get("stage") != "dft_single_point"
        for t in targets
    ):
        raise ValueError("dft_recalculation_scf_requires_single_point")
    if result["review"]["unknown_cost_task_ids"]:
        raise ValueError("dft_recalculation_cost_unknown")
    budget = action.get("budget", 0)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget < result["review"]["estimated_relative_cost"]
    ):
        raise ValueError("dft_recalculation_budget_below_base_estimate")

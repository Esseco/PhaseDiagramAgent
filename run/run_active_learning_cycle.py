"""Legacy internal compatibility wrapper.

New workflows enter through :func:`run.run_workflow`.  This module remains
temporarily for persisted callers while its scientific operations are exposed
as validated tool handlers; it is not a second public entry.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from config_layer.defaults.default_dft_decision_config import default_dft_decision_config
from execution_layer.workflows.update_dft_action_results import update_dft_action_results
from data_layer.ledger.collect_calculation_results import collect_calculation_results
from analysis_layer.convergence.build_convergence_evidence import build_convergence_evidence
from analysis_layer.convergence.check_global_convergence import check_global_convergence
from run.run_pipeline import run_pipeline
from execution_layer.state.normalize_task import normalize_task
from execution_layer.state.normalize_task_result import normalize_task_result
from execution_layer.workflows.run_qbc_dft_decision_flow import run_qbc_dft_decision_flow


def run_active_learning_cycle(
    manager: Any,
    phase_references: dict[str, Any],
    run_config: dict[str, Any],
    *,
    config_version: str,
    state: dict[str, Any] | str | Path | None = None,
    dispatcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    recovered_results: list[dict[str, Any]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    qbc_evaluator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    agent_client: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    dft_submitter: Callable[..., dict[str, Any]] | None = None,
    model_update_handler: Callable[..., dict[str, Any]] | None = None,
    dft_budget: float | None = None,
    invocation_id: str | None = None,
    execution_mode: str = "autonomous",
    human_feedback: str | dict[str, Any] | None = None,
    replay_record: dict[str, Any] | None = None,
    config_session: dict[str, Any] | None = None,
    tool_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one non-blocking search/acquisition cycle.

    The function does not train a model or execute DFT itself. It connects the
    existing interfaces, records standard pending/completed DFT tasks, and calls
    ``model_update_handler`` only when the validated retraining trigger fires.
    """
    if not config_version:
        raise ValueError("config_version 不能为空")
    search = run_pipeline(
        manager,
        phase_references,
        _active_search_config(run_config),
        state=state,
        dispatcher=dispatcher,
        recovered_results=recovered_results,
    )
    current = deepcopy(search["state"])
    if recovered_results:
        current = update_dft_action_results(current, recovered_results)
        current = _record_recovered_dft(current, recovered_results)
    if candidates is None:
        qbc_candidates, excluded_candidates = _ledger_candidates(manager, current)
        current["qbc_candidate_filter"] = {
            "eligible_count": len(qbc_candidates),
            "excluded": excluded_candidates,
        }
    else:
        qbc_candidates = deepcopy(candidates)
    decision_config = _decision_config(run_config)
    remaining_dft_budget = _remaining_dft_budget(current, run_config, dft_budget)
    cycle_id = invocation_id or f"cycle-{int(current.get('iteration', 0)):06d}-{config_version}"
    effective_dft_submitter = dft_submitter or (
        _dispatcher_dft_submitter(dispatcher) if dispatcher is not None else None
    )
    acquisition = run_qbc_dft_decision_flow(
        qbc_candidates,
        current,
        config=decision_config,
        config_version=config_version,
        dft_parameters=deepcopy((run_config.get("dft") or {}).get("parameters") or {}),
        context={
            "manager": manager,
            "phase_references": phase_references,
            "phase_diagram_state": current.get("phase_diagrams"),
            "decision_memory": current.get("decision_memory"),
            "rewards": current.get("rewards", []),
            "action_records": current.get("action_records", []),
            "remaining_dft_budget": remaining_dft_budget,
            "search_history": current.get("decisions", []),
            "training_coverage": current.get("training_coverage"),
        },
        agent_client=agent_client,
        qbc_evaluator=qbc_evaluator,
        dft_submitter=effective_dft_submitter,
        invocation_id=cycle_id,
        execution_mode=execution_mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
        execution_session=config_session,
        tool_registry=tool_registry,
    )
    current = _merge_dft_submissions(
        manager,
        acquisition["state"],
        acquisition.get("submissions", []),
        config_version=config_version,
    )
    model_update = _run_model_update(
        acquisition.get("retrain"),
        current,
        manager,
        run_config,
        model_update_handler,
    )
    if model_update is not None:
        if isinstance(model_update.get("state"), dict):
            current = deepcopy(model_update["state"])
        current.setdefault("model_update_history", []).append(
            {key: deepcopy(value) for key, value in model_update.items() if key != "state"}
        )
    convergence_evidence = build_convergence_evidence(
        current,
        manager,
        budget_remaining=remaining_dft_budget,
    )
    current.update(convergence_evidence)
    convergence = check_global_convergence(
        current,
        rules=_convergence_rules(run_config),
    )
    if convergence.get("status") == "finished":
        from data_layer.memory.maybe_draft_converged_skill import maybe_draft_converged_skill
        convergence["knowledge_draft"] = maybe_draft_converged_skill(current, run_config, convergence)
    current["convergence"] = convergence
    _persist_cycle(manager, current, run_config)
    return {
        "status": acquisition.get("status", search.get("status", "completed")),
        "search": search,
        "acquisition": {key: value for key, value in acquisition.items() if key != "state"},
        "model_update": model_update,
        "convergence": convergence,
        "state": current,
        "manager": manager,
    }


def _active_search_config(run_config: dict[str, Any]) -> dict[str, Any]:
    """Give QBC exclusive ownership of DFT submission in active-learning runs."""
    config = deepcopy(run_config)
    workflow = ((config.get("system_config") or {}).get("calculation_workflow") or {})
    for stage in workflow.get("stages") or []:
        if stage.get("name") in {"dft_single_point", "dft_relax"}:
            stage["enabled"] = False
    return config


def _ledger_candidates(manager: Any, state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates = []
    excluded = []
    pending_dft_ids = {
        item.get("structure_id") or item.get("object_id")
        for item in state.get("pending_tasks", [])
        if item.get("stage") in {"dft_single_point", "dft_relax"}
        and item.get("status") in {"pending", "running"}
    }
    for structure_id, structure in sorted(manager.data.get("structures", {}).items()):
        branch = manager.data.get("branches", {}).get(structure.get("branch_id"), {})
        metadata = structure.get("metadata") or {}
        reason = _qbc_exclusion_reason(structure_id, structure, metadata, pending_dft_ids)
        if reason:
            excluded.append({"structure_id": structure_id, "reason": reason})
            continue
        candidates.append(
            {
                "candidate_id": structure_id,
                "structure_id": structure_id,
                "branch_id": structure.get("branch_id"),
                "structure_path": structure.get("source_path"),
                "atom_count": metadata.get("atom_count") or sum((structure.get("composition") or branch.get("composition") or {}).values()) or None,
                "composition": structure.get("composition") or branch.get("composition"),
                "phase": branch.get("P"),
                "P": branch.get("P"),
                "qbc": deepcopy(metadata.get("qbc")),
                "predicted_Ehull": metadata.get("predicted_Ehull", metadata.get("ehull")),
                "estimated_cost": deepcopy(metadata.get("estimated_cost")),
                "duplicate_of": metadata.get("duplicate_of"),
            }
        )
    return candidates, excluded


def _qbc_exclusion_reason(structure_id, structure, metadata, pending_dft_ids):
    if metadata.get("duplicate_of"):
        return "duplicate_structure"
    if structure_id in pending_dft_ids:
        return "dft_already_pending"
    history = structure.get("stage_history") or {}
    if history.get("dft_single_point") or history.get("dft_relax"):
        return "dft_already_completed"
    return None


def _decision_config(run_config: dict[str, Any]) -> dict[str, Any]:
    config = default_dft_decision_config()
    supplied = deepcopy(run_config.get("qbc") or {})
    config.update({key: value for key, value in supplied.items() if key in config})
    if run_config.get("budgets"):
        costs = default_dft_decision_config(budgets=run_config["budgets"])
        config["action_costs"] = costs["action_costs"]
        config["cost_model"] = costs["cost_model"]
        config["budget_limits"] = costs["budget_limits"]
    config["mode"] = supplied.get("mode", supplied.get("selection_mode", config["mode"]))
    if supplied.get("force_max_threshold") is not None:
        config["extreme_uncertainty"]["f_std_max"] = supplied["force_max_threshold"]
    if supplied.get("energy_threshold") is not None:
        config["extreme_uncertainty"]["energy_std"] = supplied["energy_threshold"]
    return config


def _convergence_rules(run_config):
    supplied = deepcopy(run_config.get("convergence") or {})
    aliases = {
        "hull_tolerance": "hull_change_tolerance",
        "stable_rounds": "stable_iterations",
        "coverage_threshold": "minimum_coverage_fraction",
    }
    return {
        aliases.get(key, key): value
        for key, value in supplied.items()
        if value is not None
    }


def _dispatcher_dft_submitter(dispatcher):
    def submit(*, decision, dft_parameters, context):
        structure_id = decision["candidate_id"]
        stage = "dft_single_point" if decision["action"] == "DFT_SINGLE_POINT" else "dft_relax"
        task_key = decision["task_key"]
        task_id = f"TASK-{hashlib.sha256(task_key.encode()).hexdigest()[:12]}"
        return dispatcher(
            normalize_task(
                {
                    "task_id": task_id,
                    "task_key": task_key,
                    "structure_id": structure_id,
                    "object_id": structure_id,
                    "stage": stage,
                    "status": "pending",
                    "config_version": decision["config_version"],
                    "parameters": deepcopy(dft_parameters),
                    "budget": {"planned_relative_cost": decision["relative_cost"]},
                }
            )
        )

    return submit


def _remaining_dft_budget(state: dict[str, Any], config: dict[str, Any], explicit: float | None) -> float:
    if explicit is not None:
        if explicit < 0:
            raise ValueError("dft_budget 不能为负")
        return float(explicit)
    limits = (state.get("budget_limits") or config.get("budgets") or {}).get("stage_limits", {})
    total = sum(float((limits.get(stage) or {}).get("max_cost") or 0) for stage in ("dft_single_point", "dft_relax"))
    return max(0.0, total - float(state.get("used_dft_cost", 0)) - float(state.get("reserved_dft_cost", 0)))


def _record_recovered_dft(state: dict[str, Any], recovered_results: list[dict[str, Any]]) -> dict[str, Any]:
    current = deepcopy(state)
    records = current.setdefault("new_dft_records", [])
    known = {item.get("task_id") for item in records}
    for result in recovered_results:
        if result.get("stage") not in {"dft_single_point", "dft_relax"} or result.get("status") != "completed":
            continue
        if result.get("task_id") in known:
            continue
        outputs = result.get("outputs") or result.get("result") or {}
        records.append(
            {
                **deepcopy(result),
                "energy": result.get("energy", outputs.get("energy")),
                "converged": result.get("converged", outputs.get("converged")),
                "checks_passed": result.get("checks_passed", outputs.get("checks_passed", True)),
            }
        )
        known.add(result.get("task_id"))
    return current


def _merge_dft_submissions(manager: Any, state: dict[str, Any], submissions: list[dict[str, Any]], *, config_version: str) -> dict[str, Any]:
    current = deepcopy(state)
    pending = current.setdefault("pending_tasks", [])
    processed = current.setdefault("processed_task_ids", [])
    for submission in submissions:
        action = submission.get("action")
        if action not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
            continue
        structure_id = submission.get("structure_id") or submission.get("candidate_id")
        if structure_id not in manager.data.get("structures", {}):
            continue
        stage = "dft_single_point" if action == "DFT_SINGLE_POINT" else "dft_relax"
        task_key = submission["task_key"]
        task_id = submission.get("task_id") or f"TASK-{hashlib.sha256(task_key.encode()).hexdigest()[:12]}"
        task = normalize_task(
            {
                "task_id": task_id,
                "task_key": task_key,
                "structure_id": structure_id,
                "object_id": structure_id,
                "stage": stage,
                "status": "pending",
                "config_version": config_version,
                "parameters": deepcopy(submission.get("parameters") or {}),
                "budget": {"planned_relative_cost": submission.get("relative_cost")},
            }
        )
        scientific_outputs = deepcopy(submission.get("outputs") or {})
        for key in ("energy", "energy_unit", "converged", "dft_code", "dft_version", "dft_settings", "ehull", "ehull_unit", "hull_reference_version"):
            if key in submission and key not in scientific_outputs:
                scientific_outputs[key] = submission[key]
        result = normalize_task_result(task, {**submission, "outputs": scientific_outputs})
        if result["status"] in {"pending", "running"}:
            pending[:] = [item for item in pending if item.get("task_key") != task_key]
            pending.append(result)
        else:
            pending[:] = [item for item in pending if item.get("task_key") != task_key]
            collect_calculation_results(manager, structure_id, result)
            if task_id not in processed:
                processed.append(task_id)
    return current


def _run_model_update(trigger, state, manager, config, handler):
    if not trigger or trigger.get("action") != "RETRAIN_MLIP":
        return None
    if handler is None:
        return {"status": "not_configured", "reason": "model_update_handler_not_configured", "trigger": deepcopy(trigger)}
    try:
        return handler(trigger=deepcopy(trigger), state=deepcopy(state), manager=manager, config=deepcopy(config))
    except Exception as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}", "trigger": deepcopy(trigger)}


def _persist_cycle(manager: Any, state: dict[str, Any], config: dict[str, Any]) -> None:
    ledger_path = config.get("ledger_path")
    if ledger_path:
        manager.save(ledger_path)
    state_path = config.get("state_path")
    if state_path:
        output = Path(state_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        temporary.replace(output)

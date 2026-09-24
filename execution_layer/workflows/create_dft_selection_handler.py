"""Create DFT child tasks from one already-approved Agent tool action."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from scientific_layer.qbc.build_candidate_metrics import build_qbc_candidate_metrics
from execution_layer.budget.reserve_dft_actions import reserve_dft_actions
from execution_layer.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def create_dft_selection_handler(*, candidates_provider=None, qbc_evaluator=None):
    """Return the default handler without introducing a second policy gate."""

    def handler(*, action, context):
        state = deepcopy(context.get("event_state") or {})
        candidates = (
            candidates_provider(action, context)
            if callable(candidates_provider)
            else _state_candidates(state, context.get("manager"), action)
        )
        metrics_result = build_qbc_candidate_metrics(candidates, qbc_evaluator=qbc_evaluator)
        config = deepcopy(context["effective_config"].get("qbc") or {})
        config["selection_policy"] = deepcopy(
            (context["effective_config"].get("dft") or {}).get("selection") or {})
        config["budget_limits"] = deepcopy(context["effective_config"]["budgets"])
        remaining = _remaining_dft_budget(state, config)
        parameters = deepcopy(action.get("parameters") or {})
        proposal = {
            "status": "completed",
            "decisions": deepcopy(parameters.get("decisions") or []),
            "global_action": parameters.get("global_action", "CONTINUE_DATA_COLLECTION"),
            "reason": action.get("reason"),
            "source": action.get("decision_source", "llm_agent"),
        }
        validation = validate_dft_agent_decisions(
            proposal, metrics_result["metrics"], state, config=config,
            config_version=context["config_version"], remaining_budget=remaining,
        )
        if not validation["valid"]:
            return {
                "status": "rejected", "state": state, "metrics": metrics_result,
                "validation": validation, "tasks": [],
            }
        state = reserve_dft_actions(state, validation["accepted"])
        tasks = []
        confirmed_parameters = deepcopy((context["effective_config"].get("dft") or {}).get("parameters") or {})
        for decision in validation["accepted"]:
            if decision["action"] not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
                continue
            stage = "dft_single_point" if decision["action"] == "DFT_SINGLE_POINT" else "dft_relax"
            task = {
                "task_id": "DFT-" + hashlib.sha256(decision["task_key"].encode()).hexdigest()[:12],
                "task_key": decision["task_key"],
                "object_id": decision["candidate_id"],
                "structure_id": decision["candidate_id"],
                "branch_id": decision.get("branch_id"),
                "stage": stage,
                "status": "pending",
                "planned_relative_cost": decision["relative_cost"],
                "parameters": confirmed_parameters,
                "config_version": context["config_version"],
                "selection_reason": decision.get("reason"),
                "selection_source": proposal["source"],
            }
            state.setdefault("tasks", []).append(task)
            state.setdefault("pending_tasks", []).append(task)
            tasks.append(task)
        state.setdefault("qbc_selection_history", []).append({
            "task_key": action.get("task_key"), "metrics": metrics_result,
            "accepted": deepcopy(validation["accepted"]),
            "rejected": deepcopy(validation["rejected"]),
        })
        return {
            "status": "completed", "state": state, "metrics": metrics_result,
            "validation": validation, "tasks": tasks,
            "retrain_requested": validation["global_action"] == "RETRAIN_MLIP",
        }

    return handler


def _state_candidates(state, manager, action):
    available = deepcopy(state.get("qbc_candidates") or [])
    if not available and manager is not None:
        for structure_id, record in manager.data.get("structures", {}).items():
            metadata = record.get("metadata") or {}
            if metadata.get("qbc") is None:
                continue
            available.append({
                "candidate_id": structure_id,
                "structure_id": structure_id,
                "branch_id": record.get("branch_id"),
                "composition": deepcopy(record.get("composition")),
                "atom_count": metadata.get("atom_count"),
                "predicted_Ehull": metadata.get("predicted_Ehull"),
                "qbc": deepcopy(metadata.get("qbc")),
            })
    requested = set(action.get("target_ids") or [])
    return [row for row in available if not requested or (row.get("candidate_id") or row.get("structure_id")) in requested]


def _remaining_dft_budget(state, config):
    limits = config.get("budget_limits") or {}
    total = limits.get("total_relative_cost")
    if total is None:
        return float("inf")
    used = float((state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
    reserved = float(state.get("reserved_relative_cost", 0) or 0)
    return max(0.0, float(total) - used - reserved)

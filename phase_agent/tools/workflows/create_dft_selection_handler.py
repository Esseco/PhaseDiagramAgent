"""Create DFT child tasks from one already-approved Agent tool action."""

from __future__ import annotations

from phase_agent.interface_compatibility import call_with_optimizations

from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.science.qbc.build_candidate_metrics import build_qbc_candidate_metrics
from phase_agent.tools.budget.reserve_dft_actions import reserve_dft_actions
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def registered_retry_input_ids(action, state):
    """Validate explicit input-only scope; never create or re-reserve retries."""
    ids = (action.get("parameters") or {}).get("registered_retry_input_ids", [])
    if (
        not isinstance(ids, list)
        or any(not isinstance(x, str) for x in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("invalid_registered_retry_input_ids")
    tasks = {row.get("task_id"): row for row in state.get("tasks") or []}
    for task_id in ids:
        task = tasks.get(task_id) or {}
        if not (
            task.get("retry_of")
            and task.get("stage") in {"dft_single_point", "dft_relax"}
            and task.get("status") == "pending"
            and not task.get("slurm_batch_id")
            and task.get("model_version") == state.get("active_model_version")
            and task.get("config_version") == state.get("confirmed_config_version")
        ):
            raise ValueError("registered_retry_not_eligible_for_inputs:" + task_id)
    return list(ids)


def create_dft_selection_handler(*, candidates_provider=None, qbc_evaluator=None):
    """Return the default handler without introducing a second policy gate."""

    def handler(*, action, context):
        state = _input_working_state(context)
        parameters = action.get("parameters") or {}
        try:
            retry_input_ids = registered_retry_input_ids(action, state)
        except ValueError as error:
            return {"status": "rejected", "state": state, "tasks": [], "reason": str(error)}
        prepare_after_approval = bool(parameters.get("prepare_inputs_after_approval"))
        if context.get("execution_mode") == "interactive" and not (
            parameters.get("dft_input_preview") or prepare_after_approval
        ):
            return {
                "status": "rejected",
                "state": state,
                "tasks": [],
                "reason": "旧 DFT 方案缺少输入数量与成本预览，请拒绝后重新获取方案。",
            }
        candidates = (
            candidates_provider(action, context)
            if callable(candidates_provider)
            else _state_candidates(state, context.get("manager"), action)
        )
        metrics_result = build_qbc_candidate_metrics(candidates, qbc_evaluator=qbc_evaluator)
        config = deepcopy(context["effective_config"].get("qbc") or {})
        config["selection_policy"] = deepcopy(
            (context["effective_config"].get("dft") or {}).get("selection") or {}
        )
        config["budget_limits"] = deepcopy(context["effective_config"]["budgets"])
        remaining = _remaining_dft_budget(state, config)
        parameters = deepcopy(parameters)
        proposal = {
            "status": "completed",
            "decisions": deepcopy(parameters.get("decisions") or []),
            "global_action": parameters.get("global_action", "CONTINUE_DATA_COLLECTION"),
            "reason": action.get("reason"),
            "source": action.get("decision_source", "llm_agent"),
        }
        validation = validate_dft_agent_decisions(
            proposal,
            metrics_result["metrics"],
            state,
            config=config,
            config_version=context["config_version"],
            remaining_budget=remaining,
        )
        if not validation["valid"] or validation["rejected"]:
            return {
                "status": "rejected",
                "state": state,
                "metrics": metrics_result,
                "validation": validation,
                "tasks": [],
            }
        if parameters.get("dft_input_preview") and validation["rejected"]:
            return {
                "status": "rejected",
                "state": state,
                "validation": validation,
                "reason": "批准后候选或预算变化；请重新确认方案。",
                "tasks": [],
            }
        from phase_agent.tools.workflows.dft_selection_plot import require_dft_selection_plot
        plot_state = {**state, "qbc_candidates": candidates}
        try:
            require_dft_selection_plot(plot_state, validation["accepted"])
        except (ValueError, OSError, TypeError, KeyError, ImportError) as error:
            return {"status": "rejected", "state": state, "tasks": [], "reason": "dft_selection_plot_required: " + str(error)}
        reviewed_settings = None
        review = action.get("dft_template_review")
        if context.get("execution_mode") == "interactive" and not prepare_after_approval:
            from phase_agent.tools.workflows.dft_template_review import validate_review

            try:
                reviewed_settings = validate_review(review, action, context["config_version"])
            except (ValueError, OSError) as error:
                return {"status": "rejected", "state": state, "tasks": [], "reason": str(error)}
        purposes = {
            row["candidate_id"]: row["purpose"]
            for row in (action.get("scientific_decision") or {}).get("selection_purposes") or []
        }
        candidates_by_id = {
            row.get("candidate_id") or row.get("structure_id"): row for row in candidates
        }
        existing_task_ids = {
            row.get("task_id")
            for row in (state.get("tasks") or []) + (state.get("pending_tasks") or [])
        }
        input_structure_sha256 = {}
        try:
            for decision in validation["accepted"]:
                if decision["action"] not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
                    continue
                sid = decision["candidate_id"]
                selected = candidates_by_id.get(sid) or {}
                path = selected.get("structure_path")
                if not path:
                    raise ValueError("dft_input_structure_missing:" + sid)
                actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                expected = decision.get("input_structure_sha256")
                candidate_sha = selected.get("structure_sha256")
                if (expected is not None and actual != expected) or (
                    candidate_sha is not None
                    and (not isinstance(candidate_sha, str) or actual != candidate_sha.lower())
                ):
                    raise ValueError("dft_input_structure_sha256_mismatch:" + sid)
                task_id = "DFT-" + hashlib.sha256(decision["task_key"].encode()).hexdigest()[:12]
                if task_id in existing_task_ids:
                    raise ValueError("duplicate_dft_task_id:" + task_id)
                existing_task_ids.add(task_id)
                input_structure_sha256[decision["task_key"]] = actual
        except (ValueError, OSError, TypeError) as error:
            return {
                "status": "rejected", "state": state, "tasks": [],
                "metrics": metrics_result, "validation": validation, "reason": str(error),
            }
        state = call_with_optimizations(
            reserve_dft_actions, state, validation["accepted"], _optimizations={"_owned": True}
        )
        tasks = []
        confirmed_parameters = deepcopy(
            (context["effective_config"].get("dft") or {}).get("parameters") or {}
        )
        if reviewed_settings is not None:
            confirmed_parameters = deepcopy(reviewed_settings["parameters"])
        for decision in validation["accepted"]:
            if decision["action"] not in {"DFT_SINGLE_POINT", "DFT_RELAX"}:
                continue
            stage = "dft_single_point" if decision["action"] == "DFT_SINGLE_POINT" else "dft_relax"
            task = {
                "task_id": "DFT-" + hashlib.sha256(decision["task_key"].encode()).hexdigest()[:12],
                "task_key": decision["task_key"],
                "input_structure_sha256": input_structure_sha256[decision["task_key"]],
                "object_id": decision["candidate_id"],
                "structure_id": (candidates_by_id.get(decision["candidate_id"]) or {}).get("structure_id") or decision["candidate_id"],
                "branch_id": decision.get("branch_id")
                or next(
                    (
                        row.get("branch_id")
                        for row in candidates
                        if (row.get("candidate_id") or row.get("structure_id"))
                        == decision["candidate_id"]
                    ),
                    None,
                ),
                "stage": stage,
                "independent_audit": decision.get("independent_audit", False),
                "status": "pending",
                "planned_relative_cost": decision["relative_cost"],
                "parameters": confirmed_parameters,
                "config_version": context["config_version"],
                "selection_reason": decision.get("reason"),
                "selection_source": proposal["source"],
                "parent_decision_id": context.get("approval_record_id"),
                "generation_cycle": len(state.get("generation_history") or []),
                "model_version": (context["effective_config"].get("mlip") or {}).get("version")
                or (context["effective_config"].get("mlip") or {}).get("name")
                or state.get("active_model_version"),
                "upload_operation_id": hashlib.sha256(
                    str(
                        action.get("task_key")
                        or context.get("approval_record_id")
                        or decision["task_key"]
                    ).encode()
                ).hexdigest()[:12],
            }
            selected = next(
                (
                    row
                    for row in candidates
                    if (row.get("candidate_id") or row.get("structure_id"))
                    == decision["candidate_id"]
                ),
                {},
            )
            if selected.get("source_search_task_id"):
                task["source_search_task_id"] = selected["source_search_task_id"]
            if selected.get("structure_path"):
                task["structure_path"] = selected["structure_path"]
            if reviewed_settings is not None:
                task["reviewed_submit_script"] = reviewed_settings["submit_script"]
                task["dft_template_digest"] = review["digest"]
            state.setdefault("tasks", []).append(task)
            state.setdefault("pending_tasks", []).append(task)
            task["selection_purpose"] = purposes.get(decision["candidate_id"])
            task["scientific_parent_task_key"] = action.get("task_key")
            tasks.append(task)
        state.setdefault("qbc_selection_history", []).append(
            {
                "task_key": action.get("task_key"),
                "metrics": metrics_result,
                "accepted": deepcopy(validation["accepted"]),
                "rejected": deepcopy(validation["rejected"]),
            }
        )
        if (parameters.get("dft_input_preview") or prepare_after_approval) and tasks:
            from phase_agent.tools.local.prepare_dft_upload_batches import (
                prepare_dft_upload_batches,
            )

            prepared = call_with_optimizations(
                prepare_dft_upload_batches,
                action={"target_ids": [row["task_id"] for row in tasks] + retry_input_ids},
                context={**context, "event_state": state},
                _optimizations={"_owned": True},
            )
            from phase_agent.tools.workflows.dft_selection_artifacts import preparation_artifacts

            artifacts = preparation_artifacts(
                action, state, tasks, validation, prepared, context.get("manager")
            )
            state["qbc_selection_history"][-1]["artifacts"] = artifacts
            if prepared.get("state"):
                prepared["state"]["qbc_selection_history"][-1]["artifacts"] = artifacts
            return {**prepared, "tasks": tasks, "task_count": len(tasks), "validation": validation}
        from phase_agent.tools.workflows.dft_selection_artifacts import preparation_artifacts

        artifacts = preparation_artifacts(
            action, state, tasks, validation, manager=context.get("manager")
        )
        state["qbc_selection_history"][-1]["artifacts"] = artifacts
        return {
            "status": "completed",
            "state": state,
            "metrics": metrics_result,
            "validation": validation,
            "tasks": tasks,
            "retrain_requested": validation["global_action"] == "RETRAIN_MLIP",
        }

    # Runtime capability: safe to adopt only inside an already-owned action loop.
    handler._returns_owned_state = True
    return handler


def _state_candidates(state, manager, action):
    available = deepcopy(state.get("qbc_candidates") or [])
    if not available and manager is not None:
        for structure_id, record in manager.data.get("structures", {}).items():
            metadata = record.get("metadata") or {}
            if metadata.get("qbc") is None:
                continue
            available.append(
                {
                    "candidate_id": structure_id,
                    "structure_id": structure_id,
                    "branch_id": record.get("branch_id"),
                    "composition": deepcopy(record.get("composition")),
                    "atom_count": metadata.get("atom_count"),
                    "predicted_Ehull": metadata.get("predicted_Ehull"),
                    "qbc": deepcopy(metadata.get("qbc")),
                }
            )
    requested = set(action.get("target_ids") or [])
    if requested:
        # Review the full decision table without adding extra calculation targets.
        requested.update(
            row.get("candidate_id")
            for row in (action.get("parameters") or {}).get("decisions") or []
            if isinstance(row, dict) and row.get("action") in {"DEFER", "REJECT"}
        )
    return [
        row
        for row in available
        if not requested or (row.get("candidate_id") or row.get("structure_id")) in requested
    ]


def _remaining_dft_budget(state, config):
    limits = config.get("budget_limits") or {}
    total = limits.get("total_relative_cost")
    from phase_agent.tools.budget.check_budget import budget_is_hard

    if total is None or not budget_is_hard(limits):
        return float("inf")
    used = float((state.get("budget_usage") or {}).get("total_relative_cost", 0) or 0)
    reserved = float(state.get("reserved_relative_cost", 0) or 0)
    return max(0.0, float(total) - used - reserved)


def _input_working_state(context):
    source = context.get("event_state") or {}
    if not (context.get("_event_state_owned") and context.get("saved_input_only")):
        return deepcopy(source)
    # The guarded idle input path reads scientific history. Detach every ledger
    # mutated by selection, reservation and staging so failures cannot change
    # the outer action state. Other histories remain read-only until commit.
    current = dict(source)
    for key in (
        "tasks",
        "pending_tasks",
        "effective_decisions",
        "budget_reservations",
        "qbc_selection_history",
        "slurm_batches",
        "upload_layout",
    ):
        if key in source:
            current[key] = deepcopy(source[key])
    return current

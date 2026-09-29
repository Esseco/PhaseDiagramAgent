"""调度一轮生成、选择、任务派发、回收和反馈。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from execution_layer.budget.check_budget import check_budget
from execution_layer.budget.estimate_stage_cost import estimate_stage_cost
from execution_layer.budget.record_budget_usage import record_budget_usage
from data_layer.ledger.collect_calculation_results import collect_calculation_results
from decision_layer.calculation.decide_next_calculation import decide_next_calculation
from execution_layer.state.normalize_task import normalize_task
from execution_layer.state.normalize_task_result import normalize_task_result
from execution_layer.workflows.run_branch_generation import run_branch_generation
from analysis_layer.feedback.calculate_search_reward import calculate_search_reward
from decision_layer.strategy.choose_generation_strategy import choose_generation_strategy
from analysis_layer.phase.update_phase_diagram import update_phase_diagram
from analysis_layer.phase.ensure_phase_identification import ensure_phase_identification


def run_search_iteration(
    manager: Any,
    phase_references: dict[str, Any],
    state: dict[str, Any] | str | Path | None,
    *,
    structure_directory: str | Path,
    total_quota: int,
    batch_size: int,
    initial_states_per_branch: int,
    seed: int,
    dispatcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    recovered_results: list[dict[str, Any]] | None = None,
    generation_metrics: dict[str, Any] | None = None,
    generation_options: dict[str, Any] | None = None,
    state_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    phase_diagram_directory: str | Path | None = None,
    task_versions: dict[str, str | None] | None = None,
    budget_limits: dict[str, Any] | None = None,
    system_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行一个非阻塞迭代；未完成任务保留在 pending_tasks。"""
    restored = (
        json.loads(Path(state).read_text(encoding="utf-8"))
        if isinstance(state, (str, Path))
        else state
    )
    current = copy.deepcopy(
        restored
        or {
            "iteration": 0,
            "pending_tasks": [],
            "processed_task_ids": [],
            "decisions": [],
            "phase_records": [],
            "phase_diagrams": {},
            "reward_states": {},
            "rewards": [],
        }
    )
    if budget_limits is not None:
        current["budget_limits"] = copy.deepcopy(budget_limits)
    effective_budget_limits = current.get("budget_limits")
    for key, default in (
        ("phase_records", []),
        ("phase_diagrams", {}),
        ("reward_states", {}),
        ("rewards", []),
        ("budget_usage", {}),
    ):
        current.setdefault(key, default)
    recovered = []
    recovered_for_feedback = []
    for result in recovered_results or []:
        task_id = result.get("task_id")
        if not task_id or task_id in current["processed_task_ids"]:
            continue
        if result.get("status") in {"pending", "running"}:
            _upsert_task(current["pending_tasks"], result)
            collect_calculation_results(
                manager, result["structure_id"], result, ledger_path=None
            )
            continue
        collected = collect_calculation_results(
            manager, result["structure_id"], result, ledger_path=None
        )
        recovered.append(collected)
        if collected.get("phase_record"):
            phase_record = collected["phase_record"]
            if phase_record["record_id"] not in {
                item.get("record_id") for item in current["phase_records"]
            }:
                current["phase_records"].append(phase_record)
                recovered_for_feedback.append({**result, "phase_record": phase_record})
        current["processed_task_ids"].append(task_id)
        current["pending_tasks"] = [
            item for item in current["pending_tasks"] if item.get("task_id") != task_id
        ]
    paused_for_budget = current.get("run_status") == "budget_exhausted"
    allocation = choose_generation_strategy(
        generation_metrics or {}, total_quota=0 if paused_for_budget else total_quota
    )
    options = dict(generation_options or {})
    effective_system_config = system_config or manager.data.get("system_config") or {}
    options.setdefault("system_config", effective_system_config)
    generated = run_branch_generation(
        manager,
        phase_references,
        structure_directory=structure_directory,
        quotas=allocation["quotas"],
        batch_size=batch_size,
        initial_states_per_branch=initial_states_per_branch,
        seed=seed,
        ledger_path=None,
        **options,
    )
    dispatched = []
    versions = dict(task_versions or {})
    for structure_id, structure_record in sorted(manager.data["structures"].items()):
        calculation_decision = decide_next_calculation(
            structure_record,
            rules=effective_system_config.get("calculation_workflow"),
        )
        if calculation_decision["action"] not in {"run", "retry"}:
            continue
        stage = calculation_decision["stage"]
        stage_spec = next(
            (
                item
                for item in effective_system_config.get("calculation_workflow", {}).get("stages", [])
                if item.get("name") == stage
            ),
            {},
        )
        version = versions.get(stage, stage_spec.get("model_version"))
        task_key = f"{structure_id}:{stage}:{version or 'default'}"
        if any(item.get("task_key") == task_key for item in current["pending_tasks"]):
            continue
        attempts = (structure_record.get("metadata") or {}).get(
            "calculation_attempts", []
        )
        attempt_index = 1 + sum(item.get("stage") == stage for item in attempts)
        digest = hashlib.sha256(
            f"{task_key}:{attempt_index}".encode("utf-8")
        ).hexdigest()[:12]
        request = normalize_task({
            "task_id": f"TASK-{digest}",
            "task_key": task_key,
            "object_id": structure_id,
            "structure_id": structure_id,
            "status": "pending",
            "stage": stage,
            "calculation_version": version,
            "model_version": version,
            "parameters": copy.deepcopy(stage_spec.get("parameters") or {}),
            "budget": copy.deepcopy(stage_spec.get("budget") or {}),
            "structure_path": structure_record.get("source_path"),
            "decision": calculation_decision,
        })
        if effective_budget_limits:
            stage_config = effective_budget_limits.get("stage_limits", {}).get(stage, {})
            cost_estimate = estimate_stage_cost(stage, atom_count=structure_record.get("atom_count"), budgets=effective_budget_limits, mc_steps=request["parameters"].get("segment_budget"))
            planned_cost = cost_estimate["value"]
            request["cost_estimate"] = cost_estimate
            budget_check = check_budget(
                current,
                {"stage": stage, "tasks": 1, "relative_cost": planned_cost},
                effective_budget_limits,
            )
            if not budget_check["allowed"]:
                current["run_status"] = "budget_exhausted"
                current["pause_reasons"] = budget_check["reasons"]
                continue
            request["planned_relative_cost"] = planned_cost
            request["budget"]["planned_relative_cost"] = planned_cost
        response = normalize_task_result(request, dispatcher(request)) if dispatcher is not None else request
        if response.get("status") in {"pending", "running"}:
            _upsert_task(current["pending_tasks"], response)
            collect_calculation_results(
                manager, response["structure_id"], response, ledger_path=None
            )
        elif response.get("status") in {
            "completed",
            "failed",
            "paused",
            "not_configured",
        }:
            collected = collect_calculation_results(
                manager, response["structure_id"], response, ledger_path=None
            )
            current["processed_task_ids"].append(response["task_id"])
            if collected.get("phase_record"):
                phase_record = collected["phase_record"]
                if phase_record["record_id"] not in {
                    item.get("record_id") for item in current["phase_records"]
                }:
                    current["phase_records"].append(phase_record)
                    recovered_for_feedback.append(
                        {**response, "phase_record": phase_record}
                    )
        dispatched.append(response)
        if effective_budget_limits:
            current = record_budget_usage(
                current,
                {
                    "stage": stage,
                    "tasks": 1,
                    "relative_cost": request["planned_relative_cost"],
                },
            )

    feedback = []
    needs_phase_retry = any(
        row.get("status") == "pending_phase_identification"
        for row in current["phase_records"]
    )
    if recovered_for_feedback or needs_phase_retry:
        phase_cache_path = (Path(state_path).with_name("phase_identification_cache.json")
                            if state_path is not None else None)
        current, _ = ensure_phase_identification(
            current, manager, phase_references=phase_references,
            cache_path=phase_cache_path,
        )
        old_diagrams = copy.deepcopy(current["phase_diagrams"])
        diagrams = update_phase_diagram(
            current["phase_records"], output_directory=phase_diagram_directory
        )["diagrams"]
        current["phase_diagrams"] = diagrams
        for method in ("mlip", "dft"):
            relevant = [
                item
                for item in recovered_for_feedback
                if str(item["phase_record"].get("energy_method", "")).lower() == method
            ]
            before, after = old_diagrams.get(method), diagrams.get(method)
            if not relevant or not before or after.get("status") != "completed":
                continue
            batch_ids = sorted(
                {str(item.get("batch_id", "unbatched")) for item in relevant}
            )
            batch_id = (
                f"iteration-{current['iteration'] + 1}:{method}:{'+'.join(batch_ids)}"
            )
            reward = calculate_search_reward(
                before,
                after,
                batch_id=batch_id,
                task_ids=[item["task_id"] for item in relevant],
                actual_cost=sum(
                    _cost_value(item.get("actual_cost")) for item in relevant
                ),
                reward_state=current["reward_states"].get(method),
            )
            reward["energy_method"] = method
            reward["energy_basis_id"] = after.get("energy_basis_id")
            reward["previous_version"] = before.get("version")
            reward["current_version"] = after.get("version")
            current["reward_states"][method] = reward["state"]
            current["rewards"].append(
                {key: value for key, value in reward.items() if key != "state"}
            )
            feedback.append(reward)
    current["iteration"] += 1
    decision = {
        "iteration": current["iteration"],
        "seed": seed,
        "allocation": allocation,
        "generated_count": len(generated["registered"]),
        "recovered_count": len(recovered),
        "pending_count": len(current["pending_tasks"]),
        "feedback_count": len(feedback),
        "run_status": current.get("run_status", "active"),
        "pause_reasons": current.get("pause_reasons", []),
    }
    current["decisions"].append(decision)
    if ledger_path is not None:
        manager.save(ledger_path)
    if state_path is not None:
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return {
        "state": current,
        "decision": decision,
        "generation": generated,
        "dispatched": dispatched,
        "recovered": recovered,
        "feedback": feedback,
    }


def _upsert_task(tasks, task):
    task = _persistent_task(task)
    for index, current in enumerate(tasks):
        if current.get("task_id") == task.get("task_id"):
            tasks[index] = task
            return
    tasks.append(task)


def _persistent_task(task):
    fields = (
        "task_id",
        "object_id",
        "structure_id",
        "stage",
        "status",
        "structure_path",
        "checkpoint",
        "error",
        "submitted_at",
        "task_key",
        "calculation_version",
        "model_version",
        "parameters",
        "budget",
        "result",
        "actual_cost",
        "planned_relative_cost",
    )
    result = {key: task.get(key) for key in fields if key in task}
    for key in ("structure_path", "checkpoint"):
        if result.get(key) is not None:
            result[key] = str(result[key])
    return result


def _cost_value(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "core_hours", "gpu_hours", "steps"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
    raise ValueError(f"无法从 actual_cost={value!r} 提取成本数值")

"""Read-only, attributable budget experience; never infer missing spend or optimal policy."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from statistics import median

from phase_agent.analysis.state.round_budget_evidence import numeric_cost
from phase_agent.tools.cost.extract_gpu_accounting import extract_gpu_accounting

TERMINAL = {"completed", "failed", "timeout", "cancelled"}
PARAMETERS = (
    "fmax",
    "relax_steps",
    "relax_cell",
    "max_mc_steps",
    "patience_steps",
    "min_improvement",
    "mc_budget",
    "dft_budget",
    "batch_size",
    "generation_plan",
)


def task_budget_observations(state):
    """Join task receipts and cost observations by explicit task ID only."""
    history = {row["task_id"]: row for row in state.get("cost_history") or [] if row.get("task_id")}
    output = []
    seen = set()
    for task in state.get("tasks") or []:
        identity = task.get("task_id")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        record = history.get(identity) or {}
        timing = task.get("runtime_observation") or record.get("runtime_observation") or {}
        source = timing.get("source")
        elapsed = numeric_cost(timing.get("elapsed_seconds"))
        # Runtime predictions have the same numeric shape; require observation provenance.
        if source not in {"task_start_end", "sacct", "slurm_sacct", "verified_scheduler_record"}:
            elapsed = None
        if timing.get("allocation_basis") == "whole_batch_do_not_add_to_task_slices":
            elapsed = None
        actual = numeric_cost(task.get("actual_cost"))
        if actual is None and record.get("actual_cost_known") is True:
            actual = numeric_cost(record.get("actual_cost"))
        if task.get("actual_cost_known") is False or "estimated" in str(
            task.get("actual_cost_basis") or ""
        ):
            actual = None
        gpu = extract_gpu_accounting(task)
        gpu_hours = numeric_cost(gpu.get("actual_gpu_core_hours"))
        # Never multiply a whole shared-job accounting record once per structure.
        accounting = task.get("job_accounting") or task.get("scheduler_accounting") or {}
        shared = bool(task.get("batch_id") and accounting)
        if shared and accounting.get("allocation_basis") != "task_time_slice_not_whole_batch":
            gpu_hours = None
        if "estimated" in str(gpu.get("source") or ""):
            gpu_hours = None
        params = task.get("parameters") or (task.get("worker_job") or {}).get("parameters") or {}
        output.append(
            {
                "task_id": identity,
                "stage": task.get("stage"),
                "status": task.get("status"),
                "model_version": task.get("model_version"),
                "config_version": task.get("config_version"),
                "parent_decision_id": task.get("parent_decision_id"),
                "atom_count": numeric_cost(
                    task.get("atom_count")
                    or record.get("atom_count")
                    or (task.get("outputs") or {}).get("atom_count")
                ),
                "parameters": {
                    key: deepcopy(task.get(key, params.get(key)))
                    for key in PARAMETERS
                    if task.get(key, params.get(key)) is not None
                },
                "actual_relative_cost": actual,
                "estimated_relative_cost": numeric_cost(
                    task.get("estimated_cost", record.get("estimated_cost"))
                ),
                "elapsed_seconds": elapsed,
                "runtime_source": source if elapsed is not None else None,
                "gpu_hours": gpu_hours,
                "gpu_source": gpu.get("source") if gpu_hours is not None else None,
                "backend": timing.get("backend") or task.get("backend"),
                "hardware": timing.get("hardware"),
                "gpu_count": numeric_cost(timing.get("gpu_count")),
                "cpu_count": numeric_cost(timing.get("cpu_count")),
                "stop_reason": task.get("stop_reason"),
                "actual_mc_steps": numeric_cost(
                    task.get("actual_mc_steps", timing.get("actual_mc_steps"))
                ),
                "energy_improvement_ev_per_atom": _finite_number(task.get("energy_improvement"))
                if task.get("energy_improvement_unit") == "eV/atom"
                else None,
            }
        )
    return output


def budget_experiences(state):
    """Actions without round_budget_review remain eligible; no task-order guessing."""
    tasks = {row["task_id"]: row for row in task_budget_observations(state)}
    parents = defaultdict(set)
    for row in tasks.values():
        if row.get("parent_decision_id"):
            parents[row["parent_decision_id"]].add(row["task_id"])
    experiences = []
    seen = set()
    for decision in state.get("action_records") or []:
        identity = decision.get("record_id")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        action = decision.get("final_action") or {}
        result = (decision.get("execution_result") or {}).get("result") or {}
        if not action or not isinstance(result, dict):
            continue
        ids = set(parents.get(identity) or [])
        ids.update(row.get("task_id") for row in result.get("tasks") or [] if isinstance(row, dict))
        linked = [tasks[key] for key in sorted(ids - {None}) if key in tasks]
        if not linked:
            continue
        complete = all(row["status"] in TERMINAL for row in linked)
        ids = {row["task_id"] for row in linked}
        rewards = [
            row for row in state.get("rewards") or [] if ids.intersection(row.get("task_ids") or [])
        ]
        versions = {row["model_version"] for row in linked}
        configurations = {row["config_version"] for row in linked}
        experiences.append(
            {
                "decision_id": identity,
                "tool": action.get("tool") or action.get("action_type"),
                "config_version": decision.get("config_version")
                or (next(iter(configurations)) if len(configurations) == 1 else None),
                "model_version": next(iter(versions)) if len(versions) == 1 else None,
                "parameters": {
                    key: deepcopy(value)
                    for key, value in (action.get("parameters") or {}).items()
                    if key in PARAMETERS
                },
                "proposed_budget": numeric_cost(action.get("budget")),
                "budget_meaning": "authorized_ceiling_not_measured_spend",
                "status": "complete" if complete else "partial",
                "task_count": len(linked),
                "completed_count": sum(row["status"] == "completed" for row in linked),
                "failed_count": sum(row["status"] in {"failed", "timeout"} for row in linked),
                "actual_relative_cost": _complete_sum(linked, "actual_relative_cost")
                if complete
                else None,
                "serial_task_seconds": _complete_sum(linked, "elapsed_seconds")
                if complete
                else None,
                "gpu_hours": _complete_sum(linked, "gpu_hours") if complete else None,
                "measurement_counts": {
                    field: sum(row[field] is not None for row in linked)
                    for field in ("actual_relative_cost", "elapsed_seconds", "gpu_hours")
                },
                "task_observations": linked,
                "observed_rewards": [
                    {
                        key: deepcopy(row.get(key))
                        for key in (
                            "batch_id",
                            "reward",
                            "ehull_improvement",
                            "new_stable_entries",
                            "model_version",
                            "energy_method",
                            "energy_basis_id",
                        )
                    }
                    for row in rewards
                ],
                "evidence_refs": [
                    f"decision_outcome:{identity}",
                    *[f"task:{key}" for key in sorted(ids)],
                ],
                "interpretation": "Versioned observations; unexecuted alternatives unknown. Serial task seconds are not parallel wall time. No optimum or causal savings inferred.",
            }
        )
    return experiences


def budget_experience_context(state, *, limit=8, character_budget=12000):
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("budget experience limit must be between 1 and 10")
    if type(character_budget) is not int or character_budget < 1:
        raise ValueError("budget experience character_budget must be positive")
    observations = task_budget_observations(state)
    version = state.get("active_model_version")
    groups = defaultdict(list)
    for row in observations:
        atoms = row["atom_count"]
        if (
            row["status"] != "completed"
            or not atoms
            or not row["model_version"]
            or not row["config_version"]
            or not row["backend"]
            or not row["hardware"]
        ):
            continue
        if version and row["model_version"] != version:
            continue
        if all(
            row[key] is None for key in ("actual_relative_cost", "elapsed_seconds", "gpu_hours")
        ):
            continue
        key = (
            row["stage"],
            row["model_version"],
            row["config_version"],
            row["backend"],
            row["hardware"],
            row["gpu_count"],
            row["cpu_count"],
            math.floor(math.log2(atoms)),
            json.dumps(row["parameters"], sort_keys=True, default=str),
        )
        groups[key].append(row)
    cohorts = []
    # Most recently observed conditions first. Actual values never substitute for a different model.
    for key, rows in reversed(list(groups.items())):
        rows = rows[-30:]
        identity = hashlib.sha256(json.dumps(key, default=str).encode()).hexdigest()[:16]
        cohorts.append(
            {
                "cohort_id": identity,
                "conditions": {
                    field: rows[-1][field]
                    for field in (
                        "stage",
                        "model_version",
                        "config_version",
                        "backend",
                        "hardware",
                        "gpu_count",
                        "cpu_count",
                        "parameters",
                    )
                },
                "atom_count_range": [
                    min(row["atom_count"] for row in rows),
                    max(row["atom_count"] for row in rows),
                ],
                "sample_count": len(rows),
                "observations": {
                    field: _statistics(rows, field)
                    for field in (
                        "actual_relative_cost",
                        "elapsed_seconds",
                        "gpu_hours",
                        "actual_mc_steps",
                    )
                },
                "evidence_refs": [f"task:{row['task_id']}" for row in rows],
                "maturity": "single_observation" if len(rows) == 1 else "repeated_observations",
            }
        )
    all_examples = budget_experiences(state)
    examples = all_examples[-limit:]
    # Task IDs provide drill-down provenance; avoid dumping whole batches into the prompt.
    for example in examples:
        example["task_observations"] = example["task_observations"][-5:]
        example["evidence_refs"] = [f"decision_outcome:{example['decision_id']}"]
        example["observed_rewards"] = example["observed_rewards"][-5:]
        if "generation_plan" in example["parameters"]:
            example["parameters"]["generation_plan"] = example["parameters"]["generation_plan"][:12]
    selected_cohorts, selected_examples, used = [], [], 0
    for values, selected in ((cohorts, selected_cohorts), (reversed(examples), selected_examples)):
        for row in values:
            size = len(json.dumps(row, ensure_ascii=False, default=str))
            if used + size > character_budget:
                continue
            selected.append(row)
            used += size
            if len(selected) == limit:
                break
    return {
        "coverage": {
            "tasks": len(observations),
            **{
                field: sum(row[field] is not None for row in observations)
                for field in ("actual_relative_cost", "elapsed_seconds", "gpu_hours")
            },
        },
        "decision_examples": list(reversed(selected_examples)),
        "comparable_cohorts": selected_cohorts,
        "retrieval": {
            "character_budget": character_budget,
            "used_characters": used,
            "available_cohorts": len(cohorts),
            "included_cohorts": len(selected_cohorts),
            "available_decisions": len(all_examples),
            "included_decisions": len(selected_examples),
        },
        "instruction": "Use these factual observations to compare budgets and workloads within confirmed limits. A null actual_relative_cost means UNKNOWN, never zero or free. If coverage.actual_relative_cost=0 it means zero TASKS with a measured cost, not zero cost. Report this distinction explicitly. Runtime and GPU-hours do not imply relative_cost. Actual MC steps do not establish a stop reason or no-improvement event without an explicit receipt. Compare conditions, sample counts, failures and versioned gains before recommending a batch. Small samples are uncertain. Parameters are observed choices, not learned optima. Knowledge or budget increases require the existing configuration/review path; execution always requires approval.",
    }


def _complete_sum(rows, field):
    return (
        sum(row[field] for row in rows)
        if rows and all(row[field] is not None for row in rows)
        else None
    )


def _statistics(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return {
        "samples": len(values),
        "median": median(values) if values else None,
        "range": [min(values), max(values)] if values else None,
    }


def _finite_number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None

"""Read-only scenario estimates reusing the project's relative-cost model."""

from copy import deepcopy
import math

from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.analysis.cost.calibrate_relative_cost import calibrate_relative_cost
from phase_agent.analysis.cost.predict_runtime import predict_runtime


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def estimate_task_cost(
    stage,
    *,
    atom_count=None,
    budgets=None,
    state=None,
    backend=None,
    hardware=None,
    initial_state_count=1,
    patience=None,
    max_mc_steps=None,
    expected_mc_steps=None,
    relax_steps=None,
    reference_relax_steps=None,
    kpoint_count=None,
    reference_kpoint_count=None,
    electronic_steps=None,
    reference_electronic_steps=None,
):
    """Optional workload multipliers require both actual and reference values.

    Patience-only MC is a no-improvement scenario, never expected runtime.
    History calibration is backend-scoped when a backend is supplied.
    """
    if stage not in {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}:
        raise ValueError("unsupported cost stage")
    multiplier = 1.0
    assumptions = []
    for value, reference, name in (
        (relax_steps, reference_relax_steps, "relax_steps"),
        (kpoint_count, reference_kpoint_count, "kpoints"),
        (electronic_steps, reference_electronic_steps, "electronic_steps"),
    ):
        if value is not None or reference is not None:
            if value is None or reference is None:
                raise ValueError(f"{name} requires explicit workload and reference")
            multiplier *= _positive(value, name) / _positive(reference, "reference_" + name)
            assumptions.append({"parameter": name, "value": value, "reference": reference})
    factor, samples = calibrate_relative_cost(state or {}, stage, backend=backend)
    scenarios = [("configured_reference_workload", None)]
    if stage == "deep_search":
        maximum = _positive(max_mc_steps, "max_mc_steps")
        scenarios = [("maximum_steps", maximum)]
        if patience is not None:
            p = _positive(patience, "patience")
            if p > maximum:
                raise ValueError("patience cannot exceed max_mc_steps")
            scenarios.insert(0, ("no_improvement_patience_scenario", p))
        if expected_mc_steps is not None:
            expected = _positive(expected_mc_steps, "expected_mc_steps")
            if expected > maximum:
                raise ValueError("expected_mc_steps cannot exceed maximum")
            scenarios.insert(0, ("user_assumed_steps", expected))
    rows = []
    for name, steps in scenarios:
        estimate = estimate_stage_cost(
            stage,
            atom_count=atom_count,
            initial_state_count=initial_state_count,
            mc_steps=steps,
            budgets=budgets,
        )
        initial = estimate["value"] * multiplier
        timing = predict_runtime(
            stage,
            atom_count=atom_count,
            state=state,
            backend=backend,
            hardware=hardware,
            mc_steps=steps,
            patience=patience,
            maximum=max_mc_steps,
        )
        if timing.get("elapsed_seconds") is not None:
            timing["elapsed_seconds"] *= multiplier * initial_state_count
            timing["sample_range_seconds"] = [
                v * multiplier * initial_state_count for v in timing["sample_range_seconds"]
            ]
            for field in ("gpu_hours", "cpu_core_hours"):
                if timing.get(field) is not None:
                    timing[field] *= multiplier * initial_state_count
        rows.append(
            {
                "scenario": name,
                "mc_steps": steps,
                "initial_cost": initial,
                "estimated_cost": initial * factor,
                "runtime_estimate": timing,
            }
        )
    if stage == "deep_search":
        timing = rows[-1]["runtime_estimate"]
        typical = timing.get("typical_mc_steps")
        if typical is not None and expected_mc_steps is None:
            historical = deepcopy(rows[-1])
            historical["scenario"] = "historical_typical_steps"
            historical["mc_steps"] = typical
            ratio = typical / float(max_mc_steps)
            for field in ("initial_cost", "estimated_cost"):
                historical[field] *= ratio
            for field in ("elapsed_seconds", "gpu_hours", "cpu_core_hours"):
                if historical["runtime_estimate"].get(field) is not None:
                    historical["runtime_estimate"][field] *= ratio
            historical["runtime_estimate"]["sample_range_seconds"] = [
                v * ratio for v in historical["runtime_estimate"]["sample_range_seconds"]
            ]
            rows.insert(0, historical)
    return {
        "stage": stage,
        "backend": backend,
        "atom_count": atom_count,
        "initial_state_count": initial_state_count,
        "patience": patience,
        "max_mc_steps": max_mc_steps,
        "unit": "relative_cost",
        "scenarios": rows,
        "calibration_factor": factor,
        "measured_samples": samples,
        "quality": "historical_runtime_estimate"
        if any(row["runtime_estimate"].get("samples") for row in rows)
        else "historical_ratio_calibrated"
        if samples
        else "initial_rough_estimate",
        "workload_assumptions": assumptions,
        "wall_time": None,
        "core_hours": None,
        "primary_cost_basis": "scenario_runtime_estimate"
        if any(row["runtime_estimate"].get("samples") for row in rows)
        else "initial_relative_cost_reference",
        "unit_policy": "Measured elapsed time/core-hours are primary physical costs. Relative-cost budget is a separate reference unit; no automatic seconds-to-budget conversion.",
        "limitations": "Patience is consecutive non-improving steps, not total steps. Workload multipliers are rough linear scenarios. No measured time conversion is assumed.",
    }


def estimate_batch_cost(tasks, *, budgets=None, state=None):
    reports = []
    for task in tasks:
        spec = deepcopy(task)
        identifier = spec.pop("task_id", None)
        report = estimate_task_cost(**spec, budgets=budgets, state=state)
        reports.append({"task_id": identifier, **report})
    timings = [report["scenarios"][-1]["runtime_estimate"] for report in reports]
    complete_timing = bool(timings) and all(
        row.get("elapsed_seconds") is not None for row in timings
    )
    return {
        "task_count": len(reports),
        "unit": "relative_cost",
        "tasks": reports,
        "serial_maximum_runtime_seconds": sum(row["elapsed_seconds"] for row in timings)
        if complete_timing
        else None,
        "runtime_predicted_tasks": sum(row.get("elapsed_seconds") is not None for row in timings),
        "maximum_scenario_total": sum(
            max(row["estimated_cost"] for row in report["scenarios"]) for report in reports
        ),
        "lowest_scenario_total": sum(
            min(row["estimated_cost"] for row in report["scenarios"]) for report in reports
        ),
        "note": "Scenario totals are not statistical confidence bounds or guaranteed early-stop savings.",
    }

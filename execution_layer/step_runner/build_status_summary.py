"""Build a lightweight factual summary without running scientific analysis."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from execution_layer.step_runner.file_protocol import content_id


def build_status_summary(state: dict, *, config_version=None) -> dict:
    tasks = state.get("tasks") or []
    jobs = state.get("slurm_batches") or []
    summary = {
        "config_version": config_version or state.get("confirmed_config_version"),
        "model_version": state.get("active_model_version"),
        "hull_version": state.get("active_hull_version"),
        "task_counts": dict(Counter(row.get("status", "unknown") for row in tasks)),
        "task_stage_counts": dict(Counter(row.get("stage", "unknown") for row in tasks)),
        "job_counts": dict(Counter(row.get("status", "unknown") for row in jobs)),
        "budget_usage": deepcopy(state.get("budget_usage") or {}),
        "reserved_relative_cost": float(state.get("reserved_relative_cost", 0) or 0),
        "budget_remaining": state.get("budget_remaining"),
        "coverage": deepcopy(state.get("coverage_summary")),
        "phase_diagram": deepcopy(state.get("phase_diagram_summary")),
        "qbc": deepcopy(state.get("qbc_summary")),
        "convergence": deepcopy(state.get("convergence_result")),
        "pending_task_ids": [row.get("task_id") for row in tasks if row.get("status") == "pending"],
        "failed_task_ids": [row.get("task_id") for row in tasks if row.get("status") == "failed"],
        "recent_results": deepcopy((state.get("recent_results") or state.get("recovered_results") or [])[-10:]),
        "candidate_ids": [row.get("candidate_id") or row.get("structure_id")
                          for row in (state.get("candidates") or state.get("branch_candidates") or [])[:50]],
        "model_validation": deepcopy(state.get("mlip_validation") or state.get("model_validation_summary")),
        "evidence_gaps": deepcopy(state.get("evidence_gaps") or state.get("coverage_gaps") or []),
        "pending_approvals": len(state.get("pending_execution_policies") or {}),
        "memory_reviews": len([row for row in state.get("memory_review_queue") or []
                               if row.get("status") in {"pending_review", "conflict_review"}]),
    }
    summary["summary_id"] = content_id(summary, "summary")
    return summary

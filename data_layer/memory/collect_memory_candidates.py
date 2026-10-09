"""Accumulate factual search observations without promoting them to advice."""

from copy import deepcopy
import hashlib
import json


def collect_memory_candidates(state):
    updated = dict(state)
    pool = deepcopy(state.get("memory_candidates") or [])
    updated["memory_candidates"] = pool
    known = {row.get("candidate_id") for row in pool}
    model = updated.get("active_model_version")
    config = updated.get("confirmed_config_version")
    from analysis_layer.state.training_result_evidence import training_result_evidence
    from analysis_layer.state.round_budget_evidence import budget_decision_outcomes
    updated["training_result_reports"] = training_result_evidence(updated)
    sources = []
    for index, outcome in enumerate(updated.get("training_handoff_history") or []):
        sources.append(("training_handoff", str(index), outcome, tuple(outcome)))
    for report in updated["training_result_reports"]:
        if report.get("report_id"):
            source = {**report, "model_version": report.get("source_model_version")}
            sources.append(("training_report", report["report_id"], source, tuple(source)))
    for outcome in budget_decision_outcomes(updated):
        if outcome.get("decision_id"):
            sources.append(("round_budget_outcome", outcome["decision_id"], outcome, tuple(outcome)))
    tasks_by_id = {row.get("task_id"): row for row in updated.get("tasks") or [] if row.get("task_id")}
    for row in updated.get("action_records") or []:
        action = row.get("final_action") or {}
        execution = row.get("execution_result") or {}
        if not isinstance(execution, dict):
            continue
        result = execution.get("result") or {}
        if not row.get("record_id") or not action or not isinstance(result, dict):
            continue
        # Only explicit IDs returned by this action establish provenance.
        linked = [tasks_by_id[item["task_id"]] for item in result.get("tasks") or []
                  if isinstance(item, dict) and item.get("task_id") in tasks_by_id]
        if not linked:
            continue
        summary = {"record_id": row["record_id"], "decision": deepcopy(action),
                   "config_version": row.get("config_version"),
                   "model_version": next(iter({task.get("model_version") for task in linked}))
                                    if len({task.get("model_version") for task in linked}) == 1 else None,
                   "expected_relative_cost": action.get("estimated_cost"),
                   "tasks": [{key: deepcopy(task.get(key)) for key in
                              ("task_id", "stage", "status", "model_version", "actual_cost")}
                             for task in linked],
                   "outcome_status": "complete" if all(task.get("status") == "completed" for task in linked)
                                     else "partial",
                   "causality": "explicit_action_task_ids_only"}
        sources.append(("decision_outcome", row["record_id"], summary, tuple(summary)))
    for row in updated.get("rewards") or []:
        if row.get("batch_id") and row.get("energy_basis_id") and row.get("reward") is not None:
            sources.append(("reward", row["batch_id"], row,
                            ("reward", "actual_cost", "energy_method", "energy_basis_id",
                             "previous_version", "current_version")))
    for row in updated.get("model_update_epochs") or []:
        if row.get("model_version") and row.get("hull_change") is not None:
            sources.append(("model_epoch", row["model_version"], row,
                            ("model_version", "hull_change", "ground_state_unchanged")))
    for row in updated.get("tasks") or []:
        if row.get("task_id") and row.get("status") in {"failed", "timeout"}:
            sources.append(("failed_task", row["task_id"], row,
                            ("task_id", "stage", "status", "model_version", "failure_reason")))
    completed_by_batch = {}
    for task in updated.get("tasks") or []:
        if task.get("batch_id") and task.get("status") == "completed":
            completed_by_batch.setdefault(task["batch_id"], []).append(task)
    for batch in updated.get("slurm_batches") or []:
        batch_id = batch.get("batch_id")
        if not batch_id:
            continue
        completed = completed_by_batch.get(batch_id) or []
        if completed:
            summary = {"batch_id": batch_id, "stage": completed[0].get("stage"),
                       "completed_count": len(completed),
                       "model_version": batch.get("model_version"),
                       "actual_cost_known_count": sum(task.get("actual_cost") is not None for task in completed)}
            sources.append(("completed_batch", batch_id, summary, tuple(summary)))
    for kind, reference, source, fields in sources:
        facts = {key: deepcopy(source.get(key)) for key in fields if source.get(key) is not None}
        payload = {"kind": kind, "reference": str(reference), "facts": facts,
                   "config_version": source.get("config_version", config),
                   "model_version": source.get("model_version", model)}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        candidate_id = "MCAND-" + digest
        if candidate_id not in known:
            pool.append({"candidate_id": candidate_id, "status": "candidate",
                         "evidence_refs": [f"{kind}:{reference}"], **payload})
            known.add(candidate_id)
    return updated

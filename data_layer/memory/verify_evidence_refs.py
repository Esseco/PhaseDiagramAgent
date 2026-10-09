"""Check that cited local evidence exists before accepting durable knowledge."""


def verify_evidence_refs(state, refs, *, scope):
    indexes = {
        "training_report": {str(row.get("report_id")) for row in state.get("training_result_reports") or []},
        "round_budget_outcome": {str(row.get("record_id")) for row in state.get("action_records") or []
                                 if (row.get("final_action") or {}).get("round_budget_review")},
        "decision_outcome": {str(row.get("record_id")) for row in state.get("action_records") or []},
        "task": {str(row.get("task_id")) for row in state.get("tasks") or []},
        "failed_task": {str(row.get("task_id")) for row in state.get("tasks") or []
                        if row.get("status") in {"failed", "timeout"}},
        "batch": {str(row.get("batch_id")) for row in state.get("slurm_batches") or []},
        "completed_batch": {str(row.get("batch_id")) for row in state.get("slurm_batches") or []},
        "reward": {str(row.get("batch_id")) for row in state.get("rewards") or []},
        "model_epoch": {str(row.get("model_version")) for row in state.get("model_update_epochs") or []},
        "candidate": {str(row.get("candidate_id")) for row in state.get("memory_candidates") or []},
    }
    missing = []
    for ref in refs:
        kind, separator, identifier = ref.partition(":")
        if kind == "user" and scope == "user" and separator and identifier:
            continue
        if not separator or identifier not in indexes.get(kind, set()):
            missing.append(ref)
    return missing

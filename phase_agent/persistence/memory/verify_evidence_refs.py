"""Check that cited local evidence exists before accepting durable knowledge."""


def verify_evidence_refs(state, refs, *, scope):
    indexes = {
        "training_handoff": {
            str(i) for i, _ in enumerate(state.get("training_handoff_history") or [])
        },
        "training_report": {
            str(row.get("report_id")) for row in state.get("training_result_reports") or []
        },
        "round_budget_outcome": {
            str(row.get("record_id"))
            for row in state.get("action_records") or []
            if (row.get("final_action") or {}).get("round_budget_review")
        },
        "budget_experience": {
            str(row.get("record_id")) for row in state.get("action_records") or []
        },
        "decision_outcome": {
            str(row.get("record_id")) for row in state.get("action_records") or []
        },
        "task": {str(row.get("task_id")) for row in state.get("tasks") or []},
        "failed_task": {
            str(row.get("task_id"))
            for row in state.get("tasks") or []
            if row.get("status") in {"failed", "timeout"}
        },
        "batch": {str(row.get("batch_id")) for row in state.get("slurm_batches") or []},
        "completed_batch": {
            str(row.get("batch_id"))
            for row in state.get("slurm_batches") or []
            if _completed_batch(state, row)
        },
        "reward": {str(row.get("batch_id")) for row in state.get("rewards") or []},
        "model_epoch": {
            str(row.get("model_version")) for row in state.get("model_update_epochs") or []
        },
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


def _completed_batch(state, batch):
    if batch.get("status") in {"completed", "finished", "recovered"}:
        return True
    tasks = [
        row for row in state.get("tasks") or [] if row.get("batch_id") == batch.get("batch_id")
    ]
    # Historical batches may omit status; their completed child tasks remain evidence.
    return (
        not batch.get("status")
        and bool(tasks)
        and all(row.get("status") == "completed" for row in tasks)
    )


def evidence_fingerprint(state, refs):
    from phase_agent.tools.step_runner.file_protocol import content_id

    collections = {
        "task": "tasks",
        "failed_task": "tasks",
        "batch": "slurm_batches",
        "completed_batch": "slurm_batches",
        "training_report": "training_result_reports",
        "training_handoff": "training_handoff_history",
        "model_epoch": "model_update_epochs",
        "candidate": "memory_candidates",
        "reward": "rewards",
        "round_budget_outcome": "action_records",
        "budget_experience": "action_records",
        "decision_outcome": "action_records",
    }
    fields = {
        "task": "task_id",
        "failed_task": "task_id",
        "batch": "batch_id",
        "completed_batch": "batch_id",
        "training_report": "report_id",
        "model_epoch": "model_version",
        "candidate": "candidate_id",
        "reward": "batch_id",
    }
    material = {}
    for ref in refs:
        kind, _, identifier = ref.partition(":")
        rows = state.get(collections.get(kind, "")) or []
        material[ref] = (
            [row for index, row in enumerate(rows) if str(index) == identifier]
            if kind == "training_handoff"
            else [row for row in rows if str(row.get(fields.get(kind, "record_id"))) == identifier]
        )
    return content_id(material, "memory-evidence")

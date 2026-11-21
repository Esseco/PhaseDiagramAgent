"""Resolve only cited evidence collections, once per decision or memory update."""

COLLECTIONS = {
    "task": ("tasks", "task_id"),
    "failed_task": ("tasks", "task_id"),
    "batch": ("slurm_batches", "batch_id"),
    "completed_batch": ("slurm_batches", "batch_id"),
    "training_report": ("training_result_reports", "report_id"),
    "training_handoff": ("training_handoff_history", None),
    "model_epoch": ("model_update_epochs", "model_version"),
    "candidate": ("memory_candidates", "candidate_id"),
    "reward": ("rewards", "batch_id"),
    "round_budget_outcome": ("action_records", "record_id"),
    "budget_experience": ("action_records", "record_id"),
    "decision_outcome": ("action_records", "record_id"),
}


class EvidenceResolver:
    def __init__(self, state):
        self.state = state
        self.indexes = {}

    def rows(self, ref):
        kind, _, identifier = ref.partition(":")
        if kind not in COLLECTIONS:
            return []
        collection, field = COLLECTIONS[kind]
        key = (collection, field)
        if key not in self.indexes:
            index = {}
            for number, row in enumerate(self.state.get(collection) or []):
                identity = str(number) if field is None else str(row.get(field))
                index.setdefault(identity, []).append(row)
            self.indexes[key] = index
        return self.indexes[key].get(identifier, [])

    def missing(self, refs, *, scope):
        missing = []
        for ref in refs:
            kind, separator, identifier = ref.partition(":")
            if kind == "user" and scope == "user" and separator and identifier:
                continue
            rows = self.rows(ref)
            if (
                not separator
                or not rows
                or any(_invalid(row) for row in rows)
                or (
                    kind == "failed_task"
                    and not all(row.get("status") in {"failed", "timeout"} for row in rows)
                )
                or (
                    kind == "completed_batch"
                    and not all(_completed_batch(self.state, row) for row in rows)
                )
                or (
                    kind == "round_budget_outcome"
                    and not all(
                        (row.get("final_action") or {}).get("round_budget_review") for row in rows
                    )
                )
            ):
                missing.append(ref)
        return missing

    def fingerprint(self, refs):
        from phase_agent.tools.step_runner.file_protocol import content_id

        return content_id({ref: self.rows(ref) for ref in refs}, "memory-evidence")


def _invalid(row):
    return (
        row.get("status") in {"quarantined", "invalid", "retracted", "superseded", "deprecated"}
        or row.get("validity") in {"stale", "invalid", "quarantined"}
        or row.get("status") == "invalid_report"
        or row.get("valid") is False
        or row.get("quarantined") is True
    )


def verify_evidence_refs(state, refs, *, scope):
    return EvidenceResolver(state).missing(refs, scope=scope)


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
    return EvidenceResolver(state).fingerprint(refs)


def revalidation_reason(record, resolver):
    refs = record.get("evidence_refs") or []
    if not refs:
        return None
    if resolver.missing(refs, scope=record.get("scope")):
        return "evidence_missing_or_invalid"
    bound = record.get("evidence_fingerprint")
    if bound and bound != resolver.fingerprint(refs):
        return "evidence_revision_changed"
    if not bound and record.get("maturity") == "validated":
        return "legacy_evidence_version_unbound"
    return None


def revalidate_memory(state):
    """Called only on an already-owned normal save; retrieval never writes state."""
    memory = state.get("decision_memory") or {}
    resolver = EvidenceResolver(state)
    changed = []
    for row in memory.get("records") or []:
        if row.get("status") != "active":
            continue
        reason = revalidation_reason(row, resolver)
        if reason:
            row["status"] = "needs_revalidation"
            row["revalidation_reason"] = reason
            changed.append(row["knowledge_id"])
    if changed:
        memory["version"] = int(memory.get("version", 0)) + 1
        memory.setdefault("history", []).append(
            {
                "version": memory["version"],
                "needs_revalidation": changed,
                "source": "evidence_revalidation",
            }
        )
    return changed

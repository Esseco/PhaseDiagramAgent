"""Restore local dedup evidence from completed generation, never from labels alone."""
from copy import deepcopy


def restore_generation_gate(state, manager):
    current = deepcopy(state)
    if current.get("dedup_gate") is not None:
        return current
    records = manager.data.get("structures") or {}
    valid = set()
    for batch in current.get("generation_history") or []:
        summary = batch.get("summary") or {}
        # These counts are emitted only after actual structure dedup/register.
        if "unique_structures" not in summary or "registered_structures" not in summary:
            continue
        for sid in batch.get("registered_ids") or []:
            if sid in records:
                valid.add(sid)
    if valid:
        current["dedup_gate"] = {"status": "ready", "source": "local_generation_dedup",
                                 "valid_structure_ids": sorted(valid)}
    return current

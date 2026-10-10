"""Select cumulative accepted records without introducing scientific deduplication."""

from copy import deepcopy

from phase_agent.science.dft.spin_acceptance import spin_standard_passed


def cumulative_training_records(state):
    new = state.get("new_dft_records") or []
    new_ids = {r.get("data_id") for r in new if r.get("data_id")}
    rows, seen = [], set()
    for record in [*(state.get("dft_training_records") or []), *new]:
        identity = record.get("data_id")
        if identity and identity in seen:
            continue  # The same ledger record is mirrored in the new-data queue.
        if record.get("checks_passed") is not True or not spin_standard_passed(record):
            continue
        row = deepcopy(record)
        row["is_current_round"] = identity in new_ids if identity else record in new
        rows.append(row)
        if identity:
            seen.add(identity)
    return rows
